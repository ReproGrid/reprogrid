"""Run the reconstructed 0.6 s load sequence, source-window checks and FE sweep."""

from argparse import ArgumentParser
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile
from time import perf_counter

import numpy as np

from deep_metrics import SIGNALS, full_network_checks, load_curves, paper_and_cases
from deep_plots import fault, figure34, overview, panels, stress
from reproduce import CASE_FIELDS, Trial, invoke
from results import compare, describe


ROOT = Path(__file__).resolve().parents[1]
FLAGS = ['-std=c++17', '-O3', '-Wall', '-Wextra', '-Wpedantic', '-ffp-contract=off']


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def invoke_schedule(executable, trial, protocol, motor, resistance, directory, format_override=None):
    t, gate, q = (protocol[key] for key in ('timing_ps', 'gate_schedule', 'fixed_point_trial'))
    q = format_override or q
    output = directory / trial['id']
    args = [executable, output, *(motor[k] for k in CASE_FIELDS), resistance['on'], resistance['off'],
            trial.get('rectifier_ron_ohm', 0), trial['short_ohm'], trial['step_ps'],
            *(t[k] for k in ('stop', 'fault_start', 'fault_end', 'ramp_start', 'ramp_end', 'frequency_hold',
                              'clock', 'overview_stride', 'transition_stride')),
            gate['initial_half_period_ticks'], gate['final_half_period_ticks'], int(trial['phase'] == 'reset_at_change'),
            int(trial.get('verify', False)), int(trial.get('pipeline', False)),
            *(q[k] for k in ('vector_signed_bits', 'vector_fraction_bits', 'coefficient_signed_bits', 'coefficient_fraction_bits'))]
    run = subprocess.run(list(map(str, args)), capture_output=True, text=True)
    if run.returncode:
        raise RuntimeError(f"{trial['id']}: {run.stderr.strip()}")
    files = {trial['id']: output}
    if trial.get('pipeline', False):
        files.update({'single_stage_unaligned': output.with_suffix('.stage'), 'fixed_point': output.with_suffix('.fixed')})
    data = {name: np.fromfile(path).reshape(-1, 11) for name, path in files.items()}
    for path in files.values():
        path.unlink()
    return data, json.loads(run.stdout)


def table_audit(protocol):
    table2, table3 = protocol['table_ii_published'], protocol['table_iii_published']
    rows = {}
    for name in ('FPI', 'HRT'):
        t2, t3 = table2[name], table3[name]
        exact_resources = [clock for i, clock in enumerate(table2['clock_mhz'])
                           if all(t2[k][i] == t3[k] for k in ('step_ns', 'lut', 'registers'))]
        rows[name] = {'table_ii_clock_mhz': table2['clock_mhz'],
                      'table_ii_cycles_per_step': [clock * dt / 1000 for clock, dt in zip(table2['clock_mhz'], t2['step_ns'])],
                      'table_ii_steps_of_io_delay': [delay / dt for delay, dt in zip(t2['io_ns'], t2['step_ns'])],
                      'table_iii_clock_label_mhz': t3['clock_label_mhz'],
                      'table_ii_matching_clocks_mhz': exact_resources, 'normalized_dsp_fraction_pct_of_840': 100*t3['dsp']/840,
                      'table_iii_printed_dsp_percent': t3['dsp_percent']}
    return rows


def test_schedule(trial, values, wave, protocol, motor, resistance):
    t, thresholds = protocol['timing_ps'], protocol['validation']
    complete = wave[-1, 0] == t['stop'] and values['steps'] == t['stop'] // trial['step_ps']
    events_on_time = values['first_fault_tick_ps'] == t['fault_start'] and values['first_clear_tick_ps'] == t['fault_end']
    during = wave[(wave[:, 0] >= t['fault_start']) & (wave[:, 0] < t['fault_end'])]
    trace_frequency = np.unique(wave[:, 9])
    g = protocol['gate_schedule']
    expected_frequency = 1e12 / (2 * t['clock'] * np.arange(g['final_half_period_ticks'], g['initial_half_period_ticks'] + 1))
    result = {
        'full_time_sequence': bool(complete and events_on_time), 'finite_waveform': bool(np.all(np.isfinite(wave))),
        'all_commanded_frequency_levels': bool(np.array_equal(np.sort(expected_frequency), trace_frequency)),
        'current_balance': values['max_relative_kcl'] < thresholds['kcl_relative'],
        'diode_polarity': values['max_diode_polarity_violation_v'] < thresholds['diode_voltage_v'],
        'output_recovers_before_frequency_ramp': t['fault_end'] < values['first_recovery_90pct_ps'] < t['ramp_start'],
    }
    if trial['short_ohm'] == 0:
        stored_energy = motor['co_f'] * values['prefault_voltage_v']**2 / 2
        result['ideal_short_constraint'] = bool(np.all(during[:, 4] == 0))
        result['capacitor_energy_removed'] = abs(values['ideal_capacitor_energy_removed_j'] - stored_energy) < 1e-12
    if trial.get('verify'):
        segment = values['float_nodal_segments'][-1]
        result['reference_discrete_trajectory'] = (segment['max_abs'][0] < thresholds['max_voltage_delta_v'] and
                                                   max(segment['max_abs'][1:]) < thresholds['max_current_delta_a'])
        result['reference_state_agreement'] = values['nodal_mode_mismatches_outside_tie'] == 0
        network = full_network_checks(wave, motor, resistance, trial)
        values['full_network_sampled'] = network
        result['full_physical_switch_network'] = (network['checked_fault_edges'] == 2 and
                                                  max(network['state_max_abs_delta']) < thresholds['max_voltage_delta_v'] and
                                                  network['diode_states_changed_outside_tie'] == 0)
    if trial.get('pipeline'):
        result['rearranged_single_stage_values'] = max(values['single_stage_segments'][-1]['max_abs']) < thresholds['single_stage_max_error_a_or_v']
        result['rearranged_single_stage_modes'] = values['single_stage_mode_mismatches_outside_tie'] == 0
        result['fixed_point_dynamic_range'] = values['fixed_vector_saturations'] == values['fixed_coefficient_saturations'] == 0
    return result


def stress_runs(binary, protocol, base, work):
    opts = protocol['figure_3_4_detail']
    sources = base['runs']
    r = base['unreported_inputs']['baseline_switch_ohm']
    resistance = (r['on'], r['off'])
    configurations = {'set1': [opts['frequencies_hz']['set1']], 'set2': opts['stress_set2_hz']}
    out, waves = {}, {}
    for set_key, frequencies in configurations.items():
        for freq in frequencies:
            prefix = f'{set_key}_{freq}'
            nominal = opts['step_ps']
            jobs = [(nominal, 'dmm')]
            if set_key == 'set2':
                jobs += [(step, 'forward_pairs') for step in opts['stress_step_ps']]
            if freq == opts['frequencies_hz'][set_key]:
                jobs += [(nominal, 'backward_pairs'), (nominal, 'forward_full')]
                if set_key == 'set1':
                    jobs.append((nominal, 'forward_pairs'))
            for step, method in jobs:
                name = f'{prefix}_{step}_{method}'
                waveform, state, _ = invoke(binary, Trial(name, set_key, freq, method, step, resistance), base, work)
                if method == 'dmm':
                    reference = waveform
                data = describe(waveform, state, sources['evaluate_tail_from_ps'])
                rel = compare(reference, waveform, sources['evaluate_tail_from_ps'], blocking_reference=True)
                out[name] = {'method': method, 'frequency_hz': freq, 'step_ps': step, 'status': data['status'], 'end_s': data['end_s'],
                             'stop_state': data['stop_state'], 'summary': data['late_window'],
                             'ir_nrmse_pct': rel['ir_nrmse_pct'] if rel is not None else None}
                if freq == opts['frequencies_hz'][set_key]:
                    waves[name] = waveform
    return out, waves


def main():
    cli = ArgumentParser(description=__doc__)
    cli.add_argument('--out', type=Path, required=True)
    cli.add_argument('--figures', type=Path)
    cli.add_argument('--compiler', default=os.environ.get('CXX', 'c++'))
    cli.add_argument('--source-curves', type=Path, default=ROOT / 'verification/paper_curves')
    args = cli.parse_args()
    started, clock = datetime.now(timezone.utc), perf_counter()
    p_file = ROOT / 'configs/deep_protocol.json'
    protocol = read(p_file)
    old_file = p_file.with_name(protocol['base_protocol'])
    base = read(old_file)
    args.out.mkdir(parents=True, exist_ok=True)
    figure_root = args.figures or args.out / 'figures'
    figure_root.mkdir(exist_ok=True, parents=True)
    source = load_curves(args.source_curves)
    readback = read(args.source_curves / 'readback_manifest.json')
    motor = base['table_i']['set2']
    device = base['unreported_inputs']['baseline_switch_ohm']
    waves, data, checks = {}, {}, {}
    with tempfile.TemporaryDirectory(prefix='.llc-analysis-', dir=args.out) as tmp:
        work = Path(tmp)
        replay, fixed = work / 'sequence', work / 'fixed'
        for source_code, exe in [('deep_simulate.cpp', replay), ('simulate.cpp', fixed)]:
            subprocess.run([args.compiler, *FLAGS, ROOT / 'src' / source_code, '-o', exe], check=True)
        for trial in protocol['cases']:
            name = trial['id']
            print(name, flush=True)
            group, result = invoke_schedule(replay, trial, protocol, motor, device, work)
            checks[name] = test_schedule(trial, result, group[name], protocol, motor, device)
            if trial.get('pipeline'):
                dt = trial['step_ps']
                delayed = group['single_stage_unaligned']
                compensated_t = delayed[:, 0] - result['assumed_io_delay_steps'] * dt
                checks[name]['pipeline_time_alignment'] = bool(np.array_equal(compensated_t, group[name][:, 0]))
            waves.update(group)
            data[name] = result
        q = protocol['fixed_point_trial']
        original = next(trial for trial in protocol['cases'] if trial['id'] == q['case'])
        alternate = original | {'id': 'table_iii_fixed_format', 'verify': False}
        print(alternate['id'], flush=True)
        group, alternate_result = invoke_schedule(replay, alternate, protocol, motor, device, work,
                                                  format_override=q['table_iii_variant'])
        float_repeat = group[alternate['id']]
        checks[alternate['id']] = {
            'same_float_trajectory': bool(np.array_equal(float_repeat, waves[original['id']])),
            'no_coefficient_or_vector_saturation': alternate_result['fixed_coefficient_saturations'] == alternate_result['fixed_vector_saturations'] == 0,
        }
        waves['fixed_point_table_iii'] = group['fixed_point']
        print('Forward Euler stress sweep', flush=True)
        stress_data, stress_waves = stress_runs(fixed, protocol, base, work)
    checks['source_schedule'] = {
        'frequency_readback_within_100_hz': readback['plateau_check']['max_level_mismatch_khz'] < 0.1,
        'change_time_readback_within_0p1_ms': readback['plateau_check']['max_transition_time_error_s'] < 1e-4,
    }
    output = {'column_order': ['time_ps', 'ir_a', 'im_a', 'vcr_v', 'vo_v', 'vx_v', 'vy_v', 'diode_mask',
                               'gate_mask', 'switch_frequency_hz', 'fault_branch_active'],
              'error_signal_order': SIGNALS, 'cases': data, 'source_comparison': paper_and_cases(waves, source, protocol),
              'stress_sweep': stress_data, 'published_table_iv_percent': protocol['table_iv_published_percent_order_vo_ir_im'],
              'fixed_point_table_iii_variant': alternate_result, 'hardware_table_audit': table_audit(protocol),
              'checks': checks}
    save(args.out / 'deep_metrics.json', output)
    np.savez_compressed(args.out / 'sequence_trajectories.npz', **waves)
    np.savez_compressed(args.out / 'explicit_diagnostics.npz', **stress_waves)
    figures = ['full_sequence.png', 'fig11_waveform_comparison.png', 'short_and_recovery.png', 'explicit_step_sweep.png',
               'fig3_4_waveform_comparison.png']
    overview(figure_root / figures[0], waves['ideal_continuous'], source)
    panels(figure_root / figures[1], waves, source, protocol)
    fault(figure_root / figures[2], waves)
    stress(figure_root / figures[3], stress_data, protocol)
    figure34(figure_root / figures[4], stress_waves, source, protocol)
    checked = [value for group in checks.values() for value in group.values()]
    ended = datetime.now(timezone.utc)
    duration = perf_counter() - clock
    run = {'doi': protocol['doi'], 'started_utc': started.isoformat(), 'finished_utc': ended.isoformat(),
           'wall_seconds': duration, 'timing_basis': 'perf_counter including C++ compilation, runs and output',
           'protocol_sha256': digest(p_file), 'base_protocol_sha256': digest(old_file),
           'python': platform.python_version(), 'numpy': np.__version__,
           'compiler': subprocess.check_output([args.compiler, '--version'], text=True).splitlines()[0],
           'checks_passed': sum(checked), 'checks_total': len(checked),
           'implementation_sha256': {name: digest(ROOT / 'src' / name) for name in (
               'deep_reproduce.py', 'deep_simulate.cpp', 'deep_metrics.py', 'deep_plots.py', 'fixed.hpp', 'reference.hpp',
               'llc.hpp', 'simulate.cpp', 'network_check.py', 'plots.py', 'results.py', 'reproduce.py', 'read_paper_curves.py')},
           'source_curve_sha256': {name.name: digest(name) for name in sorted(args.source_curves.iterdir())
                                   if name.suffix == '.csv' or name.name == 'readback_manifest.json'},
           'data_sha256': {n: digest(args.out / n) for n in ('deep_metrics.json', 'sequence_trajectories.npz', 'explicit_diagnostics.npz')},
           'figures_sha256': {n: digest(figure_root / n) for n in figures}}
    save(args.out / 'run_manifest.json', run)
    print(f'{sum(checked)}/{len(checked)} checks passed in {duration:.2f} s', flush=True)
    return 0 if all(checked) else 1


if __name__ == '__main__':
    raise SystemExit(main())
