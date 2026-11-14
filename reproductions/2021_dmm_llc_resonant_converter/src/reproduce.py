"""Compile and replay the LLC switching benchmark, independent reference and sensitivities."""

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from math import pi, sqrt
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
from time import perf_counter

import numpy as np

from network_check import compare as compare_full_network
from results import compare, describe, ramps, verify
from plots import figures


ROOT = Path(__file__).resolve().parents[1]
CASE_FIELDS = ('lr_h', 'lm_h', 'cr_f', 'co_f', 'load_ohm', 'ratio_primary_secondary', 'input_v')


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def put(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


@dataclass(frozen=True)
class Trial:
    name: str
    set_key: str
    hz: int
    method: str
    dt_ps: int
    resistance: tuple[float, float]
    vo0: float = 0.0
    gate_clock_ps: int = 0
    schedule: dict | None = None


def plan(p):
    run, missing = p['runs'], p['unreported_inputs']
    dt = run['nominal_step_ps']
    resistance = tuple(missing['baseline_switch_ohm'][key] for key in ('on', 'off'))
    trials, comparisons, mains = [], {}, []
    for set_key, freqs in p['frequency_hz'].items():
        for hz in (freqs['primary'], *freqs['sensitivity']):
            name = f'{set_key}_{hz}'
            certified = hz == freqs['primary'] or (set_key == 'set2' and hz == 500000)
            main = name + '_be'
            trials.append(Trial(main, set_key, hz, 'verified' if certified else 'dmm', dt, resistance))
            if certified:
                mains.append(main)
            for label, method in (('be_pairs', 'backward_pairs'), ('fe_pairs', 'forward_pairs')):
                short = name + '_' + label
                trials.append(Trial(short, set_key, hz, method, dt, resistance))
                comparisons[short] = (main, short, True)
            coarse = name + '_fe_pairs_30ns'
            trials.append(Trial(coarse, set_key, hz, 'forward_pairs', run['extra_steps_ps'][-1], resistance))
            comparisons[coarse] = (main, coarse, True)
            if set_key == 'set1' and hz == 160000:
                trials.append(Trial(name + '_precharged', set_key, hz, 'dmm', dt, resistance,
                                    vo0=p['table_i'][set_key]['rated_output_v']))
            if hz == freqs['primary']:
                for card in missing['sensitivity_switch_ohm']:
                    label = 'higher_ron' if card['on'] != resistance[0] else 'lower_roff'
                    other = name + '_' + label
                    trials.append(Trial(other, set_key, hz, 'dmm', dt, (card['on'], card['off'])))
                    comparisons[other] = (main, other, False)
                other = name + '_precharged'
                trials.append(Trial(other, set_key, hz, 'dmm', dt, resistance, vo0=p['table_i'][set_key]['rated_output_v']))
                comparisons[other] = (main, other, False)
            if certified:
                for clock in ('native', 'locked'):
                    fine_name = f'{name}_{clock}_fine'
                    clock_step = dt if clock == 'locked' else 0
                    trials.append(Trial(fine_name, set_key, hz, 'dmm', run['reference_step_ps'], resistance,
                                        gate_clock_ps=clock_step))
                    comparisons[f'{name}_{clock}_15ns'] = (fine_name, main, False)
                    refinements = run['fixed_15ns_gate_refinement_steps_ps'] if clock == 'locked' else run['extra_steps_ps']
                    for alt in refinements:
                        alt_name = f'{name}_{clock}_{alt}ps'
                        trials.append(Trial(alt_name, set_key, hz, 'dmm', alt, resistance, gate_clock_ps=clock_step))
                        comparisons[alt_name] = (fine_name, alt_name, False)
    schedule = run['section_iv_d_normal_load_schedule']
    trials.append(Trial('set2_section_iv_d_nominal_load', schedule['set'], 0, 'dmm',
                        schedule['simulator_step_ps'], resistance, schedule=schedule))
    return trials, comparisons, mains


def invoke(binary, trial, p, working):
    r = p['runs']
    motor = p['table_i'][trial.set_key]
    output = working / trial.name
    threshold = r['numerical_cutoff']
    schedule = trial.schedule
    clock = [r['stop_ps'], r['save_tail_from_ps'], r['startup_save_interval_ps']]
    if schedule:
        clock = [schedule['end_ps'], schedule['tail_from_ps'], schedule['sample_stride_ps']]
    args = [binary, trial.method, output, *(motor[f] for f in CASE_FIELDS), *trial.resistance,
            trial.vo0, trial.hz, trial.dt_ps, *clock, trial.gate_clock_ps, p['criteria']['nodal_tie_v'],
            threshold['max_abs_tank_current_a'], threshold['max_voltage_multiple_of_input']]
    if schedule:
        ramp = schedule['linear_ramp']
        args.extend([round(float(ramp['start_s']) * 1e12), round(float(ramp['end_s']) * 1e12),
                     ramp['start_hz'], ramp['end_hz'], *(round(s * 1e12) for s in schedule['figure_centers_s'])])
    try:
        result = subprocess.run(list(map(str, args)), check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f'{trial.name} failed: {exc.stderr}') from exc
    wave = np.fromfile(output, dtype=np.float64).reshape(-1, 9)
    raw = json.loads(result.stdout)
    output.unlink()
    ref = None
    if trial.method == 'verified':
        other = output.with_suffix('.reference')
        ref = np.fromfile(other, dtype=np.float64).reshape(-1, 9)
        other.unlink()
    return wave, raw, ref


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--figures', type=Path)
    parser.add_argument('--compiler', default=os.environ.get('CXX', 'c++'))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    fig_root = args.figures or args.out / 'figures'
    fig_root.mkdir(parents=True, exist_ok=True)
    protocol_file = ROOT / 'configs/protocol.json'
    protocol = json.loads(protocol_file.read_text())
    if not shutil.which(args.compiler):
        raise SystemExit('A C++17 compiler is required. Set CXX or use --compiler.')

    start, tick = datetime.now(timezone.utc), perf_counter()
    trials, comparisons, mains = plan(protocol)
    metrics = {'runs': {}, 'comparisons': {}, 'validation_checks': {}, 'table_i_calculated': {}}
    store, verification = {}, {}
    flags = ['-std=c++17', '-O3', '-Wall', '-Wextra', '-Wpedantic', '-ffp-contract=off']
    with tempfile.TemporaryDirectory(prefix='.llc-work-', dir=args.out) as directory:
        working = Path(directory)
        binary = working / 'simulate'
        subprocess.run([args.compiler, *flags, ROOT / 'src/simulate.cpp', '-o', binary], check=True)
        for trial in trials:
            print(f'{trial.name} ({trial.method})', flush=True)
            wave, raw, independent = invoke(binary, trial, protocol, working)
            store[trial.name] = wave
            row = describe(wave, raw, protocol['runs']['evaluate_tail_from_ps'])
            row.update({'set': trial.set_key, 'step_ps': trial.dt_ps, 'frequency_hz': trial.hz or None,
                        'method': trial.method, 'gate_clock_ps': trial.gate_clock_ps,
                        'resistance_ohm': list(trial.resistance), 'output_initial_v': trial.vo0})
            metrics['runs'][trial.name] = row
            if trial.name in mains:
                criteria = protocol['criteria']
                verification[trial.name] = independent
                checks = verify(raw, criteria)
                row['mode_probes'] = raw['probes']
                full = compare_full_network(protocol['table_i'][trial.set_key], wave, trial.dt_ps,
                                            *trial.resistance, criteria['nodal_tie_v'])
                row['full_switch_network'] = full
                checks['full_bridge_mna'] = (full['max_current_error_a'] <= criteria['max_dmm_reference_current_error_a']
                                             and full['max_voltage_error_v'] <= criteria['max_dmm_reference_voltage_error_v']
                                             and full['mode_mismatches_outside_tie'] == 0)
                checks['four_quadrants_probed'] = all(raw['probes']['mask_counts'][m] > 0
                                                     for m in protocol['runs']['rectifier_masks'])
                metrics['validation_checks'][trial.name] = checks
            if trial.schedule:
                row['schedule_waveform_features'] = ramps(wave, trial.schedule['figure_centers_s'])
                row.pop('late_window')

    for name, (left, right, check_blocked) in comparisons.items():
        metrics['comparisons'][name] = compare(store[left], store[right], protocol['runs']['evaluate_tail_from_ps'],
                                               blocking_reference=check_blocked)
    for name, motor in protocol['table_i'].items():
        lr, lm, cr = [motor[label] for label in ('lr_h', 'lm_h', 'cr_f')]
        metrics['table_i_calculated'][name] = {'fr1_hz': 1 / (2 * pi * sqrt(lr * cr)),
                                               'fr2_hz': 1 / (2 * pi * sqrt((lr + lm) * cr))}
    put(args.out / 'metrics.json', metrics)
    np.savez_compressed(args.out / 'trajectories.npz', **store)
    np.savez_compressed(args.out / 'independent_trajectories.npz', **verification)
    figures(fig_root, store, metrics, mains)
    finish = datetime.now(timezone.utc)
    cost = perf_counter() - tick
    checks = [ok for checks in metrics['validation_checks'].values() for ok in checks.values()]
    manifest = {'doi': protocol['doi'], 'started_utc': start.isoformat(), 'finished_utc': finish.isoformat(),
                'wall_seconds': cost, 'timing_basis': 'perf_counter including compilation, numerical runs, data and figures',
                'python': platform.python_version(), 'numpy': np.__version__, 'architecture': platform.machine(),
                'compiler': subprocess.check_output([args.compiler, '--version'], text=True).splitlines()[0],
                'compiler_flags': flags, 'protocol_sha256': digest(protocol_file),
                'implementation_sha256': {name: digest(ROOT / 'src' / name) for name in
                                          ('reproduce.py', 'simulate.cpp', 'llc.hpp', 'reference.hpp',
                                           'network_check.py', 'results.py', 'plots.py')},
                'checks_passed': sum(checks), 'checks_total': len(checks), 'trials': len(trials),
                'data_sha256': {name: digest(args.out / name) for name in ('metrics.json', 'trajectories.npz',
                                                                          'independent_trajectories.npz')},
                'figures_sha256': {name: digest(fig_root / name) for name in
                                   ('rectifier_blocking.png', 'step_size.png', 'frequency_schedule.png')}}
    put(args.out / 'run_manifest.json', manifest)
    print(f'{len(trials)} trials. {sum(checks)}/{len(checks)} checks passed in {cost:.2f} s.')
    return 0 if all(checks) else 1


if __name__ == '__main__':
    raise SystemExit(main())
