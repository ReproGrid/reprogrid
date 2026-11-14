"""Run the frozen Section IV benchmark and the independently checked sensitivities."""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import scipy

from independent import independent, printed_observer_check
from metrics import capacity, compare_solvers, sampled, summarize
from model import load_cases, primary
from plots import make_figures


ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def comparisons_with_table_ii(cases, test):
    columns = test['table_ii_column_order']
    comparisons = {}
    for name, paper in test['table_ii'].items():
        computed = cases[f'equation_symbol_4__{name}']['published_metric_comparison']
        comparisons[name] = {}
        for key, reference in zip(columns, paper, strict=True):
            ours = computed[key]
            comparisons[name][key] = {'paper': reference, 'computed': ours,
                                      'signed_delta': ours - reference if ours is not None else None}
    return comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'results')
    parser.add_argument('--figures', type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    figures = args.figures or args.out / 'figures'
    compiled = ROOT / 'configs/protocol.json'
    protocol = json.loads(compiled.read_text())
    cfg = protocol['numerics']
    started = datetime.now(timezone.utc)
    tick = perf_counter()

    cases = load_cases(protocol)
    metrics = {'cases': {}, 'capacity': {}, 'cross_checks': {}, 'checks': {}}
    primary_arrays, reference_arrays = {}, {}
    for case in cases:
        print(f'Running {case.name}', flush=True)
        case_cfg = cfg
        if case.controller.law in ('eq16', 'eq17') and case.voltage_limit is not None:
            case_cfg = cfg | {'boundary_margin_a': cfg['boundary_refinement_margin_a']}
        runs = [primary(case, case_cfg), independent(case, case_cfg)]
        main_wave, ref_wave = [sampled(run, case, case_cfg['sampling_s']) for run in runs]
        main, ref = [summarize(run, array, case, protocol)
                     for run, array in zip(runs, (main_wave, ref_wave), strict=True)]
        main['event_margin_a'] = case_cfg['boundary_margin_a'] if case.controller.barrier_l else None
        data = compare_solvers(main_wave, ref_wave, main, ref, case, case_cfg)
        primary_arrays[case.name], reference_arrays[case.name] = main_wave, ref_wave
        metrics['cases'][case.name] = main
        metrics['cross_checks'][case.name] = data
        metrics['checks'][case.name] = data['passed']
        if case.family == 'main' and case.controller_name == 'dob_cc_eq16':
            metrics['capacity'][case.poles] = capacity(case.motor, case.controller)

    named = next(case for case in cases if case.name == 'equation_symbol_4__dob_cc_eq16')
    eq13_vs_14 = printed_observer_check(named.motor, named.controller.p, 1e-13)
    metrics['observer_eq13_error_nm'] = eq13_vs_14
    metrics['checks']['printed_eq13_matches_eq14'] = eq13_vs_14 < 2e-6
    m = named.motor
    statics = metrics['capacity']['equation_symbol_4']['equilibria']
    eq_residuals = [abs(m.k * row['iq_a'] - m.friction * m.reference - row['load_nm']) for row in statics]
    metrics['checks']['nominal_torque_balance'] = max(eq_residuals) < 1e-12
    metrics['table_ii_comparison'] = comparisons_with_table_ii(metrics['cases'], protocol['numerical_test'])

    data_files = ['metrics.json', 'trajectories.npz', 'verification_trajectories.npz']
    write_json(args.out / data_files[0], metrics)
    for file, arrays in zip(data_files[1:], (primary_arrays, reference_arrays), strict=True):
        np.savez_compressed(args.out / file, **{f'{name}__{column}': values
                                                for name, wave in arrays.items() for column, values in wave.items()})
    make_figures(primary_arrays, metrics, figures)
    finished = datetime.now(timezone.utc)
    manifest = {
        'doi': protocol['doi'], 'started_utc': started.isoformat(), 'finished_utc': finished.isoformat(),
        'wall_seconds': perf_counter() - tick, 'timing_basis': 'perf_counter through data and figure writing',
        'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__,
        'protocol_sha256': digest(compiled),
        'implementation_sha256': {p.name: digest(p) for p in sorted((ROOT / 'src').glob('*.py'))
                                  if p.name != 'build_report.py'},
        'checks_passed': sum(metrics['checks'].values()), 'checks_total': len(metrics['checks']),
        'data_sha256': {file: digest(args.out / file) for file in data_files},
        'figures_sha256': {n: digest(figures / n) for n in ('startup_and_load_step.png', 'overload_boundary.png',
                                                          'voltage_authority.png')},
    }
    write_json(args.out / 'run_manifest.json', manifest)
    print(f"Completed {len(cases)} cases. Checks passed {manifest['checks_passed']}/{manifest['checks_total']}.")
    return 0 if all(metrics['checks'].values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
