"""Rebuild the single-converter dVOC calculations and independent comparisons."""
from argparse import ArgumentParser
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from time import perf_counter
import csv
import json
import platform
import sys

import numpy as np
import scipy
from scipy.optimize import brentq

from experiments import (compare_voltage_dip, eigenvalues, parameters, phase_experiment,
                         stability_sweep, tracking_errors, unique_equilibrium)
from models import (AveragedModel, Parameters, cubic_coefficients, lifted_equilibrium,
                    stability_margins, voltage_bound)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'verification'))
from reference import jacobian_by_difference, quartic_coefficients, reference_rhs, routh_determinant

PROTOCOL = ROOT / 'configs/protocol.json'


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def dump(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def algebra_evidence(protocol):
    base = protocol['base']
    case_i, case_iii = protocol['case_i'], protocol['case_iii']
    post = base['vg_post']
    rng = np.random.default_rng(protocol['algebra_checks']['random_seed'])
    p = parameters(protocol, case_i, case_i['alpha'], case_i['eta_over_omega0'][0])
    vs = unique_equilibrium(p, post)
    xs = lifted_equilibrium(p, post, vs, 2)
    model = AveragedModel(p, post, 2)
    errors = rng.normal(size=(protocol['algebra_checks']['lyapunov_points'], 2)) * 2.
    margin = stability_margins(p, vs)['global_eq23']
    lyapunov_error = [e @ model.rhs(0., xs + e) / p.eta + margin * (e @ e) for e in errors]

    # Appendix A, Example 3, with the line impedance in pu.
    classical = Parameters(.4, .4, 0., 0., 1., .08, base, phi_override=np.pi / 2)
    y_rotation = classical.rotation * classical.y
    polynomial = np.polynomial.Polynomial
    quartic = polynomial([1., -1., -y_rotation.real])**2
    quartic += polynomial([0., 0., -y_rotation.imag])**2
    quartic -= polynomial([0., 0., (.1 * abs(classical.y))**2])
    critical = [0.] + [r.real for r in quartic.deriv().roots() if abs(r.imag) < 1e-10 and r.real > 0]
    no_droop_eq = min(float(quartic(v)) for v in critical)
    complex_eq = unique_equilibrium(classical, .1)
    complex_residual = AveragedModel(classical, .1, 2).rhs(0., lifted_equilibrium(classical, .1, complex_eq, 2))

    # Exact radial polynomial (s-1)^3 when the coefficients are as specified.
    unrotated_setpoint = complex(1.5, np.sqrt(3) / 2) * complex(.8, -.6)
    degenerate = Parameters(.8, .6, unrotated_setpoint.real, -unrotated_setpoint.imag, 1., .08, base)
    coeff, _, _, _ = cubic_coefficients(degenerate, 1.)
    triple_voltage = complex(-.5, np.sqrt(3) / 2)
    triple_residual = AveragedModel(degenerate, 1., 2).rhs(0., lifted_equilibrium(degenerate, 1., triple_voltage, 2))

    # Proposition 5 displays the Jacobian in physical time. Compare its
    # determinant with the displayed characteristic polynomial constant.
    unstable = parameters(protocol, case_iii, case_iii['alpha_unstable'], case_iii['eta_over_omega0'])
    v = unique_equilibrium(unstable, post)
    kappa = unstable.rotated_setpoint - unstable.rotation * unstable.y
    a = unstable.alpha * abs(v)**2 / unstable.v_star**2
    printed_constant = (kappa.real + unstable.alpha - 2 * a)**2 - a**2
    printed_constant += (kappa.imag + unstable.omega_delta / unstable.eta)**2
    J = AveragedModel(unstable, post, 2).jacobian(0., lifted_equilibrium(unstable, post, v, 2))
    determinant = float(np.linalg.det(J))
    return {
        'lyapunov': {'points': len(errors), 'margin_eq23': margin,
                     'max_Vdot_minus_proven_upper_bound': float(max(lyapunov_error))},
        'classical_example_3': {
            'classical_quartic_coefficients_ascending': quartic.coef.tolist(),
            'classical_quartic_global_minimum_on_nonnegative_voltage': no_droop_eq,
            'dVOC_equilibrium_dq_pu': [complex_eq.real, complex_eq.imag],
            'dVOC_equilibrium_residual': float(np.linalg.norm(complex_residual, ord=np.inf)),
        },
        'proposition_4_triple_root': {
            'exact_coefficients_descending': [1, -3, 3, -1], 'exact_discriminant': 0,
            'number_of_distinct_positive_roots': 1, 'exact_squared_voltage_root': 1,
            'coefficient_error_from_physical_inputs': float(max(abs(coeff - [1, -3, 3, -1]))),
            'physical_voltage_dq_pu': [triple_voltage.real, triple_voltage.imag],
            'equilibrium_residual': float(np.linalg.norm(triple_residual, ord=np.inf)),
        },
        'proposition_5_characteristic_polynomial': {
            'eta_rad_per_s': unstable.eta, 'displayed_constant': float(printed_constant),
            'direct_jacobian_determinant': determinant,
            'corrected_constant': float(unstable.eta**2 * printed_constant),
            'relative_error_corrected_constant': float(abs(unstable.eta**2 * printed_constant - determinant) / abs(determinant)),
        },
    }


def crossing_checks(protocol, sweep):
    case = protocol['case_ii']
    vg = protocol['base']['vg_post']
    rows = []
    for row in sweep:
        if row['alpha'] != case['comparison_alpha']:
            continue
        rg = row['rg']
        angle = row['angle_branch']
        override = (parameters(protocol, protocol['case_i'], 1., 0.02).phi
                    if angle == 'held_case_i_angle' else None)
        critical = row['critical_eta_over_omega0']
        p = parameters(protocol, case, row['alpha'], critical, angle, rg, override)
        vs = unique_equilibrium(p, vg)
        x = lifted_equilibrium(p, vg, vs, 4)
        J = AveragedModel(p, vg, 4).jacobian(0., x) / p.omega0
        coefficients = quartic_coefficients(p, vs)
        det = lambda r: routh_determinant(quartic_coefficients(
            parameters(protocol, case, row['alpha'], r, angle, rg, override), vs))
        routh_crossing = float(brentq(det, .95 * critical, 1.05 * critical, xtol=2e-13))
        rows.append({
            **row, 'routh_crossing': routh_crossing, 'crossing_difference': abs(routh_crossing - critical),
            'quartic_coefficient_scaled_error': float(np.max(np.abs(coefficients - np.poly(J)))
                                                       / np.max(np.abs(coefficients))),
            'post_equilibrium_v': [vs.real, vs.imag],
            'at_crossing_poles_per_s': [[float(z.real), float(z.imag)] for z in np.linalg.eigvals(J * p.omega0)],
        })
    return rows


def simulate(protocol):
    output = {}
    arrays = {}
    case_i, case_ii, case_iii = [protocol[f'case_{i}'] for i in ['i', 'ii', 'iii']]
    vg = protocol['base']['vg_post']
    controls = protocol['solver']

    def run(case, name, p, order, periodic=False):
        t, x, ref, metrics = compare_voltage_dip(
            p, order, case, protocol, reference_rhs, jacobian_by_difference, periodic=periodic)
        arrays[f'{name}__t'] = t
        arrays[f'{name}__x'] = x
        arrays[f'{name}__ref'] = ref
        output[name] = metrics
        return t, x

    for eta in case_i['eta_over_omega0']:
        p = parameters(protocol, case_i, case_i['alpha'], eta)
        for order in case_i['orders']:
            name = f'case_i_eta_{eta:.3f}_order_{order}'
            t, x = run(case_i, name, p, order)
            if order == 12:
                arrays[f'{name}__tracking_error'] = tracking_errors(p, case_i, t, x)
        print(f'Case I eta/omega0={eta:.3f} complete', flush=True)

    for eta in case_ii['critical_example_eta_ratios']:
        p = parameters(protocol, case_ii, case_ii['comparison_alpha'], eta, rg=case_ii['rg_values'][0])
        for order in (2, 4):
            run(case_ii, f'case_ii_eta_{eta:.3f}_order_{order}', p, order)
        print(f'Case II eta/omega0={eta:.3f} complete', flush=True)

    phase_statistics = {}
    for alpha in (case_iii['alpha_unstable'], case_iii['alpha_stable']):
        p = parameters(protocol, case_iii, alpha, case_iii['eta_over_omega0'])
        periodic = alpha == case_iii['alpha_unstable']
        orders = case_iii['orders'] if periodic else [2]
        for order in orders:
            run(case_iii, f'case_iii_alpha_{alpha:.1f}_order_{order}', p, order, periodic=periodic)
        t, trajectories, stats = phase_experiment(p, protocol, periodic)
        arrays[f'phase_alpha_{alpha:.1f}__t'] = t
        arrays[f'phase_alpha_{alpha:.1f}__x'] = trajectories
        phase_statistics[str(alpha)] = stats
        print(f'Case III alpha={alpha:.1f} complete', flush=True)

    angle_diagnostic = []
    for angle in ('impedance_matched', 'printed_literal'):
        for alpha in (case_iii['alpha_unstable'], case_iii['alpha_stable']):
            p = parameters(protocol, case_iii, alpha, case_iii['eta_over_omega0'], angle)
            v = unique_equilibrium(p, vg)
            angle_diagnostic.append({
                'angle_branch': angle, 'alpha': alpha, 'phi_rad': p.phi,
                'post_equilibrium_v': [v.real, v.imag], 'voltage_bound_pu': float(voltage_bound(p, vg)),
                'dominant_real_per_s': float(max(eigenvalues(p, vg, 2, v).real)),
                **stability_margins(p, v),
            })
    return output, phase_statistics, angle_diagnostic, arrays


def evaluate_checks(protocol, cases, phases, angles, crossings, algebra):
    criteria = protocol['numerical_criteria']
    cycle_counts = [cases[f'case_iii_alpha_3.0_order_{order}']['cycle'] for order in (2, 12)]
    section = phases['3.0']
    bound = next(x['voltage_bound_pu'] for x in angles if x['alpha'] == 3. and x['angle_branch'] == 'impedance_matched')
    normal_cases = [x for x in crossings if x['angle_branch'] == 'impedance_matched']
    low, high = protocol['case_ii']['critical_example_eta_ratios']
    metrics = list(cases.values())
    return {
        'equilibria_lift_and_direct_root': max(x[k] for x in metrics for k in (
            'post_equilibrium_residual', 'pre_equilibrium_residual', 'direct_equilibrium_residual',
            'independent_root_residual', 'independent_root_delta')) <= criteria['max_equilibrium_residual_pu_per_s'],
        'jacobians_independent_finite_difference': max(x['jacobian_scaled_error'] for x in metrics) <= criteria['max_jacobian_scaled_error'],
        'independent_dip_trajectories': max(x['peak_reference_difference'] for x in metrics) <= criteria['max_trajectory_peak_abs_error_pu'],
        'quartic_coefficients': max(x['quartic_coefficient_scaled_error'] for x in crossings) <= criteria['max_relative_quartic_coeff_error'],
        'routh_crossings_match_eigenvalues': max(x['crossing_difference'] for x in crossings) <= criteria['max_independent_crossing_eta_difference_over_omega0'],
        'case_ii_reported_critical_interval': low < normal_cases[0]['critical_eta_over_omega0'] < high,
        'case_ii_sufficient_bound_below_local_crossing': all(0 < x['certificate_supremum'] < x['critical_eta_over_omega0'] for x in normal_cases),
        'case_ii_second_order_stays_locally_stable': all(cases[f'case_ii_eta_{r:.3f}_order_2']['dominant_real_per_s'] < 0 for r in (low, high)),
        'case_iii_alpha_3_repelling_equilibrium': cases['case_iii_alpha_3.0_order_2']['dominant_real_per_s'] > 0,
        'case_iii_alpha_1_not_certified_but_attracts': (max(x['tail_distance_pu'] for x in phases['1.0'])
            <= criteria['max_attracting_fixed_point_tail_error_pu']
            and next(x['global_eq23'] for x in angles if x['alpha'] == 1. and x['angle_branch'] == 'impedance_matched') < 0),
        'case_iii_periodic_returns': (max(x['relative_period_spread'] for x in cycle_counts + section)
            <= criteria['max_cycle_period_relative_spread']
            and max(x['section_return_spread_pu'] for x in cycle_counts + section)
            <= criteria['max_cycle_return_section_spread_pu']),
        'case_iii_phase_initial_conditions_share_cycle': np.ptp([x['last_section_vd'] for x in section])
            <= criteria['max_cycle_return_section_spread_pu'],
        'case_iii_periods_comparable_across_order': abs(cycle_counts[1]['mean_period_s'] / cycle_counts[0]['mean_period_s'] - 1.)
            <= criteria['max_two_vs_twelve_cycle_period_relative_difference'],
        'case_iii_second_order_voltage_bound': cases['case_iii_alpha_3.0_order_2']['post_peak_voltage_pu']
            <= bound + criteria['max_voltage_bound_violation_pu'],
        'lyapunov_derivative_bound': algebra['lyapunov']['max_Vdot_minus_proven_upper_bound'] <= 1e-10,
        'example_3_equilibrium_existence': (algebra['classical_example_3']['classical_quartic_global_minimum_on_nonnegative_voltage'] > 0
            and algebra['classical_example_3']['dVOC_equilibrium_residual'] <= 1e-8),
        'triple_root_exact_construction': (algebra['proposition_4_triple_root']['equilibrium_residual'] <= 1e-8
            and algebra['proposition_4_triple_root']['coefficient_error_from_physical_inputs'] <= 1e-12),
        'characteristic_polynomial_corrected_constant': algebra['proposition_5_characteristic_polynomial']['relative_error_corrected_constant'] <= 1e-10,
    }


def main():
    ap = ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=ROOT / 'results')
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    started = utc()
    clock = perf_counter()
    protocol = json.loads(PROTOCOL.read_text())
    sweep = stability_sweep(protocol)
    crossings = crossing_checks(protocol, sweep)
    algebra = algebra_evidence(protocol)
    with (out / 'stability_sweep.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sweep[0]))
        writer.writeheader()
        writer.writerows(sweep)
    dump(out / 'critical_eta.json', crossings)
    dump(out / 'algebra_evidence.json', algebra)
    print('Equilibrium and stability sweep complete', flush=True)
    cases, phases, angles, arrays = simulate(protocol)
    dump(out / 'case_metrics.json', cases)
    dump(out / 'phase_metrics.json', phases)
    dump(out / 'rotation_angle_diagnostic.json', angles)
    np.savez_compressed(out / 'trajectories.npz', **arrays)
    checks = {k: bool(v) for k, v in evaluate_checks(protocol, cases, phases, angles, crossings, algebra).items()}
    source_code = [Path(__file__), ROOT / 'src/models.py', ROOT / 'src/experiments.py', ROOT / 'verification/reference.py']
    artifacts = ('stability_sweep.csv', 'critical_eta.json', 'algebra_evidence.json', 'case_metrics.json',
                 'phase_metrics.json', 'rotation_angle_diagnostic.json', 'trajectories.npz')
    manifest = {
        'doi': protocol['doi'], 'started_utc': started, 'finished_utc': utc(), 'wall_seconds': perf_counter() - clock,
        'python': platform.python_version(), 'platform': platform.platform(), 'numpy': np.__version__, 'scipy': scipy.__version__,
        'protocol_sha256': digest(PROTOCOL),
        'code_sha256': {str(path.relative_to(ROOT)): digest(path) for path in source_code},
        'outputs': {name: digest(out / name) for name in artifacts},
        'passed': all(checks.values()), 'checks': checks,
    }
    dump(out / 'run_manifest.json', manifest)
    print(json.dumps({'checks': checks, 'wall_seconds': manifest['wall_seconds']}, indent=2))
    return 0 if manifest['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
