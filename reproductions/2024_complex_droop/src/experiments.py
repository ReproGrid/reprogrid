"""Numerical experiments for the three single-converter case studies."""
from dataclasses import replace
from itertools import product

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, root

from models import (AveragedModel, Parameters, cubic_coefficients, equilibria,
                    eta_certificate_supremum, lifted_equilibrium, stability_margins,
                    voltage_bound)


def parameters(protocol, case, alpha, eta, angle='impedance_matched', rg=None, phi_override=None):
    return Parameters(
        rg=case['rg'] if rg is None else rg, xg=case['xg'], p_star=case['p_star'],
        q_star=case['q_star'], alpha=alpha, eta_over_omega0=eta,
        base=protocol['base'], angle=angle, phi_override=phi_override,
    )


def unique_equilibrium(p, vg):
    solutions = equilibria(p, vg)
    if len(solutions) != 1:
        raise RuntimeError(f'Expected one steady state for {p}, vg={vg}; got {solutions}')
    return solutions[0]


def eigenvalues(p, vg, order, equilibrium):
    x = lifted_equilibrium(p, vg, equilibrium, order)
    return np.linalg.eigvals(AveragedModel(p, vg, order).jacobian(0., x))


def sample_times(start, stop, interval):
    return np.linspace(start, stop, round((stop - start) / interval) + 1)


def integrate(method, rhs, jac, bounds, initial, times, controls, reference=False, event=None):
    prefix = 'reference' if reference else 'primary'
    options = {
        'method': method,
        't_eval': times,
        'rtol': controls[f'{prefix}_rtol'],
        'atol': controls[f'{prefix}_atol'],
    }
    if jac is not None:
        options['jac'] = jac
    if event is not None:
        options['events'] = event
        options['max_step'] = controls['cycle_max_step_s']
    result = solve_ivp(rhs, bounds, initial, **options)
    if not result.success:
        raise ArithmeticError(result.message)
    return result


def cycle_event(section):
    def crossing(t, x):
        return x[1] - section
    crossing.direction = 1
    return crossing


def cycle_statistics(solution, count):
    crossings = solution.t_events[0]
    points = solution.y_events[0]
    if len(crossings) < count:
        raise ArithmeticError(f'Only {len(crossings)} upward crossings in a limit-cycle experiment')
    last = crossings[-count:]
    periods = np.diff(last)
    period = periods.mean()
    return {
        'crossings': int(len(crossings)),
        'mean_period_s': float(period),
        'relative_period_spread': float(np.ptp(periods) / period),
        'section_return_spread_pu': float(np.ptp(points[-count:, 0])),
    }


def compare_voltage_dip(p, order, case, protocol, reference_rhs, diff_jacobian, periodic=False):
    base = protocol['base']
    pre, post = base['vg_pre'], base['vg_post']
    pre_voltages = equilibria(p, pre)
    pre_growth_rates = [float(max(eigenvalues(p, pre, order, v).real)) for v in pre_voltages]
    stable_pre = [v for v, growth in zip(pre_voltages, pre_growth_rates) if growth < 0]
    if len(pre_voltages) == 1:
        initial_voltage = pre_voltages[0]
    elif len(stable_pre) == 1:
        initial_voltage = stable_pre[0]
    else:
        raise RuntimeError(f'Cannot select a pre-dip equilibrium for order {order}: {pre_voltages}')
    post_voltage = unique_equilibrium(p, post)
    controls = protocol['solver']
    times = sample_times(case['start_s'], case['stop_s'], case['sample_s'])
    event_index = round((case['dip_s'] - case['start_s']) / case['sample_s'])
    times[event_index] = case['dip_s']
    post_times = times[event_index:]
    x0 = lifted_equilibrium(p, pre, initial_voltage, order)
    xs = lifted_equilibrium(p, post, post_voltage, order)
    model = AveragedModel(p, post, order)
    direct_rhs = reference_rhs(p, post, order)
    event = cycle_event(post_voltage.imag) if periodic else None
    t_span = (case['dip_s'], case['stop_s'])
    main = integrate(controls['primary_method'], model.rhs, model.jacobian, t_span, x0, post_times,
                     controls, event=event)
    direct = integrate(controls['reference_method'], direct_rhs, None, t_span, x0, post_times,
                       controls, reference=True)
    prefix = np.tile(x0, (event_index, 1))
    wave = np.vstack((prefix, main.y.T))
    reference = np.vstack((prefix, direct.y.T))
    jacobian = model.jacobian(0., xs)
    finite_difference = diff_jacobian(direct_rhs, xs)
    poles = np.linalg.eigvals(jacobian)
    peak_state_difference = float(np.max(np.abs(wave - reference)))
    direct_steady_rhs = reference_rhs(p, post, 2)
    direct_eq = root(lambda x: direct_steady_rhs(0., x), [post, 0.], tol=1e-11)
    statistics = {
        'order': order,
        'eta_over_omega0': p.eta_over_omega0,
        'alpha': p.alpha,
        'phi_rad': p.phi,
        'pre_equilibrium_v': [initial_voltage.real, initial_voltage.imag],
        'post_equilibrium_v': [post_voltage.real, post_voltage.imag],
        'pre_equilibria': [
            {'v_dq': [v.real, v.imag], 'dominant_real_per_s': growth}
            for v, growth in zip(pre_voltages, pre_growth_rates)
        ],
        'post_equilibrium_residual': float(np.linalg.norm(model.rhs(0., xs), ord=np.inf)),
        'pre_equilibrium_residual': float(np.linalg.norm(AveragedModel(p, pre, order).rhs(0., x0), ord=np.inf)),
        'direct_equilibrium_residual': float(np.linalg.norm(direct_rhs(0., xs), ord=np.inf)),
        'independent_root_delta': float(np.linalg.norm(direct_eq.x - xs[:2])),
        'independent_root_residual': float(np.linalg.norm(direct_steady_rhs(0., direct_eq.x), ord=np.inf)),
        'jacobian_scaled_error': float(np.linalg.norm(jacobian - finite_difference, ord=np.inf)
                                       / max(1., np.linalg.norm(jacobian, ord=np.inf))),
        'dominant_real_per_s': float(np.max(poles.real)),
        'poles_per_s': [[float(s.real), float(s.imag)] for s in sorted(poles, key=lambda z: (z.real, z.imag))],
        'peak_reference_difference': peak_state_difference,
        'peak_voltage_magnitude_pu': float(np.linalg.norm(wave[:, :2], axis=1).max()),
        'end_voltage_error_pu': float(np.linalg.norm(wave[-1, :2] - xs[:2])),
        'tail_voltage_error_rms_pu': float(np.sqrt(np.mean(np.sum((wave[-400:, :2] - xs[:2])**2, axis=1)))),
        'main_rhs_evaluations': main.nfev,
        'reference_rhs_evaluations': direct.nfev,
    }
    if periodic:
        statistics['cycle'] = cycle_statistics(main, controls['cycle_crossings_used'])
        statistics['post_peak_voltage_pu'] = float(np.linalg.norm(main.y.T[:, :2], axis=1).max())
    return times, wave, reference, statistics


def tracking_errors(p, case, time, wave):
    """Equation (56) error coordinates and Fig. 4's normalized magnitudes."""
    vg = np.where(time < case['dip_s'], p.base['vg_pre'], p.base['vg_post'])
    voltage_ref, current, terminal, z_v, i_f, z_c = [
        wave[:, j] + 1j * wave[:, j + 1] for j in range(0, 12, 2)
    ]
    target = unique_equilibrium(p, p.base['vg_post'])
    if_star = (p.yf * terminal + current - p.base['kp_v'] * (terminal - voltage_ref)
               - p.base['kr_v_per_s'] * z_v)
    errors = np.column_stack((
        abs(voltage_ref - target),
        abs(current - p.y * (voltage_ref - vg)),
        np.sqrt(abs(terminal - voltage_ref)**2 + abs(z_v)**2),
        np.sqrt(abs(i_f - if_star)**2 + abs(z_c)**2),
    ))
    errors[time < case['dip_s']] = 0.
    return errors / np.maximum(errors.max(axis=0), np.finfo(float).tiny)


def first_loss_of_local_stability(p, vg, equilibrium, eta_grid):
    """First increasing-eta crossing from a strictly stable starting point."""
    normalized = []
    x = lifted_equilibrium(p, vg, equilibrium, 4)
    for ratio in (0., 1.):
        model = AveragedModel(replace(p, eta_over_omega0=ratio), vg, 4)
        normalized.append(model.jacobian(0., x) / p.omega0)
    constant, linear = normalized[0], normalized[1] - normalized[0]
    values = np.linalg.eigvals(constant + eta_grid[:, None, None] * linear).real.max(axis=1)
    changes = np.flatnonzero(np.signbit(values[1:]) != np.signbit(values[:-1]))
    if values[0] >= 0:
        return 0., int(len(changes))
    if not len(changes):
        return None, 0
    i = changes[0]
    def growth_rate(ratio):
        return np.linalg.eigvals(constant + ratio * linear).real.max()
    return float(brentq(growth_rate, eta_grid[i], eta_grid[i + 1], xtol=2e-13)), int(len(changes))


def stability_sweep(protocol):
    case = protocol['case_ii']
    start, stop = case['eta_scan_range_over_omega0']
    eta_grid = np.geomspace(start, stop, case['eta_scan_points'])
    alphas = np.linspace(*case['alpha_range'], case['alpha_points'])
    vg = protocol['base']['vg_post']
    angle_i = parameters(protocol, protocol['case_i'], 1., 0.02).phi
    outputs = []
    scenarios = [
        ('impedance_matched', case['rg_values'][0], None),
        ('impedance_matched', case['rg_values'][1], None),
        ('printed_literal', case['rg_values'][0], None),
        ('held_case_i_angle', case['rg_values'][1], angle_i),
    ]
    for angle, rg, override in scenarios:
        for alpha in alphas:
            p = parameters(protocol, case, float(alpha), 0.1, angle, rg, override)
            solutions = equilibria(p, vg)
            v = max(solutions, key=abs)
            crossing, number_of_changes = first_loss_of_local_stability(p, vg, v, eta_grid)
            outputs.append({
                'angle_branch': angle, 'rg': rg, 'alpha': float(alpha),
                'phi_rad': p.phi, 'equilibrium_count': len(solutions),
                'selected_voltage_mag_pu': abs(v),
                'certificate_supremum': eta_certificate_supremum(p, v) if len(solutions) == 1 else None,
                'critical_eta_over_omega0': crossing,
                'crossings_in_scan': number_of_changes,
            })
    return outputs


def phase_experiment(p, protocol, periodic):
    case = protocol['case_iii']
    controls = protocol['solver']
    vg = protocol['base']['vg_post']
    eq = unique_equilibrium(p, vg)
    model = AveragedModel(p, vg, 2)
    points = list(product(case['phase_initial_values'], repeat=2))
    times = sample_times(0., case['phase_duration_s'], 0.002)
    traces = []
    results = []
    for point in points:
        event = cycle_event(eq.imag) if periodic else None
        sol = integrate(controls['primary_method'], model.rhs, model.jacobian,
                        (times[0], times[-1]), np.array(point, dtype=float), times, controls, event=event)
        traces.append(sol.y.T)
        if periodic:
            stats = cycle_statistics(sol, controls['cycle_crossings_used'])
            stats['last_section_vd'] = float(sol.y_events[0][-1, 0])
        else:
            stats = {'tail_distance_pu': float(np.linalg.norm(sol.y[:, -1] - [eq.real, eq.imag]))}
        results.append({'initial_v': point, **stats})
    return times, np.array(traces), results
