"""Limits, source comparisons and measurements on the two solver outputs."""

from math import sqrt

import numpy as np
from scipy.optimize import brentq

from model import RPM, command


def solver_grid(piece):
    grid = piece.sol.t.copy()
    if piece.sol.status == 1 and grid[-1] < 1e-6:
        # Tighten the already isolated root, particularly for the unlimited-voltage trial.
        index = next(i for i, events in enumerate(piece.sol.t_events) if len(events))
        interpolation = piece.sol.sol.interpolants[-1]
        target = piece.event_margins[index]
        grid[-1] = brentq(lambda time: float(interpolation(time)[1]) - target,
                          interpolation.t_old, interpolation.t, xtol=1e-23)
    return grid


def sampled(run, case, dt):
    """Store both sides of every input discontinuity and the exact terminal state."""
    columns = ('t_s', 'load_nm', 'speed_rpm', 'iq_a', 'uq_v', 'aux', 'voltage_clip')
    arrays = {name: [] for name in columns}
    for piece in run.pieces:
        part = piece.segment
        t1 = solver_grid(piece)[-1]
        grid = np.arange(int(t1 / dt) + 1) * dt
        times = np.r_[grid[grid < t1 - 5e-13], t1]
        states = piece.sol.sol(times) if t1 > 0 else piece.sol.y[:, :1]
        omega, iq, aux = piece.physical(times, states)
        values = [command(case, w, case.motor.ceiling - q, y) for w, q, y in zip(omega, iq, aux, strict=True)]
        applied, clipped = np.asarray(values).T
        absolute_times = part.start + times
        loads = [part.torque(float(t)) for t in absolute_times]
        for name, array in zip(columns, (absolute_times, loads, omega * RPM, iq, applied, aux, clipped), strict=True):
            arrays[name].append(array)
    return {name: np.concatenate(values) for name, values in arrays.items()}


def dense_peak(piece, quantity, absolute=False, sign=1.0):
    if len(piece.sol.t) == 1:
        time = piece.sol.t[0]
        value = piece.physical(time, piece.sol.y[:, 0])[quantity]
        return float(abs(value) if absolute else sign * value), float(time)

    def values(times):
        item = piece.physical(times, piece.sol.sol(times))[quantity]
        return np.abs(item) if absolute else sign * item

    # Screen every accepted interval, then use the dense polynomial's stationary points
    # on intervals competing for the peak. Treat nearly flat equilibrium intervals alike.
    t = solver_grid(piece)
    fractions = np.array([0.25, 0.5, 0.75])
    interior = t[:-1] + fractions[:, None] * np.diff(t)
    grid = np.r_[t, interior.ravel()]
    states = values(grid)
    peak = int(np.argmax(states))
    best = (float(states[peak]), float(grid[peak]))
    per_step = np.vstack((states[:len(t)-1], states[1:len(t)], states[len(t):].reshape(3, -1)))
    candidates = np.where((np.max(per_step, axis=0) >= best[0] - max(1e-7, abs(best[0]) * 1e-3))
                          & (np.ptp(per_step, axis=0) > max(1e-10, abs(best[0]) * 1e-11)))[0]
    for idx in candidates:
        interpolant = piece.sol.sol.interpolants[idx]
        if hasattr(interpolant, 'Q'):
            derivative = np.polynomial.polynomial.polyder(np.r_[0, interpolant.Q[quantity]])
            roots = interpolant.t_old + interpolant.h * np.polynomial.polynomial.polyroots(derivative)
        else:
            # BDF dense output is expressed in a shifted Newton-product basis.
            poly = np.polynomial.Polynomial([0.0])
            term = np.polynomial.Polynomial([1.0])
            for shift, scale, coef in zip(interpolant.t_shift, interpolant.denom,
                                          interpolant.D[1:, quantity], strict=True):
                term *= np.polynomial.Polynomial([(t[idx] - shift) / scale, (t[idx + 1] - t[idx]) / scale])
                poly += coef * term
            roots = t[idx] + (t[idx + 1] - t[idx]) * poly.deriv().roots()
        real = np.real(roots[np.isreal(roots)])
        valid = real[(t[idx] < real) & (real < t[idx + 1])]
        if len(valid):
            candidates_at_roots = values(valid)
            i = int(np.argmax(candidates_at_roots))
            best = max(best, (float(candidates_at_roots[i]), float(valid[i])))
    return best


def time_to_stay(piece, reference_rpm, band_rpm, expected_end):
    end = float(piece.sol.t[-1])
    if abs(end - expected_end) > 1e-12:
        return None
    t = piece.sol.t
    grid = np.sort(np.r_[t, (t[1:] + t[:-1]) * 0.5])

    def excess(times):
        return np.abs(piece.physical(times, piece.sol.sol(times))[0] * RPM - reference_rpm) - band_rpm

    outside = np.flatnonzero(excess(grid) > 0)
    if len(outside) == 0:
        return 0.0
    idx = outside[-1]
    if idx == len(grid) - 1:
        return None
    return float(brentq(excess, grid[idx], grid[idx + 1], xtol=1e-13) - t[0])


def summarize(run, arrays, case, protocol):
    summary = {'status': run.status, 'message': run.message, 'end_s': float(arrays['t_s'][-1]),
               'family': case.family, 'poles': case.poles, 'controller': case.controller_name,
               'voltage_limit_v': case.voltage_limit}
    piece_metrics = []
    ref = case.motor.reference * RPM
    limits = protocol['measurement']
    for piece in run.pieces:
        seg = piece.segment
        peak, peak_t = dense_peak(piece, 1, absolute=True)
        high, _ = dense_peak(piece, 0)
        negative_low, _ = dense_peak(piece, 0, sign=-1.0)
        t1 = float(solver_grid(piece)[-1])
        end = piece.physical(t1, piece.sol.sol(t1) if t1 > 0 else piece.sol.y[:, -1])
        record = {'segment': seg.name, 'start_s': seg.start, 'end_s': seg.start + t1,
                  'peak_abs_current_a': peak, 'peak_current_time_s': seg.start + peak_t, 'max_speed_rpm': high * RPM,
                  'min_speed_rpm': -negative_low * RPM, 'end_speed_rpm': float(end[0] * RPM),
                  'end_iq_a': float(end[1]), 'end_load_nm': seg.torque(seg.start + t1)}
        if seg.name in ('startup', 'load_step', 'second_step'):
            record['time_to_stay_s'] = time_to_stay(piece, ref, limits['band_rpm'], seg.end - seg.start)
        if seg.name == 'sinusoid' and abs(t1 - (seg.end - seg.start)) < 1e-12:
            # Integrate squared tracking error using four Gauss nodes per adaptive solver step.
            nodes, weights = np.polynomial.legendre.leggauss(4)
            t = piece.sol.t
            half_widths = (t[1:] - t[:-1]) / 2
            quadrature_times = (t[1:] + t[:-1]) / 2 + nodes[:, None] * half_widths
            speeds = piece.physical(quadrature_times.ravel(), piece.sol.sol(quadrature_times.ravel()))[0]
            square = (speeds.reshape(quadrature_times.shape) * RPM - ref)**2
            record['rmse_rpm'] = float(sqrt(np.dot(weights @ square, half_widths) / t1))
        piece_metrics.append(record)
    summary['segments'] = piece_metrics
    summary['peak_abs_current_a'] = max(piece['peak_abs_current_a'] for piece in piece_metrics)
    summary['min_current_margin_a'] = case.motor.ceiling - summary['peak_abs_current_a']
    summary['peak_abs_voltage_v_sampled'] = float(np.max(np.abs(arrays['uq_v'])))
    start = next((piece for piece in piece_metrics if piece['segment'] == 'startup'), {})
    step = next((piece for piece in piece_metrics if piece['segment'] == 'load_step'), {})
    sinus = next((piece for piece in piece_metrics if piece['segment'] == 'sinusoid'), {})
    summary['published_metric_comparison'] = {
        'overshoot_rpm': max(0.0, start['max_speed_rpm'] - ref) if start else None,
        'settling_s': start.get('time_to_stay_s'), 'peak_current_a': start.get('peak_abs_current_a'),
        'recovery_s': step.get('time_to_stay_s'), 'rmse_rpm': sinus.get('rmse_rpm')}
    dt = np.diff(arrays['t_s'])
    clipped = arrays['voltage_clip'] != 0
    summary['voltage_clip_duration_s_sampled'] = float(np.dot(dt, (clipped[:-1] + clipped[1:].astype(float)) * 0.5))
    summary['rhs_evaluations'] = sum(int(piece.sol.nfev) for piece in run.pieces)
    return summary


def compare_solvers(main, other, main_metrics, ref_metrics, case, tolerances):
    dt = tolerances['sampling_s']
    start, stop = case.segments[0].start, min(main['t_s'][-1], other['t_s'][-1])
    grid = start + np.arange(int((stop - start) / dt) + 1) * dt
    knots = [segment.start for segment in case.segments if segment.start <= stop]
    t = np.unique(np.r_[grid[grid <= stop], knots, stop])

    def max_difference(column):
        return float(np.max(np.abs(np.interp(t, main['t_s'], main[column]) - np.interp(t, other['t_s'], other[column]))))

    speed = max_difference('speed_rpm')
    iq = max_difference('iq_a')
    peak = abs(main_metrics['peak_abs_current_a'] - ref_metrics['peak_abs_current_a'])
    delta_t = abs(main_metrics['end_s'] - ref_metrics['end_s'])
    status = main_metrics['status'] == ref_metrics['status'] and main_metrics['status'] != 'numerical_failure'
    m, bound = case.motor, case.voltage_limit
    if bound is not None:
        status &= all(np.max(np.abs(arr['uq_v'])) <= bound + 1e-10 for arr in (main, other))
    if case.controller.barrier_l:
        status &= max(main_metrics['peak_abs_current_a'], ref_metrics['peak_abs_current_a']) <= m.ceiling + 1e-8
    event_residuals = []
    if main_metrics['status'] == 'near_boundary':
        terminal = m.ceiling - tolerances['boundary_margin_a']
        event_residuals = [float(abs(abs(arr['iq_a'][-1]) - terminal)) for arr in (main, other)]
        status &= max(event_residuals) < 3e-8
    return {'passed': bool(status and speed <= tolerances['check_max_speed_difference_rpm']
                           and max(iq, peak) <= tolerances['check_max_current_difference_a']
                           and delta_t <= tolerances['check_boundary_time_difference_s']),
            'max_sampled_speed_difference_rpm': speed, 'max_sampled_iq_difference_a': iq,
            'max_dense_peak_current_difference_a': peak, 'end_time_difference_s': delta_t,
            'primary_status': main_metrics['status'], 'independent_status': ref_metrics['status'],
            'event_residual_a': event_residuals}


def capacity(motor, controller):
    m = motor
    w = m.reference
    torques = [1.0, 2.8, 3.0, 3.2]
    k1_min = 1 / (2 * m.friction)
    k1_max = sqrt((controller.p / m.inertia - 1) * m.inductance)
    k2_min = controller.p * m.inductance / (2 * m.inertia)
    return {
        'pole_pairs': m.pairs, 'torque_constant_nm_per_a': m.k,
        'load_at_ceiling_nm': m.torque_at_ceiling, 'proof_assumption_1_max_load_nm': m.torque_at_ceiling / 2,
        'eq16_over_eq17_penalty_at_same_l': 1 / m.k**2,
        'equilibria': [{'load_nm': torque, 'iq_a': (torque + m.friction * w) / m.k,
                       'uq_v': m.k * w + m.resistance * (torque + m.friction * w) / m.k,
                       'speed_at_current_ceiling_rpm': (m.k * m.ceiling - torque) * RPM / m.friction}
                      for torque in torques],
        'printed_proof_conditions': {'k1_lower': k1_min, 'k1_upper': k1_max, 'k2_lower': k2_min,
                                     'applied_k1': controller.k1, 'applied_k2': controller.k2,
                                     'feasible_k1_interval': k1_min < k1_max}}
