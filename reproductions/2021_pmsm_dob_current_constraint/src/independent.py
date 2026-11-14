"""Independent Eq. (3) calculation in speed and shifted torque-error states."""

from math import ulp

import numpy as np
from scipy.integrate import solve_ivp

from model import Run, TrajectoryPiece


def observer_solution(segment, initial, decay):
    """Analytical solution with the same exact forcing as the PMSM model."""
    def particular(t):
        value = segment.torque_start + segment.slope * (t - segment.start - 1 / decay)
        for amplitude, frequency in segment.harmonics:
            phase = frequency * t
            value += (amplitude * (np.sin(phase) - (frequency / decay) * np.cos(phase))
                      / (1 + (frequency / decay)**2))
        return value

    transient = initial - particular(segment.start)

    def estimate(t):
        return particular(t) + transient * np.exp(-decay * (t - segment.start))

    return estimate


def error_voltage(case, e1, shifted_e2, state):
    """Compute and limit -u, the voltage variable of Eq. (3) with reversed sign."""
    m, c = case.motor, case.controller
    e2 = shifted_e2 - m.torque_at_ceiling
    f = (m.k**2 + m.resistance * m.friction) * m.reference - m.k**2 * e1 - m.resistance * e2
    drive = e2 + (state if c.p else 0.0)
    command = f + c.k1 * e1 + c.k2 * drive + c.ki * state
    limit = m.k * case.voltage_limit if case.voltage_limit is not None else None
    if c.barrier_l and drive:
        scale = m.k**2 if c.law == 'eq17' else 1.0
        product = max(0.0, (2 * m.k * m.ceiling - shifted_e2) * shifted_e2)
        numerator = command * product + c.barrier_l * scale * drive
        if limit is not None and abs(numerator) > limit * product:
            return (limit, 1, f) if numerator > 0 else (-limit, -1, f)
        command = numerator / (product or ulp((m.k * m.ceiling)**2))
    if limit is not None and abs(command) > limit:
        return (limit, 1, f) if command > 0 else (-limit, -1, f)
    return command, 0, f


def independent(case, numerics):
    m, c = case.motor, case.controller
    w0 = m.reference - case.speed_0
    q0 = m.k * (m.ceiling - case.iq_0)
    estimate_0 = case.auxiliary_0
    initial = [w0, q0] if c.p else [w0, q0, estimate_0]
    width = 2 * m.k * m.ceiling
    guard = m.k * numerics['boundary_margin_a']

    def positive_edge(_t, state):
        return state[1] - guard

    def negative_edge(_t, state):
        return width - state[1] - guard

    for edge in (positive_edge, negative_edge):
        edge.terminal = True
        edge.direction = -1
    pieces = []
    for segment in case.segments:
        estimate = observer_solution(segment, estimate_0, c.p / m.inertia) if c.p else None
        start = segment.start

        def physical(times, states, estimate=estimate, start=start):
            auxiliary = estimate(start + times) if estimate is not None else states[2]
            return np.array([m.reference - states[0], m.ceiling - states[1] / m.k, auxiliary])

        def rhs(t, state):
            e1, shifted_e2 = map(float, state[:2])
            auxiliary = float(estimate(start + t)) if estimate is not None else float(state[2])
            minus_u, clipped, f = error_voltage(case, e1, shifted_e2, auxiliary)
            speed_error_rate = (-m.friction * e1 + (shifted_e2 - m.torque_at_ceiling)
                                + segment.torque(start + t)) / m.inertia
            torque_error_rate = (f - minus_u) / m.inductance
            if estimate is not None:
                return [speed_error_rate, torque_error_rate]
            integral_rate = 0.0 if case.antiwindup and clipped * e1 > 0 else e1
            return [speed_error_rate, torque_error_rate, integral_rate]

        def jac(t, state):
            e1, shifted_e2 = map(float, state[:2])
            auxiliary = float(estimate(start + t)) if estimate is not None else float(state[2])
            _, clipped, _ = error_voltage(case, e1, shifted_e2, auxiliary)
            u_e1 = u_e2 = u_integral = 0.0
            if not clipped:
                u_e1, u_e2, u_integral = c.k1 - m.k**2, c.k2 - m.resistance, c.ki
                if c.barrier_l:
                    scale = m.k**2 if c.law == 'eq17' else 1.0
                    derivative = width - 2 * shifted_e2
                    product = max(0.0, (width - shifted_e2) * shifted_e2) or ulp((width / 2)**2)
                    drive = shifted_e2 - m.torque_at_ceiling + (auxiliary if c.p else 0.0)
                    u_e2 += c.barrier_l * scale * (1 - drive * derivative / product) / product
            e1_jac = [-m.friction / m.inertia, 1 / m.inertia]
            e2_jac = [(-m.k**2 - u_e1) / m.inductance, (-m.resistance - u_e2) / m.inductance]
            if estimate is not None:
                return [e1_jac, e2_jac]
            e1_jac.append(0.0)
            e2_jac.append(-u_integral / m.inductance)
            integral_e1 = 0.0 if case.antiwindup and clipped * e1 > 0 else 1.0
            return [e1_jac, e2_jac, [integral_e1, 0.0, 0.0]]

        atol = [numerics['atol_other_states'], m.k * numerics['atol_margin']]
        if estimate is None:
            atol.append(numerics['atol_other_states'])
        sol = solve_ivp(rhs, (0.0, segment.end - start), initial, method='BDF', jac=jac, dense_output=True,
                        events=(positive_edge, negative_edge) if c.barrier_l else None,
                        max_step=numerics['max_step_s'], rtol=numerics['independent_rtol'], atol=atol)
        pieces.append(TrajectoryPiece(segment, sol, physical, (guard, width - guard)))
        if sol.status != 0:
            return Run(pieces, 'near_boundary' if sol.status == 1 else 'numerical_failure', sol.message)
        initial = sol.y[:, -1]
        if estimate is not None:
            estimate_0 = float(estimate(start + sol.t[-1]))
    return Run(pieces, 'completed', '')


def printed_observer_check(motor, p, atol):
    """Numerically integrate z/p exactly as in Eq. (13) on a nonsingular startup segment."""
    torque = 1.0
    init = [motor.reference, -motor.reference]

    def rhs(_t, y):
        e1, z_over_p = y
        # Prescribed zero q-current for this verification, not a closed-loop run.
        e2 = motor.friction * motor.reference
        estimate = p * (e1 + z_over_p)
        common = motor.friction * e1 - e2
        return [(torque - common) / motor.inertia, (common - estimate) / motor.inertia]

    times = np.linspace(0.0, 12 * motor.inertia / p, 241)
    out = solve_ivp(rhs, (0.0, times[-1]), init, t_eval=times, method='Radau', rtol=1e-13, atol=atol)
    estimate13 = p * (out.y[0] + out.y[1])
    estimate14 = -torque * np.expm1(-p * times / motor.inertia)
    return float(np.max(np.abs(estimate13 - estimate14)))
