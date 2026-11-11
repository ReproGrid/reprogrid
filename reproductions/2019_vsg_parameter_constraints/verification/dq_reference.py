"""Independent dq circuit and state-space reference for the series-equivalent VSG."""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm


J = np.array([[0., -1.], [1., 0.]])
I = np.eye(2)


def line_response(s, model, angle=0.):
    """Differentiate measured power at the physical converter terminals."""
    e = model.voltage * np.array([np.cos(angle), np.sin(angle)])
    grid = np.array([model.voltage, 0.])
    x = model.omega * model.inductance
    current = np.linalg.solve(model.resistance * I + x * J, e - grid)
    s = np.atleast_1d(s)
    impedance = (model.resistance + s[:, None, None] * model.inductance) * I + x * J
    excitation = np.broadcast_to((J @ e)[:, None], (len(s), 2, 1))
    delta_current = np.linalg.solve(impedance, excitation)[..., 0]
    gain = grid + (2 * model.grid_resistance + s[:, None] * model.grid_inductance) * current
    return 1.5 * np.sum(gain * delta_current, axis=1)


def state_matrices(model):
    """No-load linearization in physical state coordinates."""
    # States are reference phase, frequency, terminal phase, terminal phase
    # rate, voltage-loop integral, and (for L > 0) grid d/q current.
    n = 7 if model.inductance > 0 else 5
    a = np.zeros((n, n))
    a[0, 1] = 1.
    a[1, 1] = -model.damping / model.inertia
    a[2, 3] = 1.
    cap_tau = model.capacitance * model.current_tau
    voltage_gain = model.voltage_kp / cap_tau
    a[3, [0, 2, 3, 4]] = [voltage_gain, -voltage_gain, -1 / model.current_tau, model.voltage_ki / cap_tau]
    a[4, [0, 2]] = [1., -1.]
    b = np.zeros(n)
    b[1] = 1 / model.inertia
    c = np.zeros(n)
    if n == 7:
        a[5:, 5:] = -model.resistance / model.inductance * I - model.omega * J
        a[6, 2] = model.voltage / model.inductance
        c[5] = 1.5 * model.voltage
        a[1] -= c / model.inertia
    return a, b, c


def closed_response(model, s):
    a, b, c = state_matrices(model)
    q = np.atleast_1d(s)
    lhs = q[:, None, None] * np.eye(len(a)) - a
    rhs = np.broadcast_to(b[:, None], (len(q), len(b), 1))
    return (np.linalg.solve(lhs, rhs)[..., 0]) @ c


def linear_step(model, time_s, step_w):
    a, b, c = state_matrices(model)
    n = len(a)
    augmented = np.zeros((n + 1, n + 1))
    augmented[:n, :n] = a
    augmented[:n, n] = b * step_w
    update = expm(augmented * (time_s[1] - time_s[0]))
    state = np.r_[np.zeros(n), 1.]
    power = np.empty(len(time_s))
    for k in range(len(power)):
        power[k] = c @ state[:n]
        state = update @ state
    return power


def nonlinear_step(model, time_s, step_w, rtol, atol):
    """Retain trigonometry and the instantaneous physical-terminal power."""
    if model.inductance <= 0:
        raise ValueError("Use the algebraic line limit for zero total inductance.")

    def current_and_power(x):
        angle, current = x[2], x[5:]
        emf_minus_grid = model.voltage * np.array([-2 * np.sin(angle / 2)**2, np.sin(angle)])
        current_rate = (emf_minus_grid - model.resistance * current) / model.inductance - model.omega * J @ current
        power = 1.5 * (model.voltage * current[0] + model.grid_resistance * (current @ current)
                       + model.grid_inductance * (current @ current_rate))
        return current_rate, power

    def rhs(_, x):
        _, frequency, _, phase_rate, integral, _, _ = x
        angle_error = x[0] - x[2]
        i_rate, power = current_and_power(x)
        acceleration = (step_w - power - model.damping * frequency) / model.inertia
        phase_acceleration = ((model.voltage_kp * angle_error + model.voltage_ki * integral) / model.capacitance
                              - phase_rate) / model.current_tau
        return np.r_[frequency, acceleration, phase_rate, phase_acceleration, angle_error, i_rate]

    sol = solve_ivp(rhs, (time_s[0], time_s[-1]), np.zeros(7), t_eval=time_s, method="DOP853",
                    rtol=rtol, atol=atol, max_step=0.002)
    if not sol.success or len(sol.t) != len(time_s):
        raise RuntimeError(sol.message)
    return np.array([current_and_power(x)[1] for x in sol.y.T]), sol.y[2]
