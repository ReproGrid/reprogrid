"""Independent network and swing/filter realization for verification."""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


def phasor_slopes(paper, bus_voltage, line_mh, active):
    base_v = paper["line_to_line_rms_base_v"] / np.sqrt(3)
    frequency = paper["frequency_hz"]
    slopes, angles, numeric = [], [], []
    for x_virtual, inductance, source_v_pu, p_ref in zip(
        paper["virtual_reactance_ohm"], line_mh, paper["rated_source_voltage_pu"], active, strict=True
    ):
        z = 1j * (x_virtual + 2 * np.pi * frequency * inductance * 1e-3)
        bus = base_v * bus_voltage
        source = base_v * source_v_pu
        base_power = paper["power_base_per_converter_w"]

        def power(angle):
            voltage = source * np.exp(1j * angle)
            return (3 * voltage * ((voltage - bus) / z).conjugate()).real / base_power

        angle = brentq(lambda d: power(d) - p_ref, -np.pi / 2, np.pi / 2, xtol=1e-15)
        voltage = source * np.exp(1j * angle)
        current, dv = (voltage - bus) / z, 1j * voltage
        slopes.append((3 * (dv * current.conjugate() + voltage * (dv / z).conjugate())).real / base_power)
        dx = 1e-4
        numeric.append((power(angle - 2 * dx) - 8 * power(angle - dx) + 8 * power(angle + dx) - power(angle + 2 * dx)) / (12 * dx))
        angles.append(angle)
    return np.array(slopes), np.array(angles), np.array(numeric)


def swing_filter(stiffness, inertia, droop, omega, controller):
    """Assemble by evaluating the bus balance and two washout filters per converter."""
    k1, k2, k3, k4 = np.asarray(controller).T
    traditional = not np.any(np.asarray(controller)[:, [0, 2]])
    dimension = 3 if traditional else 7

    def evaluate(state, load):
        speed = state[:2]
        source_phase = np.array([state[2], 0.])
        bus_phase = (stiffness @ source_phase - load) / stiffness.sum()
        power = stiffness * (source_phase - bus_phase)
        damping_power = 0.
        if not traditional:
            speed_hp, power_hp = speed - state[3:5], power - state[5:7]
            damping_power = -k1 * speed_hp - k3 * power_hp
        acceleration = (damping_power - speed / droop - power) / (2 * inertia)
        derivative = np.r_[acceleration, omega * (speed[0] - speed[1])]
        if not traditional:
            derivative = np.r_[derivative, k2 * speed_hp, k4 * power_hp]
        return derivative, power

    columns = [evaluate(state, 0.) for state in np.eye(dimension)]
    a, c = (np.column_stack(v) for v in zip(*columns, strict=True))
    b, d = evaluate(np.zeros(dimension), 1.)
    return a, b, c, d


def nodal_response(stiffness, inertia, droop, omega, controller, s):
    """Solve the swing equations and power balance in the frequency domain."""
    k1, k2, k3, k4 = np.asarray(controller).T
    s = np.atleast_1d(s)
    pole = s[:, None]
    speed_coefficient = 2 * inertia * pole + 1 / droop + k1 * pole / (pole + k2)
    power_coefficient = 1 + k3 * pole / (pole + k4)
    system = np.zeros((len(s), 3, 3), dtype=complex)
    for index in range(2):
        system[:, index, index] = speed_coefficient[:, index] + power_coefficient[:, index] * stiffness[index] * omega / s
        system[:, index, 2] = -power_coefficient[:, index] * stiffness[index]
        system[:, 2, index] = stiffness[index] * omega / s
    system[:, 2, 2] = -stiffness.sum()
    rhs = np.broadcast_to(np.array([[0.], [0.], [1.]]), (len(s), 3, 1))
    result = np.linalg.solve(system, rhs)[..., 0]
    return (omega / pole * result[:, :2] - result[:, [2]]) * stiffness


def reference_step(system, time, size):
    a, b, c, d = system
    sol = solve_ivp(lambda _, x: a @ x + size * b, (0., time[-1]), np.zeros(len(a)),
                    method="DOP853", t_eval=time, rtol=2e-11, atol=1e-13, max_step=.02)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y.T @ c.T + size * d
