"""Balanced three-phase averaged circuit with a sampled dq controller."""

from dataclasses import dataclass

import numpy as np


EYE = np.eye(2)
J = np.array([[0., -1.], [1., 0.]])


@dataclass(frozen=True)
class Circuit:
    omega: float
    filter_l: float
    filter_c: float
    filter_r: float
    grid_l: float
    grid_r: float
    grid_c: float
    current_kp: np.ndarray
    current_ki: float
    voltage_kp: np.ndarray
    voltage_ki: float


def make_circuit(protocol, case):
    thesis = protocol["thesis"]
    model = protocol["independent_benchmark"]
    gains = thesis["vsc2_pi_table_5_4"]
    omega = 2 * np.pi * thesis["fundamental_hz"]
    grid_z = model["line_to_line_rms_v"] ** 2 / (case["scr"] * thesis["base_power_w"])
    grid_r = grid_z / np.hypot(1., model["line_x_over_r"])
    return Circuit(
        omega, model["filter_l_h"], model["filter_c_f"], model["filter_series_r_ohm"],
        grid_r * model["line_x_over_r"] / omega, grid_r, case["parallel_c_uf"] * 1e-6,
        gains["current_kp_real"] * EYE + gains["current_kp_cross"] * J,
        gains["current_ki"],
        gains["voltage_kp_real"] * EYE + gains["voltage_kp_cross"] * J,
        gains["voltage_ki"],
    )


def plant(c, twin):
    # The twin is current driven. The physical circuit contains the grid
    # branch and an ideal shunt capacitor at the filter output.
    n = 4 if twin else 6
    select = np.eye(n)
    current, voltage = select[:2], select[2:4]
    total_c = c.filter_c if twin else c.filter_c + c.grid_c
    a = np.zeros((n, n))
    a[:2] = -(c.filter_r / c.filter_l * EYE + c.omega * J) @ current - voltage / c.filter_l
    a[2:4] = current / total_c - c.omega * J @ voltage
    pwm = np.zeros((n, 2))
    pwm[:2] = EYE / c.filter_l
    if twin:
        external = np.zeros((n, 2))
        external[2:4] = EYE / total_c
        port_current = None
    else:
        grid_current = select[4:6]
        a[2:4] -= grid_current / total_c
        a[4:6] = voltage / c.grid_l - (c.grid_r / c.grid_l * EYE + c.omega * J) @ grid_current
        external = pwm
        port_current = grid_current + c.grid_c / total_c * (current - grid_current)
    return a, pwm, external, current, voltage, port_current


def feedback(c, current, voltage, sample_step):
    # The controller states store the integral up to the left half-step.
    # The instantaneous half-step yields the trapezoidal PI transfer function.
    n = voltage.shape[1]
    voltage_error = np.hstack((-voltage, np.zeros((2, 4))))
    voltage_integral = np.eye(n + 4)[n:n + 2]
    current_integral = np.eye(n + 4)[n + 2:n + 4]
    current_error = ((c.voltage_kp + sample_step / 2 * c.voltage_ki * EYE) @ voltage_error
                     + c.voltage_ki * voltage_integral
                     - np.hstack((current, np.zeros((2, 4)))))
    command = ((c.current_kp + sample_step / 2 * c.current_ki * EYE) @ current_error
               + c.current_ki * current_integral)
    return command, np.vstack((voltage_error, current_error))


def continuous_closed_matrix(c, twin):
    a, pwm, _, current, voltage, _ = plant(c, twin)
    command, integral_rates = feedback(c, current, voltage, 0.)
    return np.vstack((np.hstack((a, np.zeros((len(a), 4)))) + pwm @ command, integral_rates))


def sampled_circuit(c, twin, sample_step, plant_step):
    a, pwm, external, current, voltage, port_current = plant(c, twin)
    n = len(a)
    substeps = round(sample_step / plant_step)
    if not np.isclose(substeps * plant_step, sample_step, rtol=1e-13, atol=0):
        raise ValueError("Control period must contain an integer number of plant steps.")
    eye = np.eye(n)
    # Trapezoidal integration with the control command and external input held
    # during a tick. The lifted matrix has the same state at every 40th step.
    fine = np.linalg.solve(eye - plant_step * a / 2,
                           np.column_stack((eye + plant_step * a / 2,
                                            plant_step * pwm, plant_step * external)))
    lifted = np.eye(n + 4)
    lifted[:n] = fine
    lifted = np.linalg.matrix_power(lifted, substeps)[:n]
    command, integral_rates = feedback(c, current, voltage, sample_step)
    a_closed = np.eye(n + 4)
    a_closed[:n] = np.hstack((lifted[:, :n], np.zeros((n, 4)))) + lifted[:, n:n + 2] @ command
    a_closed[n:] += sample_step * integral_rates
    b_closed = np.vstack((lifted[:, n + 2:], np.zeros((4, 2))))
    outputs = voltage if twin else np.vstack((voltage, port_current))
    return a_closed, b_closed, np.hstack((outputs, np.zeros((len(outputs), 4))))


def periodic_records(system, excitation):
    a, b, output = system
    radius = float(max(abs(np.linalg.eigvals(a))))
    if radius >= 1.:
        raise ValueError(f"No stable periodic operating record (spectral radius {radius:.8g}).")
    increments = excitation @ b.T
    terminal = np.zeros(len(a))
    for increment in increments:
        terminal = a @ terminal + increment
    initial = np.linalg.solve(np.eye(len(a)) - np.linalg.matrix_power(a, len(excitation)), terminal)
    state = initial.copy()
    states = np.empty((len(excitation), len(a)))
    for k, increment in enumerate(increments):
        states[k] = state
        state = a @ state + increment
    closure = np.linalg.norm(state - initial) / max(1., np.linalg.norm(initial))
    return states @ output.T, float(closure)


def sampled_frequency_response(system, frequency_hz, step_s):
    a, b, output = system
    z = np.exp(2j * np.pi * frequency_hz * step_s)
    dynamics = z[:, None, None] * np.eye(len(a)) - a
    state_response = np.linalg.solve(dynamics, np.broadcast_to(b, (len(z), *b.shape)))
    return output @ state_response


def continuous_impedance(c, frequency_hz):
    """Independent frequency-domain circuit equations used for the full GNC."""
    s = 2j * np.pi * np.asarray(frequency_hz)
    # Evaluate the limit at zero explicitly because both PI gains contain 1/s.
    nonzero_s = np.where(s == 0, 1., s)[:, None, None]
    p = s[:, None, None] * EYE + c.omega * J
    gc = c.current_kp + c.current_ki / nonzero_s * EYE
    gv = c.voltage_kp + c.voltage_ki / nonzero_s * EYE
    h = c.filter_r * EYE + c.filter_l * p + gc
    z_source = np.linalg.solve(EYE + gc @ gv + c.filter_c * (h @ p), h)
    z_source[s == 0] = 0.
    y_grid = np.linalg.inv(c.grid_r * EYE + c.grid_l * p) + c.grid_c * p
    return z_source, y_grid
