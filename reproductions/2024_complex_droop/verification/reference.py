"""Direct two-axis circuit and controller equations, assembled independently."""
import math

import numpy as np

IDENTITY = np.eye(2)
J = np.array([[0., -1.], [1., 0.]])


def reference_rhs(p, vg, order):
    frequency = 2 * math.pi * p.base['nominal_frequency_hz']
    grid_frequency = p.base['grid_frequency_pu'] * frequency
    delta = frequency - grid_frequency
    lg = p.xg / frequency
    network = p.rg * IDENTITY + grid_frequency * lg * J
    admittance = np.linalg.inv(network)
    if p.phi_override is not None:
        phi = p.phi_override
    else:
        numerator = lg if p.angle == 'printed_literal' else grid_frequency * lg
        phi = math.atan2(numerator, p.rg)
    rotation = math.cos(phi) * IDENTITY + math.sin(phi) * J
    setpoint = rotation @ (p.p_star * IDENTITY - p.q_star * J) / p.base['v_star']**2
    eta = p.eta_over_omega0 * frequency
    grid = np.array([vg, 0.])
    cf = p.base['filter_susceptance_pu'] / frequency
    lf = p.base['filter_x_pu'] / frequency
    filter_y = p.base['filter_g_pu'] * IDENTITY + grid_frequency * cf * J
    filter_z = p.base['filter_r_pu'] * IDENTITY + grid_frequency * lf * J

    def rhs(t, x):
        reference_voltage = x[:2]
        terminal_voltage = x[4:6] if order >= 8 else reference_voltage
        current = admittance @ (terminal_voltage - grid) if order == 2 else x[2:4]
        dvoc = delta * (J @ reference_voltage)
        dvoc += eta * (setpoint @ reference_voltage - rotation @ current)
        dvoc += eta * p.alpha * (1 - (reference_voltage @ reference_voltage) / p.base['v_star']**2) * reference_voltage
        if order == 2:
            return dvoc
        network_current_rate = (terminal_voltage - grid - network @ current) / lg
        if order == 4:
            return np.concatenate((dvoc, network_current_rate))

        voltage_integrator = x[6:8]
        voltage_error = terminal_voltage - reference_voltage
        feedforward_current = filter_y @ terminal_voltage + current
        current_reference = feedforward_current - p.base['kp_v'] * voltage_error - p.base['kr_v_per_s'] * voltage_integrator
        filter_current = current_reference if order == 8 else x[8:10]
        voltage_rate = (filter_current - feedforward_current) / cf
        voltage_integrator_rate = delta * (J @ voltage_integrator) + voltage_error
        slow = (dvoc, network_current_rate, voltage_rate, voltage_integrator_rate)
        if order == 8:
            return np.concatenate(slow)

        current_integrator = x[10:12]
        current_error = filter_current - current_reference
        internal_voltage = (filter_z @ filter_current + terminal_voltage
                            - p.base['kp_c'] * current_error - p.base['kr_c_per_s'] * current_integrator)
        filter_current_rate = (internal_voltage - terminal_voltage - filter_z @ filter_current) / lf
        current_integrator_rate = delta * (J @ current_integrator) + current_error
        return np.concatenate((*slow, filter_current_rate, current_integrator_rate))

    return rhs


def jacobian_by_difference(rhs, x):
    """Central finite difference of the separately assembled vector field."""
    steps = 1e-5 * np.maximum(1., np.abs(x))
    columns = []
    for j, step in enumerate(steps):
        dx = np.zeros(len(x))
        dx[j] = step
        columns.append((rhs(0., x + dx) - rhs(0., x - dx)) / (2 * step))
    return np.column_stack(columns)


def quartic_coefficients(p, vs):
    """Schur determinant of the 4th-order system, in normalized time.

    For T=sI−ηB and U=sI+Z/Lg, elimination of grid current yields
    det(TU + ηR/Lg). All coefficients below use s divided by omega0.
    """
    voltage = np.array([vs.real, vs.imag])
    rotation = math.cos(p.phi) * IDENTITY + math.sin(p.phi) * J
    setpoint = rotation @ (p.p_star * IDENTITY - p.q_star * J) / p.v_star**2
    cubic_slope = (abs(vs)**2 * IDENTITY + 2 * np.outer(voltage, voltage)) / p.v_star**2
    top = p.eta_over_omega0 * (setpoint + p.alpha * (IDENTITY - cubic_slope))
    top += (p.omega_delta / p.omega0) * J
    bottom = (p.rg * IDENTITY + p.omega_g * p.lg * J) / (p.lg * p.omega0)
    first = bottom - top
    zero = -top @ bottom + p.eta_over_omega0 * rotation / (p.lg * p.omega0)
    mixed = first[0, 0] * zero[1, 1] + zero[0, 0] * first[1, 1]
    mixed -= first[0, 1] * zero[1, 0] + zero[0, 1] * first[1, 0]
    return np.array([1., np.trace(first), np.trace(zero) + np.linalg.det(first), mixed, np.linalg.det(zero)])


def routh_determinant(coeffs):
    _, a, b, c, d = coeffs
    return a * b * c - c**2 - a**2 * d
