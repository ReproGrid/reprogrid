"""Reference implementation for the same circuit using stored energy and flux."""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


def bus_equilibrium(source, resistance, power):
    conductance = np.sum(1 / resistance)

    def balance(voltage):
        return voltage * (source - voltage) * conductance - power

    if balance(source / 2) < 0:
        return None
    upper = source * (2 + np.sqrt(abs(power) / (source**2 * conductance)))
    return brentq(balance, source / 2, upper, xtol=1e-12)


def thresholds(source, resistance, inductance, capacitance):
    """From the network admittance, without using the reduced limit formula."""
    conductance = np.sum(1 / resistance)
    fold = source**2 * conductance / 4
    weights = 1 / resistance
    time_constants = inductance / resistance

    # With v~exp(jw t), the imaginary term at the bus is
    # w*(C - sum(L/(R**2 + (w*L)**2))).
    if capacitance >= weights @ time_constants:
        return fold, None

    def imaginary_over_frequency(w):
        return capacitance - np.sum(weights * time_constants / (1 + (w * time_constants)**2))

    upper = np.sqrt(np.sum(weights / time_constants) / capacitance)
    w = brentq(imaginary_over_frequency, 0, upper, xtol=1e-11)
    conductance_required = np.sum(weights / (1 + (w * time_constants)**2))
    voltage = source * conductance / (conductance + conductance_required)
    return voltage**2 * conductance_required, w


def reference_trace(source, resistance, inductance, capacitance, power, initial, interval, solver, stop, current_limit_knee=None):
    # Scale flux by L* and stored energy by C* so reference states
    # remain close to unit magnitude. Use q=(v/U)^2.
    current_base = source * np.sum(1 / resistance)
    initial_ref = np.r_[initial[:-1] / current_base, (initial[-1] / source)**2]

    def field(_, state):
        voltage = source * np.sqrt(state[-1])
        current = current_base * state[:-1]
        load = power if current_limit_knee is None else power * min(1., voltage / current_limit_knee)
        return np.r_[((source - voltage) - resistance * current) / (inductance * current_base),
                     2 * (voltage * current.sum() - load) / (capacitance * source**2)]

    def low_energy(_, state):
        return state[-1] - (stop / source)**2

    low_energy.terminal = True
    low_energy.direction = -1
    sol = solve_ivp(field, interval, initial_ref, method="Radau", dense_output=True, events=low_energy, **solver)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol, current_base


def decode(source, current_base, values):
    return np.column_stack((values[:-1].T * current_base, source * np.sqrt(values[-1])))
