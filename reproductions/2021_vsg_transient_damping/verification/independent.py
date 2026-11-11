"""Check the algebraic Q-V closure and Eq. (7) with a complex network model."""

import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.optimize import brentq


def phasor_power(angle, grid_voltage, reference_voltage, q_ref, droop, impedance):
    unit_phasor = np.exp(1j * angle)

    def power(voltage):
        pcc = voltage * unit_phasor
        return 1.5 * pcc * np.conj((pcc - grid_voltage) / impedance)

    voltage = brentq(lambda v: v - reference_voltage - droop * (q_ref - power(v).imag), 0., 3 * reference_voltage)
    apparent = power(voltage)
    return voltage, apparent.real, apparent.imag


def reference_run(power, p_ref, inertia, damping, alpha, gain, initial_angle, stable, saddle, sample_time, horizon):
    """Use the paper's HPF output x as the third state and adaptive DOP853."""
    def derivative(_, state):
        angle, speed, hpf_output = state
        acceleration = (p_ref - power(angle) - damping * speed - hpf_output) / inertia
        return speed, acceleration, gain * acceleration - alpha * hpf_output

    def right_saddle(_, state):
        return state[0] - saddle

    def left_saddle(_, state):
        return state[0] - saddle + 2 * np.pi

    right_saddle.direction = 1
    left_saddle.direction = -1
    trajectory_mode = sample_time is not None
    right_saddle.terminal = not trajectory_mode
    left_saddle.terminal = not trajectory_mode
    sol = solve_ivp(derivative, (0., horizon), [initial_angle, 0., 0.], t_eval=sample_time,
                    method="DOP853", events=(right_saddle, left_saddle), rtol=2e-11, atol=1e-11,
                    max_step=.15)
    if not sol.success:
        raise RuntimeError(sol.message)
    crossings = np.concatenate(sol.t_events)
    if len(crossings):
        outcome, crossing = -1, float(crossings.min())
    else:
        angle, speed, hpf_output = sol.y[:, -1]
        def potential(at):
            span = at - stable
            return span * quad(lambda f: power(stable + span * f) - p_ref, 0., 1., epsabs=1e-8)[0]

        stored = (gain * speed - hpf_output)**2 / (2 * alpha * gain) if alpha > 0 and gain > 0 else 0.
        energy = potential(angle) + inertia * speed**2 / 2 + stored
        barrier = min(potential(saddle), potential(saddle - 2 * np.pi))
        outcome, crossing = int(energy < barrier), None
    return sol.y.T, outcome, crossing
