"""PI and ESO designs in Eqs. (18)–(27), and their pulse-rate updates."""

import numpy as np
from scipy.optimize import brentq


OBSERVER_NAMES = ("PI", "ESO_C1", "ESO_C2")


def gains(kind, zeta, design_bandwidth):
    if kind == "PI":
        b = 2 * zeta**2 + 1
        omega = design_bandwidth / np.sqrt(np.hypot(b, 1.) + b)
        return np.array([2 * zeta * omega, omega**2])
    omega = 0.25648 * design_bandwidth
    if kind == "ESO_C1":
        return np.array([(2 * zeta + 1) * omega, (2 * zeta + 1) * omega**2, omega**3])
    if kind == "ESO_C2":
        return np.array([3 * zeta**2 * omega, 3 * zeta * omega**2, omega**3])
    raise ValueError(f"Unknown observer {kind}")


def transfer(gain, s):
    correction = np.polyval(gain, s)
    return correction / (s**len(gain) + correction)


def poles(gain, detector_gain=1.):
    return np.roots(np.r_[1., detector_gain * gain])


def bandwidth(gain):
    return float(brentq(lambda omega: abs(transfer(gain, 1j * omega)) - 1 / np.sqrt(2),
                        1e-7, max(gain[0] * 1e3, 10.)))


def state_matrix(gain):
    # Eq. (18) or (21), in angle, speed and optional acceleration states.
    size = len(gain)
    a = np.zeros((size, size))
    a[:, 0] = -gain
    a[np.arange(size - 1), np.arange(1, size)] = 1.
    return a, gain.copy(), np.eye(size)[0]


def advance(state, error, gain, dt):
    """Exact integration of the observer integrator chain under held error."""
    increment = -dt * error[..., None] * gain
    increment[:, 0] += dt * state[:, 1] - .5 * dt**2 * gain[1] * error
    if len(gain) == 3:
        increment[:, 0] += .5 * dt**2 * state[:, 2] - dt**3 / 6 * gain[2] * error
        increment[:, 1] += dt * state[:, 2] - .5 * dt**2 * gain[2] * error
    return state + increment
