"""Stationary-frame matrix reference for the standstill current equations."""

import numpy as np
from scipy.linalg import expm


def held_alpha_beta(resistance, inductance, rotor, current, voltage, dt):
    c, s = rotor
    rotation = np.array([[c, -s], [s, c]])
    inv_l = np.linalg.inv(rotation @ np.diag(inductance) @ rotation.T)
    # Augment current with a constant input to integrate directly in alpha-beta.
    a = np.zeros((3, 3))
    a[:2, :2] = -resistance * inv_l
    a[:2, 2] = inv_l @ voltage
    return (expm(dt * a) @ np.r_[current, 1.])[:2]


def observer_response(a, b, c, frequencies):
    x = 1j * frequencies[:, None, None] * np.eye(len(a)) - a
    u = np.broadcast_to(b[:, None], (len(frequencies), len(a), 1))
    return np.linalg.solve(x, u)[..., 0] @ c
