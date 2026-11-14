"""Published per-unit power-angle model and augmented state matrices."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OperatingPoint:
    omega_n: float
    base_impedance_ohm: float
    reactance_ohm: np.ndarray
    transfer_capacity_pu: np.ndarray
    angle_rad: np.ndarray
    stiffness: np.ndarray


def operating_point(paper, bus_voltage, line_inductance_mh, active_power_pu):
    omega = 2 * np.pi * paper["frequency_hz"]
    base_impedance = paper["line_to_line_rms_base_v"]**2 / paper["power_base_per_converter_w"]
    reactance = np.asarray(paper["virtual_reactance_ohm"]) + omega * np.asarray(line_inductance_mh) / 1000
    capacity = np.asarray(paper["rated_source_voltage_pu"]) * bus_voltage * base_impedance / reactance
    if np.any(np.abs(active_power_pu) >= capacity):
        raise ValueError("Operating power requires an angle outside the positive stiffness branch")
    angle = np.arcsin(active_power_pu / capacity)
    return OperatingPoint(omega, base_impedance, reactance, capacity, angle, capacity * np.cos(angle))


def baseline(stiffness, inertia, droop, omega_n):
    """Eqs. (7)–(10), with x = [Δω1*, Δω2*, Δδ]."""
    h1, h2 = 2 * np.asarray(inertia)
    d1, d2 = droop
    k1, k2 = stiffness
    coupling = k1 * k2 / (k1 + k2)
    share = np.array([k1, k2]) / (k1 + k2)
    a = np.array([[-1 / (h1 * d1), 0, -coupling / h1],
                  [0, -1 / (h2 * d2), coupling / h2],
                  [omega_n, -omega_n, 0]])
    b = np.array([-share[0] / h1, -share[1] / h2, 0.])
    c = np.array([[0, 0, coupling], [0, 0, -coupling]])
    return a, b, c, share


def augmented(stiffness, inertia, droop, omega_n, controller):
    """Eqs. (26)–(30), with controller states ordered as in Eq. (27)."""
    a, b, c, d = baseline(stiffness, inertia, droop, omega_n)
    (k11, k12, k13, k14), (k21, k22, k23, k24) = controller
    if k11 == k13 == k21 == k23 == 0:
        return a, b, c, d
    h1, h2 = 2 * np.asarray(inertia)
    direct = np.array([[k13 / h1, 0], [0, k23 / h2], [0, 0]])
    into_swing = np.array([[1 / h1, -1 / h1, 0, 0], [0, 0, 1 / h2, -1 / h2], [0, 0, 0, 0]])
    acceleration = np.array([[0, 0, 0], [k11, 0, 0], [0, 0, 0], [0, k21, 0]])
    power = np.array([[k13 * k14, 0], [-k11 * k13 / h1, 0], [0, k23 * k24], [0, -k21 * k23 / h2]])
    filters = np.array([[-k14, 0, 0, 0], [k11 / h1, -k12 - k11 / h1, 0, 0],
                        [0, 0, -k24, 0], [0, 0, k21 / h2, -k22 - k21 / h2]])
    return (np.block([[a - direct @ c, into_swing], [acceleration @ a + power @ c, filters]]),
            np.r_[b - direct @ d, acceleration @ b + power @ d],
            np.pad(c, ((0, 0), (0, 4))), d)


def response(system, s):
    a, b, c, d = system
    s = np.asarray(s)
    work = s[..., None, None] * np.eye(len(a)) - a
    states = np.linalg.solve(work, np.broadcast_to(b[:, None], work.shape[:-1] + (1,)))[..., 0]
    return states @ c.T + d


def rightmost(poles, count):
    return poles[np.argsort(poles.real, kind="stable")[-count:][::-1]]
