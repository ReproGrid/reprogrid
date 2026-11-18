"""Triangular PWM edge locations and trapezoidal voltage interpolation."""

from dataclasses import dataclass

import numpy as np

from circuit import command_voltage


@dataclass(frozen=True)
class PWM:
    period: float
    dc_v: float
    edges: np.ndarray
    legs: np.ndarray
    signs: np.ndarray
    gate: np.ndarray
    bounds: np.ndarray
    integrated_gate: np.ndarray

    def node_gate(self, time):
        time = np.mod(time, self.period)
        return self.gate[np.searchsorted(self.edges, time, side="right")].copy()

    def integral_gate(self, time):
        """Exact on-time of each leg in [0,t], allowing cyclic extension."""
        cycles = np.floor(time / self.period)
        time = time - cycles * self.period
        idx = np.minimum(np.searchsorted(self.edges, time, side="right"), len(self.edges))
        return (cycles[:, None] * self.integrated_gate[-1]
                + self.integrated_gate[idx]
                + (time - self.bounds[idx])[:, None] * self.gate[idx])

    def voltage(self, gate):
        return self.dc_v * (gate - np.mean(gate, axis=-1, keepdims=True))

    def interpolate_nodes(self, time, step):
        """Replace the single nearest node for each edge per TPEL Eq. (1).

        A transition at time e represented at node t has fractional step
        1/2 + (t-e)/h. Add the difference from its binary step. Additive
        corrections preserve volt-seconds when two edges share a node.
        """
        gate = self.node_gate(time)
        position = (self.edges - time[0]) / step
        nearest = np.floor(position + 0.5).astype(int)
        signed_distance = nearest - position
        node = nearest % len(time)
        physical_bound = time[node] + (nearest // len(time)) * self.period
        binary_high_step = (physical_bound >= self.edges).astype(float)
        corrections = self.signs * (0.5 + signed_distance - binary_high_step)
        np.add.at(gate, (node, self.legs), corrections)
        return gate

    def fourier_voltage(self, freq, *, differential=True):
        """Continuous Fourier coefficients (not FFT samples) of bridge PWM."""
        freq = np.atleast_1d(freq)
        if np.any(freq <= 0):
            raise ValueError("Positive harmonic frequencies required.")
        omega = 2j * np.pi * freq[:, None]
        start = np.exp(-omega * self.bounds[:-1])
        end = np.exp(-omega * self.bounds[1:])
        voltage = self.voltage(self.gate) if differential else self.dc_v * self.gate
        return ((start - end) @ voltage) / (omega * self.period)


def triangular_pwm(grid_hz, carrier_hz, dc_v, bridge_command):
    period = 1.0 / grid_hz
    halves = round(2 * carrier_hz / grid_hz)
    if not np.isclose(halves, 2 * carrier_hz / grid_hz) or halves % 2:
        raise ValueError("Carrier and grid need an integer period ratio.")
    half = 0.5 / carrier_hz
    starts = np.arange(halves) * half
    duty = 0.5 + command_voltage(starts, grid_hz, bridge_command) / dc_v
    if np.any((duty <= 0) | (duty >= 1)):
        raise ValueError("The unmodified PWM reference is outside the linear range.")
    ascending = np.arange(halves) % 2 == 0
    fractional_times = np.where(ascending[:, None], duty, 1 - duty)
    times = (starts[:, None] + half * fractional_times).ravel()
    order = times.argsort(kind="stable")
    edges = times[order]
    legs = np.tile(np.arange(3), halves)[order]
    signs = np.repeat(np.where(ascending, -1, 1), 3)[order]
    transitions = np.zeros((len(edges), 3))
    transitions[np.arange(len(edges)), legs] = signs
    gate = np.concatenate((np.ones((1, 3)), 1 + np.cumsum(transitions, axis=0)))
    bounds = np.r_[0, edges, period]
    cumulative = np.vstack((np.zeros(3), np.cumsum(np.diff(bounds)[:, None] * gate, axis=0)))
    return PWM(period, dc_v, edges, legs, signs, gate, bounds, cumulative)
