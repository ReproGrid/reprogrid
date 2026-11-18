"""Balanced EMT circuit of CIRED 2019 Figures 1–2, referred to one side."""

from dataclasses import dataclass

import numpy as np


PHASES = np.array([0.0, -2.0 * np.pi / 3.0, 2.0 * np.pi / 3.0])


@dataclass(frozen=True)
class Circuit:
    lf: float
    cf: float
    rd: float
    l2: float
    l1: float
    cl: float
    rf: float
    r2: float
    r1: float

    def matrices(self):
        """Inputs are inverter terminal and infinite-bus phase voltages.

        With cable C the state is [if, vf_cap, i2, v_line, i1].
        Without it the two series branch currents are identical.
        """
        lf, cf, rd = self.lf, self.cf, self.rd
        l2, l1, cl = self.l2, self.l1, self.cl
        rf, r2, r1 = self.rf, self.r2, self.r1
        if cl > 0:
            a = np.array([
                [-(rf + rd) / lf, -1 / lf, rd / lf, 0, 0],
                [1 / cf, 0, -1 / cf, 0, 0],
                [rd / l2, 1 / l2, -(rd + r2) / l2, -1 / l2, 0],
                [0, 0, 1 / cl, 0, -1 / cl],
                [0, 0, 0, 1 / l1, -r1 / l1],
            ])
            b = np.zeros((5, 2))
        else:
            lg, rg = l2 + l1, r2 + r1
            a = np.array([
                [-(rf + rd) / lf, -1 / lf, rd / lf],
                [1 / cf, 0, -1 / cf],
                [rd / lg, 1 / lg, -(rd + rg) / lg],
            ])
            b = np.zeros((3, 2))
        b[0, 0] = 1 / lf
        b[-1, 1] = -1 / (l1 if cl > 0 else l1 + l2)
        return a, b

    def outputs(self, state, grid):
        """Recover terminal and branch observables in three-wire abc."""
        i_filter, v_cap, i_trans = np.moveaxis(state[:, :3, :], 1, 0)
        v_filter = v_cap + self.rd * (i_filter - i_trans)
        if self.cl > 0:
            v_line, i_grid = np.moveaxis(state[:, 3:, :], 1, 0)
        else:
            i_grid = i_trans
            deriv = (v_filter - grid - (self.r1 + self.r2) * i_grid) / (self.l1 + self.l2)
            v_line = grid + self.r1 * i_grid + self.l1 * deriv
        return {
            "i_filter": i_filter,
            "v_filter": v_filter,
            "v_line": v_line,
            "i_cable": i_trans - i_grid,
            "i_grid": i_grid,
        }


def terminal_command(circuit, line_v_rms, frequency, power):
    """Calculate the imposed fundamental bridge phasor from grid P and Q=0."""
    s = 2j * np.pi * frequency
    z1 = circuit.r1 + s * circuit.l1
    z2 = circuit.r2 + s * circuit.l2
    zf = circuit.rf + s * circuit.lf
    yf = 1 / (circuit.rd + 1 / (s * circuit.cf))
    vg = line_v_rms / np.sqrt(3.0)
    i1 = power / (3.0 * vg)
    vl = vg + z1 * i1
    i2 = i1 + s * circuit.cl * vl
    vp = vl + z2 * i2
    i_filter = i2 + yf * vp
    return np.sqrt(2.0) * (vp + zf * i_filter), np.sqrt(2.0) * vg


def grid_voltage(time, frequency, fundamental_peak, fifth_relative):
    angle = 2 * np.pi * frequency * np.asarray(time)[..., None] + PHASES
    return fundamental_peak * (np.cos(angle) + fifth_relative * np.cos(5 * angle))


def command_voltage(time, frequency, bridge_peak_phasor):
    angle = 2 * np.pi * frequency * np.asarray(time)[..., None] + PHASES
    return np.real(bridge_peak_phasor * np.exp(1j * angle))
