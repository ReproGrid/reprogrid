"""Independent complex nodal verification of the continuous periodic solution."""

import numpy as np

from circuit import PHASES


def ac_response(circuit, frequency, bridge, grid):
    """Complex Fourier response from Kirchhoff current balance at two nodes."""
    s = 1j * 2 * np.pi * np.asarray(frequency)
    yf = 1.0 / (circuit.rf + s * circuit.lf)
    y2 = 1.0 / (circuit.r2 + s * circuit.l2)
    y1 = 1.0 / (circuit.r1 + s * circuit.l1)
    yd = 1.0 / (circuit.rd + 1.0 / (s * circuit.cf))
    yc = s * circuit.cl
    diagonal_filter = yf + y2 + yd
    diagonal_line = y2 + y1 + yc
    determinant = diagonal_filter * diagonal_line - y2 * y2
    vf = ((yf * diagonal_line)[:, None] * bridge + (y2 * y1)[:, None] * grid) / determinant[:, None]
    vl = ((y2 * yf)[:, None] * bridge + (diagonal_filter * y1)[:, None] * grid) / determinant[:, None]
    i_filter = yf[:, None] * (bridge - vf)
    i_grid = y1[:, None] * (vl - grid)
    i_2 = y2[:, None] * (vf - vl)
    i_cable = yc[:, None] * vl
    mismatch = max(np.max(abs(i_filter - i_2 - yd[:, None] * vf)),
                   np.max(abs(i_2 - i_grid - i_cable)))
    return {
        "u_bridge": bridge,
        "i_filter": i_filter,
        "v_filter": vf,
        "v_line": vl,
        "i_cable": i_cable,
        "i_grid": i_grid,
    }, float(mismatch)


def nodal_spectrum(circuit, pwm, frequencies, grid_hz, grid_peak, grid_fifth):
    bridge = pwm.fourier_voltage(frequencies)
    injected = np.zeros_like(bridge)
    for order, amplitude in ((1, grid_peak), (5, grid_peak * grid_fifth)):
        injected[frequencies == order * grid_hz] = 0.5 * amplitude * np.exp(1j * order * PHASES)
    return ac_response(circuit, frequencies, bridge, injected)


def measured_spectrum(wave, grid_hz, frequencies):
    n = len(wave.time)
    indices = np.rint(frequencies / grid_hz).astype(int)
    if np.any(indices >= n / 2):
        raise ValueError("Requested spectrum includes the Nyquist frequency or an aliased bin.")
    correction = np.exp(-2j * np.pi * frequencies * wave.time[0])[:, None] / n
    return {key: np.fft.rfft(value, axis=0)[indices] * correction
            for key, value in wave.signals.items()}


def observables(spectrum):
    """Phase-A current and AB voltages used in the paper-style comparison."""
    return {
        "u_bridge_ab": spectrum["u_bridge"][:, 0] - spectrum["u_bridge"][:, 1],
        "v_filter_ab": spectrum["v_filter"][:, 0] - spectrum["v_filter"][:, 1],
        "v_line_ab": spectrum["v_line"][:, 0] - spectrum["v_line"][:, 1],
        "i_filter_a": spectrum["i_filter"][:, 0],
        "i_cable_a": spectrum["i_cable"][:, 0],
        "i_grid_a": spectrum["i_grid"][:, 0],
    }
