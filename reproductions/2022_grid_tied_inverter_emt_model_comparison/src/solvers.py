"""Periodic switching reference and fixed-step EMT integrators."""

from dataclasses import dataclass

import numpy as np

from circuit import PHASES, command_voltage, grid_voltage


@dataclass(frozen=True)
class Waveforms:
    time: np.ndarray
    signals: dict
    gate: np.ndarray | None
    switching_steps: int

SIGNALS = ("u_bridge", "i_filter", "v_filter", "v_line", "i_cable", "i_grid")


def exact_switching(circuit, pwm, time, grid_hz, grid_peak, grid_fifth):
    """Propagate each constant switching interval exactly, with analytic grid forcing."""
    a, b = circuit.matrices()
    roots, modes = np.linalg.eig(a)
    if np.any(roots.real >= 0):
        raise ValueError("A unique attracting periodic state is required.")
    modal_input = np.linalg.solve(modes, b[:, 0])
    fixed_state = (-modal_input / roots)[:, None] * pwm.voltage(pwm.gate)[:, None, :]
    flow = np.exp(np.diff(pwm.bounds)[:, None] * roots)
    q = np.zeros((len(roots), 3), dtype=complex)
    for gain, dc in zip(flow, fixed_state, strict=True):
        q = gain[:, None] * (q - dc) + dc
    periodic_start = q / (-np.expm1(roots * pwm.period))[:, None]
    z_edges = [periodic_start]
    for gain, dc in zip(flow, fixed_state, strict=True):
        z_edges.append(gain[:, None] * (z_edges[-1] - dc) + dc)
    tmod = np.mod(time, pwm.period)
    segment = np.searchsorted(pwm.edges, tmod, side="right")
    elapsed = (tmod - pwm.bounds[segment])[:, None]
    dc = fixed_state[segment]
    z = np.exp(elapsed * roots)[:, :, None] * (np.array(z_edges)[segment] - dc) + dc
    state = np.real(modes @ z)
    for harmonic, amplitude in ((1, grid_peak), (5, grid_peak * grid_fifth)):
        omega = harmonic * 2 * np.pi * grid_hz
        xg = amplitude * np.linalg.solve(1j * omega * np.eye(len(a)) - a, b[:, 1])
        phase = omega * time[:, None] + harmonic * PHASES
        state += np.real(xg[None, :, None] * np.exp(1j * phase[:, None, :]))
    grid = grid_voltage(time, grid_hz, grid_peak, grid_fifth)
    gate = pwm.node_gate(time)
    signals = {"u_bridge": pwm.voltage(gate)} | circuit.outputs(state, grid)
    return Waveforms(time, signals, gate, len(pwm.edges))


def periodic_recurrence(matrix, forcing):
    """Solve x[n+1] = matrix x[n] + forcing[n] with a periodic boundary."""
    q = np.zeros_like(forcing[0])
    for force in forcing:
        q = matrix @ q + force
    start = np.linalg.solve(np.eye(len(matrix)) - np.linalg.matrix_power(matrix, len(forcing)), q)
    out = np.empty((len(forcing), *start.shape))
    state = start
    for k, force in enumerate(forcing):
        out[k] = state
        state = matrix @ state + force
    return out


def fixed_step(circuit, pwm, model, step, grid_hz, grid_peak, grid_fifth,
               bridge_command, offset_fraction=0.0, solver="trapezoidal"):
    steps = round(pwm.period / step)
    if not np.isclose(steps * step, pwm.period, rtol=0, atol=1e-14):
        raise ValueError("The time step must divide the electrical period.")
    time = (np.arange(steps) + offset_fraction) * step
    if model == "SW":
        gate = pwm.node_gate(time)
        bridge = pwm.voltage(gate)
    elif model == "VI":
        gate = pwm.interpolate_nodes(time, step)
        bridge = pwm.voltage(gate)
    elif model == "AV":
        gate = None
        bridge = command_voltage(time, grid_hz, bridge_command)
    else:
        raise ValueError(f"Unknown bridge model {model}.")
    grid = grid_voltage(time, grid_hz, grid_peak, grid_fifth)
    a, b = circuit.matrices()
    eye = np.eye(len(a))
    drive = b[:, 0, None] * bridge[:, None, :] + b[:, 1, None] * grid[:, None, :]
    next_drive = np.roll(drive, -1, axis=0)
    if solver == "trapezoidal":
        implicit = np.linalg.solve(eye - 0.5 * step * a, eye)
        matrix = implicit @ (eye + 0.5 * step * a)
        forcing = implicit @ (0.5 * step * (drive + next_drive))
    elif solver == "sdirk2":
        # Stiffly accurate two-stage second-order DIRK. VI uses the linear
        # interpolant of its nodal voltage at the intermediate stage.
        gamma = 1.0 - 1.0 / np.sqrt(2.0)
        implicit = np.linalg.solve(eye - gamma * step * a, eye)
        if model == "AV":
            stage_bridge = command_voltage(time + gamma * step, grid_hz, bridge_command)
        else:
            stage_bridge = (1 - gamma) * bridge + gamma * np.roll(bridge, -1, axis=0)
        stage_grid = grid_voltage(time + gamma * step, grid_hz, grid_peak, grid_fifth)
        stage_drive = b[:, 0, None] * stage_bridge[:, None, :] + b[:, 1, None] * stage_grid[:, None, :]
        matrix = implicit @ (eye + (1 - gamma) * step * a @ implicit)
        forcing = implicit @ ((1 - gamma) * step * (implicit @ stage_drive) + gamma * step * next_drive)
    else:
        raise ValueError(f"Unknown solver {solver}.")
    state = periodic_recurrence(matrix, forcing)
    signals = {"u_bridge": bridge} | circuit.outputs(state, grid)
    return Waveforms(time, signals, gate, steps)
