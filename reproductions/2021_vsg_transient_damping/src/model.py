"""Single VSG with static Q-V droop and high-pass transient damping."""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import cumulative_simpson
from scipy.optimize import brentq, minimize_scalar


CAPTURED, UNRESOLVED, CROSSED = 1, 0, -1


@dataclass(frozen=True)
class VSG:
    p_ref: float
    q_ref: float
    v_ref: float
    v_before: float
    v_sag: float
    resistance: float
    reactance: float
    inertia: float
    damping: float
    voltage_droop: float
    gain_unit: float
    impedance_base: float

    @classmethod
    def from_protocol(cls, paper, setup, branch):
        omega = paper["nominal_angular_frequency_rad_s"]
        if "nominal_frequency_hz" in branch:
            omega = 2 * np.pi * branch["nominal_frequency_hz"]
        v_ref = paper["voltage_reference_v_peak_phase"]
        base = 1.5 * v_ref**2 / setup["power_base_w"]
        gain_unit = setup["power_base_w"] / omega
        reactance = omega * paper["grid_inductance_h"]
        if branch.get("use_exact_scr"):
            reactance = base / paper["figure4_short_circuit_ratio"]
        return cls(
            paper["active_reference_w"], paper["reactive_reference_var"], v_ref,
            paper["grid_voltage_before_v_peak_phase"], paper["grid_voltage_during_v_peak_phase"],
            branch.get("grid_resistance_ohm", paper["grid_resistance_ohm"]), reactance,
            paper["inertia_pu"] * gain_unit, paper["damping_pu"] * gain_unit,
            paper["reactive_droop_pu"] * v_ref / setup["reactive_base_var"], gain_unit, base,
        )

    def power(self, angle, grid_voltage):
        r, x = self.resistance, self.reactance
        scale = 1.5 / (r**2 + x**2)
        cosine, sine = np.cos(angle), np.sin(angle)
        a = self.voltage_droop * scale * x
        b = 1 - self.voltage_droop * scale * grid_voltage * (x * cosine + r * sine)
        c = self.v_ref + self.voltage_droop * self.q_ref
        # Rationalize the positive root of a*V^2+b*V-c=0.
        v = 2 * c / (b + np.sqrt(b**2 + 4 * a * c))
        common = v * (v - grid_voltage * cosine)
        quadrature = v * grid_voltage * sine
        return v, scale * (r * common + x * quadrature), scale * (x * common - r * quadrature)

    def equilibria(self, grid_voltage):
        def mismatch(angle):
            return self.power(angle, grid_voltage)[1] - self.p_ref

        peak = minimize_scalar(lambda d: -mismatch(d), bounds=(0, np.pi), method="bounded")
        if not peak.success or mismatch(peak.x) <= 0:
            raise ValueError("The specified power-angle branch has no pair of equilibria")
        return brentq(mismatch, -np.pi / 2, peak.x), brentq(mismatch, peak.x, np.pi)

    def rhs(self, state, alpha, gain):
        angle, speed, low_pass = state
        _, active, _ = self.power(angle, self.v_sag)
        high_pass = speed - low_pass
        speed_derivative = (self.p_ref - active - self.damping * speed - gain * high_pass) / self.inertia
        return np.array([speed, speed_derivative, alpha * high_pass])


def rk4_step(model, state, alpha, gain, step):
    k1 = model.rhs(state, alpha, gain)
    k2 = model.rhs(state + step / 2 * k1, alpha, gain)
    k3 = model.rhs(state + step / 2 * k2, alpha, gain)
    k4 = model.rhs(state + step * k3, alpha, gain)
    return state + step / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def waveforms(model, initial_angle, alpha, gains_pu, step, duration):
    gains = np.asarray(gains_pu) * model.gain_unit
    time = step * np.arange(round(duration / step) + 1)
    states = np.zeros((len(time), 3, len(gains)))
    states[0, 0] = initial_angle
    for i in range(1, len(time)):
        states[i] = rk4_step(model, states[i - 1], alpha, gains, step)
    return time, states


@dataclass(frozen=True)
class Potential:
    angle: np.ndarray
    energy: np.ndarray
    left_saddle: float
    right_saddle: float
    barrier: float

    @classmethod
    def build(cls, model, stable, right_saddle, nodes):
        left_saddle = right_saddle - 2 * np.pi
        angle = np.linspace(left_saddle, right_saddle, nodes)
        mismatch = model.power(angle, model.v_sag)[1] - model.p_ref
        energy = cumulative_simpson(mismatch, x=angle, initial=0.)
        energy -= np.interp(stable, angle, energy)
        return cls(angle, energy, left_saddle, right_saddle, min(energy[0], energy[-1]))

    def total(self, model, state, storage_coefficient):
        angle, speed, low_pass = state
        return np.interp(angle, self.angle, self.energy) + model.inertia / 2 * speed**2 + storage_coefficient * low_pass**2


def sweep(model, initial_angle, potential, alpha, gain_pu, step, horizon, guard):
    """Stop on a saddle crossing or a sublevel of the dissipative energy."""
    alpha, gain_pu = np.broadcast_arrays(alpha, gain_pu)
    shape = alpha.shape
    alpha = alpha.ravel()
    gain = gain_pu.ravel() * model.gain_unit
    storage = np.divide(gain, 2 * alpha, out=np.zeros_like(gain), where=alpha > 0)
    state = np.zeros((3, len(alpha)))
    state[0] = initial_angle
    active = np.arange(len(alpha))
    outcome = np.full(len(active), UNRESOLVED, dtype=np.int8)
    time = np.full(len(active), horizon)
    for i in range(1, round(horizon / step) + 1):
        state = rk4_step(model, state, alpha, gain, step)
        angle = state[0]
        crossed = (angle >= potential.right_saddle) | (angle <= potential.left_saddle)
        captured = (~crossed) & (potential.total(model, state, storage) < potential.barrier - guard)
        outcome[active[crossed]] = CROSSED
        outcome[active[captured]] = CAPTURED
        done = crossed | captured
        time[active[done]] = i * step
        keep = ~done
        if not keep.any():
            break
        if done.any():
            active, state, alpha, gain, storage = active[keep], state[:, keep], alpha[keep], gain[keep], storage[keep]
    return outcome.reshape(shape), time.reshape(shape)
