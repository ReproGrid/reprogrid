"""Fig. 1 averaged electrical plant and an explicit 10 kHz dual-PI controller."""

from dataclasses import dataclass
from math import cos, sin

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy.optimize import brentq


@dataclass(frozen=True)
class Electrical:
    omega: float
    r_grid: float
    x_grid: float
    x_filter: float
    b_cap: float

    def rhs(self, state, u, v_grid):
        i_f, v, i = state
        w = self.omega
        return w * np.array([(u - v) / self.x_filter - 1j*i_f,
                             (i_f - i) / self.b_cap - 1j*v,
                             (v - v_grid - self.r_grid*i) / self.x_grid - 1j*i])

    def energy(self, state):
        return (np.abs(state)**2 @ np.array([self.x_filter, self.b_cap, self.x_grid])) / (2*self.omega)


class HeldPlant:
    """Exact sample-to-sample map for the stated input holds in grid dq."""

    def __init__(self, circuit, dt, hold):
        self.electrical = circuit
        self.dt = dt
        self.hold_omega = circuit.omega if hold == "stationary" else 0.
        A = np.column_stack([circuit.rhs(z, 0, 0) for z in np.eye(3)])
        bus = np.column_stack([circuit.rhs(np.zeros(3), 1, 0), circuit.rhs(np.zeros(3), 0, 1)])
        # The physical held inverter vector rotates in grid dq. The grid
        # source and its magnitude remain constant within the sample.
        generator = np.zeros((5, 5), dtype=complex)
        generator[:3, :3] = A
        generator[:3, 3:] = bus
        generator[3, 3] = -1j*self.hold_omega
        transition = expm(dt * generator)[:3, :]
        self.F, self.Bu, self.Bg = transition[:, :3], transition[:, 3], transition[:, 4]
        self.hold_rotation = complex(cos(self.hold_omega * dt), -sin(self.hold_omega * dt))
        identity_minus_F = np.eye(3) - self.F
        self.steady_u = np.linalg.solve(identity_minus_F, self.Bu)
        self.steady_grid = np.linalg.solve(identity_minus_F, self.Bg)

    def step(self, state, u, v_grid):
        return self.F @ state + self.Bu*u + self.Bg*v_grid

    def static(self, v_cap, v_grid):
        command = (v_cap - self.steady_grid[1]*v_grid) / self.steady_u[1]
        return self.steady_u*command + self.steady_grid*v_grid, command

    def check(self):
        circuit = self.electrical
        trial = np.array([0.8 + 0.12j, 0.97 + 0.12j, 0.73 - 0.25j])
        command, grid = 1.04 - 0.11j, 0.5
        result = solve_ivp(lambda t, z: circuit.rhs(z, command*np.exp(-1j*self.hold_omega*t), grid),
                           (0., self.dt), trial, method="DOP853", rtol=2e-13, atol=1e-14)
        if not result.success:
            raise RuntimeError(result.message)
        step_error = float(np.max(np.abs(result.y[:, -1] - self.step(trial, command, grid))))
        z_dot = circuit.rhs(trial, command, grid)
        i_f, _, i = trial
        dt = 1e-7
        energy_rate = (circuit.energy(trial + dt*z_dot) - circuit.energy(trial - dt*z_dot))/(2*dt)
        electrical_power = (command*i_f.conjugate() - grid*i.conjugate()).real - circuit.r_grid*abs(i)**2
        return {"held_sample_independent_ode_error_pu": step_error,
                "instantaneous_energy_balance_error_pu": float(abs(energy_rate-electrical_power))}


@dataclass(frozen=True)
class State:
    circuit: np.ndarray
    delta: float
    voltage_integral: complex
    current_integral: complex
    power_filtered: float
    previous_command: complex

    def coordinates(self):
        return np.r_[self.circuit.real, self.circuit.imag, self.delta,
                     self.voltage_integral.real, self.voltage_integral.imag,
                     self.current_integral.real, self.current_integral.imag,
                     self.power_filtered, self.previous_command.real, self.previous_command.imag]

    @classmethod
    def from_coordinates(cls, row):
        return cls(row[:3] + 1j*row[3:6], row[6], complex(*row[7:9]), complex(*row[9:11]), row[11], complex(*row[12:14]))


class DigitalGFM:
    columns = ["time_s", "angle_rad", "active_power_pu", "reactive_power_pu", "converter_current_pu", "grid_current_pu",
               "capacitor_voltage_pu", "limited_current_reference_pu", "circular_limiter_active", "bridge_unlimited_command_pu",
               "grid_voltage_pu", "bridge_limited_command_pu"]

    def __init__(self, electrical, table, gains, scenario):
        self.dt = 1/gains["sampling_hz"]
        self.plant = HeldPlant(electrical, self.dt, scenario["voltage_hold"])
        self.v_ref = table["voltage_reference_pu"]
        self.p_ref = table["active_reference_pu"]
        self.i_max = table["current_limit_pu"]
        self.droop = electrical.omega * table["power_droop_pu"]
        pi_scale = electrical.omega if scenario["pi_time_base"] == "electrical" else 1.
        self.voltage_kp = gains["voltage_kp_pu"]
        self.voltage_integral_step = self.dt * pi_scale * gains["voltage_ki_pu"]
        self.current_kp = gains["current_kp_pu"]
        self.current_integral_step = self.dt * pi_scale * gains["current_ki_pu"]
        self.delay = scenario["delay"]
        self.retain_voltage_integral = scenario.get("retain_voltage_pi_state", False)
        self.bridge_limit = scenario.get("dc_bus_v", np.inf) / (np.sqrt(3)*table["voltage_phase_peak_v"])
        self.power_memory = np.exp(-2*np.pi*scenario["power_filter_hz"]*self.dt) if scenario["power_filter_hz"] else 0.

    def update(self, state, v_grid, t):
        e = self.plant.electrical
        rotation = complex(cos(state.delta), sin(state.delta))
        i_f, v, i = state.circuit * rotation.conjugate()
        p = (v*i.conjugate()).real
        q = (v*i.conjugate()).imag
        filtered = self.power_memory*state.power_filtered + (1-self.power_memory)*p
        phase = state.delta + self.dt*self.droop*(self.p_ref-filtered)

        voltage_error = self.v_ref - v
        base = self.voltage_kp*voltage_error + i + 1j*e.b_cap*v
        unlimited = base + state.voltage_integral
        if abs(unlimited) > self.i_max:
            voltage_integral = state.voltage_integral if self.retain_voltage_integral else 0j
            unlimited = base
        else:
            voltage_integral = state.voltage_integral + self.voltage_integral_step*voltage_error
        norm = abs(unlimited)
        limited = norm > self.i_max
        current_ref = self.i_max/norm * unlimited if limited else unlimited
        current_error = current_ref - i_f
        raw_voltage = rotation * (self.current_kp*current_error + state.current_integral + v + 1j*e.x_filter*i_f)
        command = raw_voltage if abs(raw_voltage) <= self.bridge_limit else raw_voltage/abs(raw_voltage)*self.bridge_limit
        current_integral = state.current_integral + self.current_integral_step*current_error
        applied = state.previous_command*self.plant.hold_rotation if self.delay else command
        next_circuit = self.plant.step(state.circuit, applied, v_grid)
        vector = (t, state.delta, p, q, abs(i_f), abs(i), abs(v), abs(current_ref), limited, abs(raw_voltage), v_grid, abs(command))
        return State(next_circuit, phase, voltage_integral, current_integral, filtered, command), vector

    def equilibrium(self):
        def power(delta):
            z, _ = self.plant.static(self.v_ref*np.exp(1j*delta), 1)
            return (z[1]*z[2].conjugate()).real

        delta = brentq(lambda delta: power(delta)-self.p_ref, 0, 0.35, xtol=2e-14)
        z, applied = self.plant.static(self.v_ref*np.exp(1j*delta), 1)
        command = applied/self.plant.hold_rotation if self.delay else applied
        rotation = np.exp(-1j*delta)
        i_f, v, i = z*rotation
        e = self.plant.electrical
        if abs(i_f) >= self.i_max:
            raise ValueError("No unsaturated nominal equilibrium for these settings")
        return State(z, delta, i_f - i - 1j*e.b_cap*v, command*rotation - v - 1j*e.x_filter*i_f,
                     self.p_ref, command)

    def linearize(self, state):
        x = state.coordinates()
        next_x, _ = self.update(state, 1.0, 0.)
        residual = float(np.max(np.abs(next_x.coordinates() - x)))

        def step(row):
            return self.update(State.from_coordinates(row), 1., 0.)[0].coordinates()

        perturb = np.eye(len(x))*1e-5
        jac = np.column_stack([(step(x + dx) - step(x - dx))/(2e-5) for dx in perturb])
        spectrum = np.linalg.eigvals(jac)
        return residual, spectrum

    def simulate(self, init, duration, end_after_sag, sag_v):
        clearing = round(duration/self.dt)
        steps = clearing + round(end_after_sag/self.dt) + 1
        out = np.zeros((steps, len(self.columns)))
        state = init
        for k in range(steps):
            v_grid = sag_v if k < clearing else 1.
            state, quantities = self.update(state, v_grid, k*self.dt)
            out[k] = quantities
            if not np.isfinite(state.delta) or np.max(np.abs(state.circuit)) > 100:
                return out[:k+1], clearing, True
        return out, clearing, False
