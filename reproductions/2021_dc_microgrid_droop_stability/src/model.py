"""First-stage ideal droop-controlled branches and the common DC bus."""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class Plant:
    source: float
    resistance: np.ndarray
    inductance: np.ndarray
    capacitance: float

    @property
    def r_eq(self):
        return float(1 / np.sum(1 / self.resistance))

    @property
    def l_eq(self):
        return float(1 / np.sum(1 / self.inductance))

    @property
    def fold(self):
        return self.source**2 / (4 * self.r_eq)

    def parallel_equivalent(self):
        return Plant(self.source, np.array([self.r_eq]), np.array([self.l_eq]), self.capacitance)

    def equilibrium(self, power):
        discriminant = 1 - power / self.fold
        if discriminant < 0:
            return None
        voltage = self.source * (1 + np.sqrt(discriminant)) / 2
        return np.r_[(self.source - voltage) / self.resistance, voltage]

    def rhs(self, state, power, current_limit_knee=None):
        current, voltage = state[:-1], state[-1]
        denominator = voltage if current_limit_knee is None else max(voltage, current_limit_knee)
        return np.r_[(self.source - voltage - self.resistance * current) / self.inductance,
                     (current.sum() - power / denominator) / self.capacitance]

    def jacobian(self, voltage, power):
        count = len(self.resistance)
        jacobian = np.zeros((count + 1, count + 1))
        jacobian[:-1, :-1] = np.diag(-self.resistance / self.inductance)
        jacobian[:-1, -1] = -1 / self.inductance
        jacobian[-1, :-1] = 1 / self.capacitance
        jacobian[-1, -1] = power / (self.capacitance * voltage**2)
        return jacobian

    def stored_energy(self, state):
        return (np.square(state[..., :-1]) @ self.inductance + self.capacitance * state[..., -1]**2) / 2

    def power_balance(self, state, power):
        i = state[..., :-1]
        return self.source * i.sum(axis=-1) - np.square(i) @ self.resistance - power


def reduced_limit(source, resistance, inductance, capacitance):
    """Eq. (48) evaluated on the high-voltage branch of Eq. (32)."""
    ratio = resistance**2 * capacitance / inductance
    fold = source**2 / (4 * resistance)
    return fold * (1 if ratio >= 1 else 4 * ratio / (1 + ratio)**2)


def voltage_limits(source, resistance, bounds):
    voltage = source * np.asarray(bounds)
    return voltage * (source - voltage) / resistance


def integrate(plant, power, initial, interval, solver, stop, current_limit_knee=None):
    def field(_, state):
        return plant.rhs(state, power, current_limit_knee)

    def low_voltage(_, state):
        return state[-1] - stop

    def turn(_, state):
        return field(0., state)[-1]

    low_voltage.terminal = True
    low_voltage.direction = -1
    sol = solve_ivp(field, interval, initial, method="DOP853", dense_output=True, events=(low_voltage, turn), **solver)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol


def sampled(sol, sample_s):
    t0, t1 = sol.t[0], sol.t[-1]
    grid = t0 + sample_s * np.arange(int((t1 - t0) / sample_s) + 1)
    if t1 - grid[-1] > 1e-13:
        grid = np.r_[grid, t1]
    return grid, sol.sol(grid).T


def voltage_extrema(sol):
    values = np.r_[sol.y[-1, [0, -1]], np.asarray(sol.y_events[1]).reshape(-1, len(sol.y))[:, -1]]
    return float(min(values)), float(max(values))
