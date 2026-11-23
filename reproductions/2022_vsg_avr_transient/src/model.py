"""Single-converter VSG in Chen et al., Eqs. (1), (7), (13), (14), (29)."""

from dataclasses import dataclass
from math import cos, pi, sin, sqrt

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, minimize_scalar


@dataclass(frozen=True)
class Vsg:
    omega_n: float
    x: float
    pref: float
    qref: float
    vset: float
    dp: float
    dq: float
    inertia: float
    kq: float

    def power(self, angle, voltage, grid):
        p = grid * voltage * sin(angle) / self.x
        q = voltage * (voltage - grid * cos(angle)) / self.x
        return p, q

    def algebraic_voltage(self, angle, speed, grid, k=0.0):
        """Positive branch of the steady AVR equation at the given angle/speed."""
        a = self.dq / self.x
        b = 1.0 - a * grid * cos(angle)
        c = self.vset + self.dq * self.qref
        base = 2 * c / (b + sqrt(b*b + 4*a*c))
        if k == 0:
            return base
        slope = grid * sin(angle) / self.x
        input_power = self.pref - speed / self.dp
        sign = 1.0 if input_power >= slope*base else -1.0
        b += sign * k * slope
        c += sign * k * input_power
        root = sqrt(b*b + 4*a*c)
        return 2*c/(b + root) if b >= 0 else (-b + root)/(2*a)

    def equilibria(self, grid, constant_voltage=None):
        def mismatch(angle):
            voltage = self.algebraic_voltage(angle, 0, grid) if constant_voltage is None else constant_voltage
            return self.power(angle, voltage, grid)[0] - self.pref

        # Max P need not be at pi/2. The two reported sags straddle P(pi/2).
        return tuple(brentq(mismatch, a, b, xtol=1e-14) for a, b in ((0, pi/2), (pi/2, pi)))

    def simulate(self, grid, initial, saddle, k, mode, numerical, horizon, independent=False):
        """A finite AVR, a static AVR, or a voltage held at its pre-fault value."""
        def circuit_power(angle, voltage):
            if independent:
                internal = voltage * np.exp(1j * angle)
                current = (internal - grid) / (1j * self.x)
                pcc_power = internal * current.conjugate()
                return pcc_power.real, pcc_power.imag
            return self.power(angle, voltage, grid)

        def rhs(_, state):
            angle, speed = state[:2]
            voltage = state[2] if mode == "finite_avr" else (
                initial[2] if mode == "fixed_voltage" else self.algebraic_voltage(angle, speed, grid, k))
            p, q = circuit_power(angle, voltage)
            mismatch = self.pref - p - speed / self.dp
            rates = (self.omega_n * speed, mismatch / (2 * self.inertia))
            if mode == "finite_avr":
                return (*rates, self.kq * (self.vset + self.dq * self.qref - voltage - self.dq * q + k * abs(mismatch)))
            return rates

        def cross_saddle(_, state):
            return state[0] - saddle

        def first_turn(_, state):
            return state[1]

        def slip(_, state):
            return state[0] - initial[0] - 2*pi

        cross_saddle.direction = slip.direction = 1
        first_turn.direction = -1
        slip.terminal = True
        initial_state = initial if mode == "finite_avr" else initial[:2]
        sol = solve_ivp(rhs, (0, horizon), initial_state, dense_output=True,
                        events=(cross_saddle, first_turn, slip),
                        method="Radau" if independent else "DOP853",
                        rtol=numerical["solver_rtol"], atol=numerical["solver_atol"],
                        max_step=numerical["max_step_s"])
        if not sol.success:
            raise RuntimeError(sol.message)
        return sol

    def voltage_at(self, sol, time, grid, initial, k, mode):
        state = sol.sol(time)
        if mode == "finite_avr":
            return state[2]
        if mode == "fixed_voltage":
            return initial[2]
        return self.algebraic_voltage(*state[:2], grid, k)

    def voltage_peak(self, sol, grid, initial, k, mode):
        def voltage(time):
            return float(self.voltage_at(sol, time, grid, initial, k, mode))

        values = np.array([voltage(t) for t in sol.t])
        peaks = np.flatnonzero((values[1:-1] >= values[:-2]) & (values[1:-1] > values[2:])) + 1
        refined = [minimize_scalar(lambda t: -voltage(t), bounds=tuple(sol.t[j-1:j+2:2]),
                                   method="bounded", options={"xatol": 1e-10}) for j in peaks]
        point = int(values.argmax())
        peak_voltage, peak_time = float(values[point]), float(sol.t[point])
        for result in refined:
            if -result.fun > peak_voltage:
                peak_voltage, peak_time = float(-result.fun), float(result.x)
        return peak_voltage, peak_time
