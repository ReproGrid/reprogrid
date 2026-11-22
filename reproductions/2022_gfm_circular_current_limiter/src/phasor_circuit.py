"""Fig. 2 fundamental-frequency circuit with the converter-side capacitor retained."""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq, minimize_scalar


@dataclass(frozen=True)
class PhasorCircuit:
    r_grid: float
    x_grid: float
    b_cap: float
    reference_v: float
    current_limit: float

    def circuit(self, angle, grid_v):
        z = self.r_grid + 1j*self.x_grid
        grid = grid_v*np.exp(-1j*np.asarray(angle))
        y = 1/z + 1j*self.b_cap
        f = y*self.reference_v - grid/z
        over = np.maximum(0, np.abs(f/self.current_limit)**2 - 1)
        # |1 + R_e Y|^2 = |f/I_M|^2, or R_e=0 when the limiter is inactive.
        ge = y.real
        inserted = over / (ge + np.sqrt(ge**2 + abs(y)**2*over))
        i_f = f/(1+inserted*y)
        terminal = self.reference_v - inserted*i_f
        i = (terminal-grid)/z
        return terminal, i, i_f, inserted

    def power(self, angle, grid_v):
        terminal, grid_current, _, _ = self.circuit(angle, grid_v)
        return (terminal * grid_current.conjugate()).real

    def independent_error(self, grid_v):
        z = self.r_grid + 1j*self.x_grid
        errors = []
        for angle in np.linspace(-np.pi, np.pi, 161):
            grid = grid_v*np.exp(-1j*angle)

            def solve_at(series_resistor):
                # Unknowns [v, i, i_f], from the three simultaneous KVL/KCL rows.
                A = np.array([[1, -z, 0], [-1j*self.b_cap, -1, 1], [1, 0, series_resistor]])
                return np.linalg.solve(A, [grid, 0j, self.reference_v])

            def current_error(resistor):
                return abs(solve_at(resistor)[2]) - self.current_limit

            series = brentq(current_error, 0, 16) if current_error(0.) > 0 else 0.
            expected = solve_at(series)
            errors.append(float(np.max(np.abs(expected - np.array(self.circuit(angle, grid_v)[:3])))))
        return max(errors)


def scalar_bound(power, grid_v, p_ref, droop, durations):
    def stable_root(a):
        return power(a, 1.) - p_ref

    sep = brentq(stable_root, .10, .25)
    uep = brentq(stable_root, .30, .45)
    sample = np.linspace(0, 2*np.pi, 1001)
    peak_idx = int(np.argmax(power(sample, grid_v)))
    peak = minimize_scalar(lambda a: -power(a, grid_v),
                           bounds=(sample[peak_idx-1], sample[peak_idx+1]), method="bounded")
    p_max = -peak.fun
    if not peak.success or p_max >= p_ref:
        raise ValueError("This sag has an equilibrium and requires a different quadrature branch")

    def travel(a):
        return quad(lambda delta: 1/(droop*(p_ref-power(delta, grid_v))), sep, a,
                    epsabs=2e-12, epsrel=2e-12)[0]

    cct = travel(uep)
    cases = []
    for seconds in durations:
        clear = brentq(lambda a: travel(a) - seconds, sep, sep+1.5)
        cases.append({"duration_s": seconds, "clear_angle_rad": clear, "drift_rad": clear-sep,
                      "min_drift_from_max_fault_power_rad": droop*(p_ref-p_max)*seconds,
                      "above_first_unstable_angle": clear > uep,
                      "min_grid_phase_advance_at_clear_deg_for_same_basin": max(0., (clear-uep)*180/np.pi)})
    return {"stable_angle_rad": sep, "first_unstable_angle_rad": uep,
            "available_uniform_angle_margin_rad": uep-sep, "peak_fault_power_pu": p_max,
            "same_basin_boundary_s": cct, "cases": cases}
