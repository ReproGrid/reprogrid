"""Reproduce Fan and Wang's scalar P–angle model with circular current limiting."""

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.optimize import brentq, minimize_scalar

CASE = Path(__file__).resolve().parents[1]
PERIOD = 2 * np.pi


@dataclass(frozen=True)
class Circuit:
    resistance: float
    reactance: float
    reference_v: float
    current_limit: float

    def observe(self, angle, grid_v):
        grid = grid_v * np.exp(-1j * np.asarray(angle))
        voltage_difference = self.reference_v - grid
        total_r = np.maximum(self.resistance, np.sqrt(np.maximum(
            0., (abs(voltage_difference) / self.current_limit)**2 - self.reactance**2
        )))
        limiter_r = total_r - self.resistance
        current = voltage_difference / (total_r + 1j * self.reactance)
        terminal = self.reference_v - limiter_r * current
        p = (terminal * current.conjugate()).real
        return p, abs(current), limiter_r

    def formula_power(self, angle, grid_v):
        # Eqs. (5)–(6). On the unsaturated branch Re = 0.
        v = self.reference_v
        squared_difference = v*v - 2*v*grid_v*np.cos(angle) + grid_v*grid_v
        total_r = np.maximum(self.resistance, np.sqrt(np.maximum(
            0., squared_difference/self.current_limit**2 - self.reactance**2
        )))
        return ((total_r*(v*v-v*grid_v*np.cos(angle)) + self.reactance*v*grid_v*np.sin(angle))
                / (total_r**2 + self.reactance**2) - (total_r-self.resistance)*self.current_limit**2)

    def kcl_reference(self, angle, grid_v):
        # Solve the terminal-current magnitude constraint directly, then use grid P plus copper loss.
        grid = grid_v * np.exp(-1j * angle)
        voltage_difference = self.reference_v - grid
        z = complex(self.resistance, self.reactance)
        if abs(voltage_difference/z) <= self.current_limit:
            current = voltage_difference/z
        else:
            bound = abs(voltage_difference)/self.current_limit
            inserted = brentq(lambda r: abs(voltage_difference/(z+r))-self.current_limit, 0., bound)
            current = voltage_difference/(z+inserted)
        return (grid*current.conjugate()).real + self.resistance*abs(current)**2


def equilibrium_angles(circuit, grid_v, p_ref):
    angles = np.linspace(0., PERIOD, 6001)
    errors = circuit.observe(angles, grid_v)[0] - p_ref
    crossings = np.flatnonzero(np.signbit(errors[:-1]) != np.signbit(errors[1:]))
    return [brentq(lambda d: circuit.observe(d, grid_v)[0]-p_ref, angles[i], angles[i+1])
            for i in crossings]


def simulate(circuit, cfg, start, duration, droop, solver):
    reference = cfg["table_i"]["active_reference_pu"]
    fault_v = cfg["fault_grid_voltage_pu"]
    dt = cfg["observation_step_s"]
    state = [start]
    segments = []
    for a, b, grid_v in ((0., duration, fault_v), (duration, duration+cfg["postfault_stop_s"], 1.)):
        def rhs(_, y):
            return [droop * (reference - circuit.observe(y[0], grid_v)[0])]

        result = solve_ivp(rhs, (a,b), state, dense_output=True, **solver)
        if not result.success:
            raise RuntimeError(result.message)
        grid = np.round(np.linspace(a, b, round((b-a)/dt)+1), 12)
        if a == 0:
            grid = grid[:-1]
        angle = result.sol(grid)[0]
        power, current, limiter_r = circuit.observe(angle, grid_v)
        segments.append(np.column_stack((grid, angle, power, current, limiter_r)))
        state = result.y[:, -1]
    return np.concatenate(segments)


def analyze(circuit, cfg, droop):
    ref = cfg["table_i"]["active_reference_pu"]
    fault_v = cfg["fault_grid_voltage_pu"]
    sep, uep = equilibrium_angles(circuit, 1., ref)
    angle_grid = np.linspace(0., PERIOD, 6001)
    fault_power = circuit.observe(angle_grid, fault_v)[0]
    i = int(np.argmax(fault_power))
    if i in (0, len(angle_grid)-1):
        raise RuntimeError("Maximum search requires an interior sampled peak")
    peak = minimize_scalar(lambda d: -circuit.observe(d, fault_v)[0],
                           bounds=(angle_grid[i-1], angle_grid[i+1]), method="bounded")
    if not peak.success or -peak.fun >= ref:
        raise RuntimeError("Fault-on trajectory may contain an equilibrium")

    threshold, _ = quad(lambda d: 1/(droop*(ref-circuit.observe(d, fault_v)[0])), sep, uep,
                        epsabs=1e-12, epsrel=1e-11)
    cases = [{"figure": None, "duration_s": cfg["diagnostic_duration_s"]}, *cfg["paper_test_s"]]
    data, traces = [], {}
    for entry in cases:
        duration = entry["duration_s"]
        trace = simulate(circuit, cfg, sep, duration, droop, cfg["solver"])
        refined = simulate(circuit, cfg, sep, duration, droop, cfg["refinement"])
        clear = np.searchsorted(trace[:, 0], duration)
        crossing = int(np.rint((trace[-1, 1]-sep)/PERIOD))
        end_error = abs(trace[-1, 1]-sep-crossing*PERIOD)
        inactive = np.flatnonzero((trace[:, 0] >= duration) & (trace[:, 4] == 0.))
        traces[f"{round(1000*duration)}ms"] = trace
        data.append({
            **entry, "clear_angle_rad": float(trace[clear, 1]),
            "basin_offset_revolutions": crossing,
            "end_angle_residual_rad": float(end_error),
            "first_postfault_limiter_exit_sample_s_after_clear": (
                float(trace[inactive[0], 0]-duration) if inactive.size else None
            ),
            "max_current_pu": float(max(trace[:, 3])),
            "max_refined_angle_difference_rad": float(np.max(abs(trace[:, 1]-refined[:, 1]))),
        })

    points = np.linspace(-np.pi, np.pi, 401)
    errors = []
    for grid_v in (1., fault_v):
        observed = circuit.observe(points, grid_v)[0]
        independent = np.array([circuit.kcl_reference(d, grid_v) for d in points])
        calculated = circuit.formula_power(points, grid_v)
        errors.extend((np.max(abs(independent-observed)), np.max(abs(calculated-observed))))
    return {
        "grid_resistance_pu": circuit.resistance, "grid_reactance_pu": circuit.reactance,
        "postfault_sep_rad": sep, "postfault_uep_rad": uep,
        "fault_peak_power_pu": float(-peak.fun),
        "same_basin_clearing_boundary_s": threshold,
        "droop_rad_s_per_pu": droop,
        "circuit_max_power_difference_pu": float(max(errors)),
        "cases": data,
    }, traces


def plot(circuit, cfg, summary, traces, output):
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0), layout="constrained")
    angles = np.linspace(0., PERIOD, 1600)
    for v, color in ((1., "#2166ac"), (cfg["fault_grid_voltage_pu"], "#a94c83")):
        axes[0].plot(angles, circuit.observe(angles, v)[0], color=color, label=f"Grid voltage {v:g} pu")
    axes[0].axhline(cfg["table_i"]["active_reference_pu"], ls=":", color=".3", label="Power reference")
    for point, face in (("sep", "#2166ac"), ("uep", "white")):
        axes[0].scatter(summary[f"postfault_{point}_rad"], cfg["table_i"]["active_reference_pu"],
                        c=face, edgecolors="#2166ac", zorder=3)
    for (label, trace), color in zip(traces.items(), ("#3c7c52", "#2166ac", "#a94c21"), strict=True):
        line, = axes[1].plot(trace[:, 0], trace[:, 1], color=color, label=label.replace("ms", " ms sag"))
        duration = int(label.removesuffix("ms"))/1000
        t_index = np.searchsorted(trace[:, 0], duration)
        axes[1].scatter(duration, trace[t_index, 1], color=line.get_color(), s=19, zorder=3)
    axes[1].axhline(summary["postfault_uep_rad"], ls=":", color=".3", label="Original-basin boundary")
    axes[0].set(xlabel="Converter-grid angle (rad)", ylabel="Terminal active power (pu)")
    axes[1].set(xlabel="Time from sag onset (s)", ylabel="Unwrapped angle (rad)", xlim=(0., 3.))
    for ax in axes:
        ax.grid(alpha=.22)
        ax.legend(fontsize=8)
    fig.savefig(output, dpi=165)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=CASE / "results")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    path = CASE / "configs/protocol.json"
    cfg = json.loads(path.read_text())
    table = cfg["table_i"]
    omega = PERIOD*table["frequency_hz"]
    s_base = table["active_reference_w"]/table["active_reference_pu"]
    v_base = table["voltage_phase_peak_v"]
    i_base = s_base/(1.5*v_base)
    z_base = v_base/i_base
    grid_parameters = {
        "si_derived": (table["grid_resistance_ohm"]/z_base, omega*table["grid_inductance_h"]/z_base),
        "printed_pu": (table["grid_resistance_printed_pu"], table["grid_reactance_printed_pu"]),
    }
    models, traces = {}, {}
    for label, (r, x) in grid_parameters.items():
        circuit = Circuit(r, x, table["voltage_reference_pu"], table["current_limit_pu"])
        models[label], case_traces = analyze(circuit, cfg, omega*table["power_droop_pu"])
        traces.update({f"{label}_{key}": value for key, value in case_traces.items()})
        if label == "si_derived":
            plot(circuit, cfg, models[label], case_traces, args.output / "power_angle_and_recovery.png")
    limits = cfg["acceptance"]
    checks = {
        "circuit_identities": all(m["circuit_max_power_difference_pu"] <= limits["circuit_pu"] for m in models.values()),
        "voltage_sag_removes_equilibria": all(m["fault_peak_power_pu"] < table["active_reference_pu"] for m in models.values()),
        "refined_angle_and_current_bound": all(
            case["max_refined_angle_difference_rad"] <= limits["angle_refinement_rad"]
            and case["max_current_pu"] <= table["current_limit_pu"]+limits["circuit_pu"]
            for m in models.values() for case in m["cases"]
        ),
        "settled_basin_matches_clearing_integral": all(
            case["end_angle_residual_rad"] <= limits["settled_angle_rad"]
            and case["basin_offset_revolutions"] == int(case["duration_s"] > m["same_basin_clearing_boundary_s"])
            for m in models.values() for case in m["cases"]
        ),
    }
    summary = {
        "doi": cfg["doi"], "protocol_sha256": sha256(path.read_bytes()).hexdigest(),
        "implementation_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "base": {"s_va": s_base, "i_peak_a": i_base, "z_ohm": z_base, "omega_rad_s": omega},
        "trace_columns": ["time_s", "angle_rad", "terminal_active_power_pu", "current_pu", "equivalent_limiter_r_pu"],
        "models": models, "checks": checks, "passed": all(checks.values()),
    }
    np.savez_compressed(args.output / "trajectories.npz", **traces)
    (args.output / "run_manifest.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"checks": checks, "same_basin_s": {k: v["same_basin_clearing_boundary_s"] for k,v in models.items()}}))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
