"""Reproduce the single-converter equal-area and first-swing CCT calculations."""

import argparse
from dataclasses import dataclass, replace
from hashlib import sha256
import json
from math import acos, asin, ceil, cos, hypot, sin, sqrt
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.optimize import brentq

CASE = Path(__file__).resolve().parents[1]
PROTOCOL = CASE / "configs/protocol.json"


@dataclass(frozen=True)
class PowerAngleCase:
    power: float
    unsaturated_peak: float
    saturated_peak: float
    omega_base: float
    inertia: float
    kp: float
    phi: float

    @property
    def delta0(self):
        return asin(self.power / self.unsaturated_peak)

    @property
    def saddle(self):
        return self.phi + acos(self.power / self.saturated_peak)

    def fault_state(self, duration):
        z = self.power * duration / (2 * self.inertia)
        delta = self.delta0 + self.omega_base * duration * (self.kp * self.power + z / 2)
        return np.array([delta, z])

    def postfault(self, _, state):
        delta, z = state
        error = self.power - self.saturated_peak * cos(delta - self.phi)
        return np.array([self.omega_base * (z + self.kp * error), error / (2 * self.inertia)])


def equal_area(model):
    """Printed Eqs. (7)–(9) and (10)–(12), with damping neglected."""
    delta = model.phi + asin(
        model.power * (model.delta0 - model.saddle) / model.saturated_peak
        + sin(model.saddle - model.phi)
    )
    duration = 2 * sqrt(model.inertia * (delta - model.delta0) / (model.omega_base * model.power))
    return delta, duration


def energy_boundary(model):
    """Independently balance the accelerating and decelerating areas by quadrature."""
    def energy_difference(duration):
        delta, _ = model.fault_state(duration)
        deceleration = quad(
            lambda angle: model.saturated_peak * cos(angle - model.phi) - model.power,
            delta, model.saddle,
        )[0]
        return model.power * (delta - model.delta0) - deceleration

    upper = 2 * sqrt(model.inertia * (model.saddle - model.delta0) / (model.omega_base * model.power))
    return brentq(energy_difference, 0., upper, xtol=1e-13)


def stable_manifold_cct(model, solver, offset):
    """Integrate the stable branch of the post-fault saddle backward to the fault curve."""
    slope = model.saturated_peak * sin(model.saddle - model.phi)
    a = model.omega_base * model.kp * slope
    b = slope / (2 * model.inertia)
    eigenvalue = (a - hypot(a, 2 * sqrt(model.omega_base * b))) / 2
    initial = [model.saddle - offset, offset * (a - eigenvalue) / model.omega_base]

    def fault_curve(_, state):
        delta, z = state
        duration = 2 * model.inertia * z / model.power
        return delta - model.fault_state(duration)[0]

    fault_curve.terminal = True
    result = solve_ivp(lambda t, x: -model.postfault(t, x), (0., 30.), initial,
                       events=fault_curve, **solver)
    if not result.success or not result.t_events[0].size:
        raise RuntimeError("Backward stable manifold did not intersect the fault trajectory")
    _, z = result.y_events[0][0]
    return float(2 * model.inertia * z / model.power)


def forward_first_swing(model, duration, numerics, *, dense=False):
    """Stop at the first angular turnaround or crossing of the upper saddle angle."""
    initial = model.fault_state(duration)

    def saddle_crossing(_, state):
        return state[0] - model.saddle

    def first_turn(t, state):
        return model.postfault(t, state)[0]

    saddle_crossing.terminal, saddle_crossing.direction = True, 1
    first_turn.terminal, first_turn.direction = True, -1
    if initial[0] >= model.saddle or first_turn(0., initial) <= 0.:
        raise ValueError("Selected forward check must start below the saddle with positive speed")
    solution = solve_ivp(model.postfault, (0., numerics["max_forward_time_s"]), initial,
                         events=(saddle_crossing, first_turn), dense_output=dense, **numerics["refined_solver"])
    if not solution.success or solution.status != 1:
        raise RuntimeError("Forward first swing did not reach a classification event")
    outcome = "crossed_upper_angle" if solution.t_events[0].size else "turned_before_upper_angle"
    return outcome, solution


def build_model(cfg, scenario, power):
    plant = cfg["plant"]
    ug, ec = plant["grid_voltage_pu"], plant["converter_voltage_pu"]
    return PowerAngleCase(
        power, ug * ec / (plant["x_c_pu"] + plant["x_g_pu"]), ug * plant["i_max_pu"],
        cfg["base"]["omega_b_rad_s"], scenario["inertia_s"], scenario["kp_pu"], scenario["phi_rad"],
    )


def compare_tables(cfg):
    rows = []
    n = cfg["numerics"]
    half_width = n["forward_bracket_half_width_ms"] / 1000
    for scenario in cfg["scenarios"]:
        for p, printed, dynamic in zip(cfg["powers_pu"], scenario["paper_predicted_ms"],
                                       scenario["paper_dynamic_ms"], strict=True):
            model = build_model(cfg, scenario, p)
            undamped = replace(model, kp=0.)
            angle, formula = equal_area(undamped)
            area = energy_boundary(undamped)
            manifold = stable_manifold_cct(undamped, n["solver"], n["saddle_offset_rad"])
            damped = stable_manifold_cct(model, n["solver"], n["saddle_offset_rad"])
            refined = stable_manifold_cct(model, n["refined_solver"], n["refined_saddle_offset_rad"])
            below, _ = forward_first_swing(model, damped - half_width, n)
            above, _ = forward_first_swing(model, damped + half_width, n)
            rows.append({
                "table": scenario["table"], "p_ref_pu": p, "paper_predicted_ms": printed,
                "paper_dynamic_ms": dynamic, "delta_critical_rad": angle, "formula_ms": formula * 1000,
                "quadrature_ms": area * 1000, "undamped_manifold_ms": manifold * 1000,
                "damped_manifold_ms": damped * 1000, "refined_damped_ms": refined * 1000,
                "forward_below": below, "forward_above": above,
            })
    return rows


def check_angle_bound(cfg):
    bound, plant = cfg["angle_bound"], cfg["plant"]
    ug, ec, xc = plant["grid_voltage_pu"], plant["converter_voltage_pu"], plant["x_c_pu"]
    psat = ug * plant["i_max_pu"]
    pmax, scr_max = bound["power_range_pu"][-1], bound["scr_range"][-1]

    def allowed(p, scr):
        return np.arccos(p / psat) + np.arcsin(p * (xc + 1 / scr) / (ug * ec))

    power_grid = np.linspace(*bound["power_range_pu"], 201)
    scr_grid = np.linspace(*bound["scr_range"], 241)
    sampled = allowed(power_grid[:, None], scr_grid)
    i, j = np.unravel_index(np.argmin(sampled), sampled.shape)
    rated = build_model(cfg, cfg["scenarios"][0], pmax)
    return {
        "minimum_rad": float(allowed(pmax, scr_max)),
        "minimizer": {"p_pu": pmax, "scr": scr_max},
        "grid_search": {"minimum_rad": float(sampled[i, j]), "p_pu": float(power_grid[i]), "scr": float(scr_grid[j])},
        "paper_fig9_min_rad": bound["paper_fig9_min_rad"],
        "rated_table_i_bound_rad": rated.delta0 + acos(pmax / psat),
        "rated_table_i_initial_delta_rad": rated.delta0,
        "rated_table_i_saturated_equilibria_on_ascending_power_branch": [
            {"phi_rad": phi, "delta_rad": phi - acos(pmax / psat)}
            for phi in (bound["paper_selected_rad"], bound["fig10_larger_angle_rad"])
        ],
    }


def save_trajectories(cfg, target):
    arrays, metadata = {}, []
    n = cfg["numerics"]
    scenarios = {row["table"]: row for row in cfg["scenarios"]}
    for item in cfg["rated_trajectories"]:
        model = build_model(cfg, scenarios[item["table"]], max(cfg["powers_pu"]))
        clear = item["duration_ms"] / 1000
        outcome, post = forward_first_swing(model, clear, n, dense=True)
        fault_t = np.linspace(0., clear, ceil(clear / n["archive_step_s"]), endpoint=False)
        post_t = np.linspace(0., post.t[-1], ceil(post.t[-1] / n["archive_step_s"]) + 1)
        delta, z = post.sol(post_t)
        key = f"{item['table']}_{item['duration_ms']}ms"
        arrays[f"{key}_time_s"] = np.r_[fault_t, clear + post_t]
        arrays[f"{key}_delta_rad"] = np.r_[model.fault_state(fault_t)[0], delta]
        arrays[f"{key}_p_pu"] = np.r_[np.zeros(fault_t.size), model.saturated_peak * np.cos(delta - model.phi)]
        arrays[f"{key}_z_pu"] = np.r_[model.fault_state(fault_t)[1], z]
        metadata.append({**item, "trace_key": key, "outcome": outcome, "phi_rad": model.phi,
                         "saddle_rad": model.saddle, "event_time_s": float(clear + post.t[-1]),
                         "event_delta_rad": float(delta[-1])})
    np.savez_compressed(target, **arrays)
    return arrays, metadata


def plot_results(rows, traces, examples, output):
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.7), layout="constrained", sharey=True)
    for axis, table in zip(axes, ("II", "III", "IV"), strict=True):
        points = [row for row in rows if row["table"] == table]
        power = [row["p_ref_pu"] for row in points]
        axis.plot(power, [row["formula_ms"] for row in points], color="#185a88", label="Equal-area formula")
        axis.plot(power, [row["damped_manifold_ms"] for row in points], color="#ad5325", ls="--",
                  label="Damped reduced model")
        axis.scatter(power, [row["paper_dynamic_ms"] for row in points], color=".22", marker="+",
                     label="Paper simulation")
        axis.set(title=f"Table {table}", xlabel="Active-power setpoint (pu)")
        axis.grid(alpha=.22)
    axes[0].set_ylabel("Critical clearing time (ms)")
    axes[-1].legend(fontsize=8)
    fig.savefig(output / "cct_comparison.png", dpi=165)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), layout="constrained", sharey=True)
    for axis, table in zip(axes, ("II", "IV"), strict=True):
        for item in examples:
            if item["table"] == table:
                key = item["trace_key"]
                t, delta = traces[f"{key}_time_s"] * 1000, traces[f"{key}_delta_rad"]
                line, = axis.plot(t, delta, label=f"{item['duration_ms']} ms sag")
                axis.scatter(t[-1], delta[-1], color=line.get_color(), s=16, zorder=3)
        example = next(row for row in examples if row["table"] == table)
        axis.axhline(example["saddle_rad"], ls=":", lw=1, color=".35", label="Upper saddle angle")
        axis.set(title=f"$\\phi={example['phi_rad']:g}$ rad", xlabel="Time from sag onset (ms)")
        axis.grid(alpha=.22)
        axis.legend(fontsize=8)
    axes[0].set_ylabel("Converter angle relative to grid (rad)")
    fig.savefig(output / "rated_first_swing.png", dpi=165)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=CASE / "results")
    args = parser.parse_args()
    cfg = json.loads(PROTOCOL.read_text())
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    rows = compare_tables(cfg)
    angle = check_angle_bound(cfg)
    arrays, trajectories = save_trajectories(cfg, output / "rated_trajectories.npz")
    tolerance = cfg["checks"]
    errors = {
        "formula_vs_printed_max_ms": max(abs(row["formula_ms"] - row["paper_predicted_ms"]) for row in rows),
        "formula_vs_energy_or_saddle_max_ms": max(
            abs(row["formula_ms"] - row[key]) for row in rows for key in ("quadrature_ms", "undamped_manifold_ms")
        ),
        "damped_refinement_max_ms": max(abs(row["damped_manifold_ms"] - row["refined_damped_ms"]) for row in rows),
    }
    checks = {
        "formula_matches_printed_precision": errors["formula_vs_printed_max_ms"] <= tolerance["paper_integer_ms_absolute_tolerance"],
        "independent_undamped_methods_agree": errors["formula_vs_energy_or_saddle_max_ms"] <= tolerance["analytic_vs_independent_energy_or_saddle_ms"],
        "damped_solution_refinement": errors["damped_refinement_max_ms"] <= tolerance["refinement_ms"],
        "forward_brackets": all(row["forward_below"] == "turned_before_upper_angle"
                                and row["forward_above"] == "crossed_upper_angle" for row in rows),
        "angle_bound_matches_fig9_and_sweep": (
            abs(angle["minimum_rad"] - angle["paper_fig9_min_rad"]) <= tolerance["phi_fig9_rad_absolute_tolerance"]
            and angle["minimum_rad"] == angle["grid_search"]["minimum_rad"]
        ),
    }
    report = {
        "doi": cfg["doi"], "protocol_sha256": sha256(PROTOCOL.read_bytes()).hexdigest(),
        "implementation_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": cfg["scope"], "checks": checks, "errors": errors,
        "table_rows": rows, "angle_bound": angle, "rated_trajectories": trajectories,
        "passed": all(checks.values()),
    }
    (output / "cct_results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    plot_results(rows, arrays, trajectories, output)
    print(json.dumps({"cases": len(rows), "passed": report["passed"], **errors}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
