"""Reproduce the scalar PSC dynamics using a nodal AC network and event ODEs."""
import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad, solve_ivp


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/protocol.json"


@dataclass(frozen=True)
class Network:
    Y: np.ndarray
    voltage: float
    grid_voltage: float
    residual: float

    def current(self, angle):
        return self.Y[0, 0] * self.voltage * np.exp(1j * angle) + self.Y[0, 2] * self.grid_voltage

    def power(self, angle):
        voltage = self.voltage * np.exp(1j * angle)
        return np.real(voltage * np.conj(self.current(angle)))

    @property
    def power_max(self):
        return float(self.power(np.pi / 2))

def network(reactance, line2, fault, cfg):
    # Nodes PCC, sending bus, infinite bus. Ground has fixed zero voltage.
    Y = np.zeros((3, 3), dtype=complex)
    edges = [(0, 1, reactance["T"]), (1, 2, reactance["line1"])]
    if line2:
        edges.append((1, 2, reactance["line2"]))
    for a, b, x in edges:
        admittance = 1 / (1j * x)
        Y[a, a] += admittance
        Y[b, b] += admittance
        Y[a, b] -= admittance
        Y[b, a] -= admittance
    if fault:
        Y[1, 1] += 1 / (1j * reactance["ground"])
    boundary_to_sending = -Y[1, [0, 2]] / Y[1, 1]
    residual = np.max(np.abs(Y[1, [0, 2]] + Y[1, 1] * boundary_to_sending))
    reduced = Y.copy()
    reduced[:, [0, 2]] += Y[:, 1, None] * boundary_to_sending
    return Network(reduced, cfg["base"]["PCC_voltage_pu"], cfg["base"]["grid_voltage_pu"], float(residual))


def equilibria(net, reference):
    if not abs(reference) < net.power_max:
        return None
    stable = np.arcsin(reference / net.power_max)
    return float(stable), float(np.pi - stable)


def angle_event(target):
    def event(_time, state):
        return state[0] - target
    event.direction = 1
    return event


def solve_segment(net, start, stop, angle, gain, reference, solver, events=None):
    def derivative(_time, state):
        return [gain * (reference - net.power(state[0]))]
    result = solve_ivp(derivative, (start, stop), [angle], dense_output=True, events=events, **solver)
    if not result.success:
        raise RuntimeError(result.message)
    return result


def scenario_segments(name, nets, schedule):
    start = schedule["start_s"]
    fault = schedule["disturbance_s"]
    end = schedule["stop_s"]
    pre, during, post = nets
    if name == "type_i_line_open":
        return [(start, fault, pre), (fault, end, post)]
    if name == "type_ii_uncleared":
        return [(start, fault, pre), (fault, end, during)]
    duration = schedule["early_fault_duration_s"] if name.endswith("early") else schedule["late_fault_duration_s"]
    clearing = round(fault + duration, 12)
    return [(start, fault, pre), (fault, clearing, during), (clearing, end, post)]


def run_scenario(name, nets, cfg, solver):
    schedule = cfg["schedule"]
    base = cfg["base"]
    gain = base["Ki_rad_per_W_s"] * base["power_W"]
    reference = base["Pref_pu"]
    current_limit = base["current_limit_pu"]
    segments = scenario_segments(name, nets, schedule)
    count = round((schedule["stop_s"] - schedule["start_s"]) / cfg["numerics"]["sample_s"])
    time = np.round(np.linspace(schedule["start_s"], schedule["stop_s"], count + 1), 12)
    arrays = {k: np.empty(time.shape) for k in ("angle_rad", "angle_rate_rad_s", "power_pu", "current_pu")}
    angle = equilibria(nets[0], reference)[0]
    first_above = None
    crossing_times = []
    boundaries = []
    for index, (start, stop, net) in enumerate(segments):
        mask = (time >= start) & ((time <= stop) if index == len(segments) - 1 else (time < stop))
        def current_event(_time, state):
            return abs(net.current(state[0])) - current_limit
        current_event.direction = 1
        result = solve_segment(net, start, stop, angle, gain, reference, solver,
                               events=[current_event, angle_event(2*np.pi)])
        if first_above is None:
            if abs(net.current(angle)) >= current_limit:
                first_above = start
            elif len(result.t_events[0]):
                first_above = float(result.t_events[0][0])
        crossing_times.extend(result.t_events[1].tolist())
        at_times = result.sol(time[mask])[0]
        arrays["angle_rad"][mask] = at_times
        arrays["power_pu"][mask] = net.power(at_times)
        arrays["current_pu"][mask] = abs(net.current(at_times))
        arrays["angle_rate_rad_s"][mask] = gain * (reference - arrays["power_pu"][mask])
        boundaries.append({"time_s": start, "angle_rad": float(angle)})
        angle = float(result.y[0, -1])
    post_eq = equilibria(nets[-1], reference)
    if name == "type_ii_uncleared":
        basin = target = None
    else:
        last_start = boundaries[-1]["angle_rad"]
        basin = int(np.ceil((last_start - post_eq[1]) / (2*np.pi)))
        target = post_eq[0] + basin * 2*np.pi
    case = {"name": name, "segment_starts": boundaries, "basin_index": basin,
            "classification": "persistent_phase_drift" if basin is None else
            ("same_basin" if basin == 0 else "re_synchronized_after_phase_slip"),
            "clear_angle_deg": float(np.degrees(boundaries[-1]["angle_rad"])) if len(segments) == 3 else None,
            "final_angle_deg": float(np.degrees(angle)),
            "target_angle_deg": float(np.degrees(target)) if target is not None else None,
            "first_2pi_crossing_s": crossing_times[0] if crossing_times else None,
            "first_current_limit_crossing_s": first_above,
            "sampled_peak_current_pu": float(np.max(arrays["current_pu"]))}
    return {"time_s": time, **arrays}, case


def phase_portrait(cfg, networks, figures):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7), layout="constrained")
    gain = cfg["base"]["Ki_rad_per_W_s"] * cfg["base"]["power_W"]
    reference = cfg["base"]["Pref_pu"]
    descriptions = [("type_i", np.pi, "Type I"),
                    ("type_ii", np.pi, "Type II, first basin"),
                    ("type_ii", 2.5*np.pi, "Type II, next basin")]
    for ax, (kind, stop, title) in zip(axes, descriptions):
        angle = np.linspace(0, stop, 600)
        for label, net, color, style in zip(("Before", "During fault", "After"), networks[kind],
                                           ("#3466a9", "#9365ae", "#cb4c2f"), ("--", "-.", "-")):
            if net is None:
                continue
            ax.plot(angle/np.pi, gain*(reference - net.power(angle)), ls=style, color=color, label=label)
            eq = equilibria(net, reference)
            if label == "After" and eq:
                for offset in (0., 2*np.pi):
                    for index, point in enumerate(eq):
                        if point + offset <= stop:
                            ax.scatter((point+offset)/np.pi, 0, facecolors=color if index == 0 else "white",
                                       edgecolors=color, zorder=5)
        ax.axhline(0, color="0.4", linewidth=.8)
        ax.set(title=title, xlabel="Unwrapped power angle / π", ylabel="Angle rate (rad/s)")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle("Independent reconstruction of the PSC phase portraits")
    fig.savefig(figures / "phase_portraits.png", dpi=170)
    plt.close(fig)


def plot_trajectories(cfg, runs, figures, branch):
    fig, axes = plt.subplots(3, 4, figsize=(14, 7), sharex=True, layout="constrained")
    for column, (name, data, result) in enumerate(runs):
        for row, (field, ylabel) in enumerate((("angle_rad", "Angle / π"),
                                               ("power_pu", "P at PCC (pu)"),
                                               ("current_pu", "Current magnitude (pu)"))):
            ax = axes[row, column]
            scale = np.pi if field == "angle_rad" else 1.
            ax.plot(data["time_s"], data[field]/scale, color="#28649a")
            for event in result["segment_starts"][1:]:
                ax.axvline(event["time_s"], color=".45", linestyle=":", linewidth=.9)
            if row == 2:
                ax.axhline(cfg["base"]["current_limit_pu"], color="#cb4c2f", ls="--", lw=1)
            if column == 0:
                ax.set_ylabel(ylabel)
            if row == 0:
                ax.set_title(name)
            if row == 2:
                ax.set_xlabel("Time (s)")
            ax.grid(alpha=.2)
    fig.suptitle(f"Eq. (5), {branch}: ideal PCC voltage, no limiter dynamics")
    fig.savefig(figures / f"trajectories_{branch}.png", dpi=165)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--figures", type=Path, default=ROOT / "figures")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.figures.mkdir(parents=True, exist_ok=True)
    start_time = datetime.now(timezone.utc)
    timer = perf_counter()
    cfg = json.loads(PROTOCOL.read_text())
    source = json.loads((ROOT / "sources/source_manifest.json").read_text())
    numerics = cfg["numerics"]
    base = cfg["base"]
    reference = base["Pref_pu"]
    gain = base["Ki_rad_per_W_s"] * base["power_W"]
    target = cfg["paper_targets"]
    impedance_base = 3*base["phase_RMS_voltage_V"]**2/base["power_W"]
    reactance_scale = 2*np.pi*base["frequency_Hz"]/impedance_base
    results = {}
    checks = {}
    for branch, params in cfg["branches"].items():
        networks = {}
        for kind in ("type_i", "type_ii"):
            x = params[kind]
            if params["units"] == "inductance_H":
                x = {name: value*reactance_scale for name, value in x.items()}
            networks[kind] = (network(x, True, False, cfg),
                              network(x, True, True, cfg) if kind == "type_ii" else None,
                              network(x, False, False, cfg))
        before_i, _, after_i = networks["type_i"]
        before_ii, during_ii, after_ii = networks["type_ii"]
        pre_i = equilibria(before_i, reference)[0]
        post_i = equilibria(after_i, reference)[0]
        pre_ii = equilibria(before_ii, reference)[0]
        post_ii, cca = equilibria(after_ii, reference)
        cct, quad_error = quad(lambda angle: 1/(gain*(reference-during_ii.power(angle))), pre_ii, cca)
        hit = solve_segment(during_ii, 0, cfg["schedule"]["stop_s"], pre_ii, gain, reference,
                            numerics["primary"], events=[angle_event(cca)]).t_events[0][0]
        fault_min_rate = gain * (reference - during_ii.power_max)
        period = 2*np.pi / (gain*np.sqrt(reference**2 - during_ii.power_max**2))
        powers = {kind: [net.power_max if net else None for net in nets] for kind, nets in networks.items()}
        summary = {"type_i_pre_deg": float(np.degrees(pre_i)), "type_i_post_deg": float(np.degrees(post_i)),
                   "type_ii_pre_deg": float(np.degrees(pre_ii)), "type_ii_post_deg": float(np.degrees(post_ii)),
                   "type_ii_cca_deg": float(np.degrees(cca)), "type_ii_cct_s": cct, "cct_event_s": float(hit),
                   "cct_quadrature_error_estimate_s": quad_error, "fault_slip_period_s": float(period),
                   "fault_min_angle_rate_rad_s": fault_min_rate, "power_max_pu_pre_fault_post": powers,
                   "cases": {}}
        if branch == "table_pu":
            summary["printed_target_matches"] = {
                k: abs(summary[k] - target[k]) <= target[tol] for k, tol in
                (("type_i_pre_deg", "type_i_angle_rounding_tolerance_deg"),
                 ("type_i_post_deg", "type_i_angle_rounding_tolerance_deg"),
                 ("type_ii_cca_deg", "cca_rounding_tolerance_deg"),
                 ("type_ii_cct_s", "cct_rounding_tolerance_s"))}
            phase_portrait(cfg, networks, args.figures)
        base_checks = {
            "nodal_kcl": max(net.residual for group in networks.values() for net in group if net) < numerics["max_equilibrium_or_kcl_residual"],
            "equilibrium_residual": max(abs(net.power(angle)-reference) for net, angle in
                                        ((before_i, pre_i), (after_i, post_i), (before_ii, pre_ii),
                                         (after_ii, post_ii), (after_ii, cca))) < numerics["max_equilibrium_or_kcl_residual"],
            "fault_lacks_equilibrium": fault_min_rate > 0,
            "cct_event_agrees_with_quadrature": abs(hit-cct) < numerics["event_vs_integral_tolerance_s"]}
        runs = []
        for name, kind, expected_basin in (("type_i_line_open", "type_i", 0),
                                            ("type_ii_uncleared", "type_ii", None),
                                            ("type_ii_early", "type_ii", 0),
                                            ("type_ii_late", "type_ii", 1)):
            data, case = run_scenario(name, networks[kind], cfg, numerics["primary"])
            refined, check_case = run_scenario(name, networks[kind], cfg, numerics["refined"])
            differences = {field: float(np.max(np.abs(data[field]-refined[field]))) for field in data}
            case["max_refinement_differences"] = differences
            base_checks[f"{name}_integrator_agreement"] = max(differences.values()) < numerics["max_trajectory_difference"]
            base_checks[f"{name}_basin"] = case["basin_index"] == check_case["basin_index"] == expected_basin
            if expected_basin is not None:
                base_checks[f"{name}_terminal_angle"] = abs(case["final_angle_deg"]-case["target_angle_deg"]) < numerics["terminal_angle_tolerance_deg"]
            if name == "type_i_line_open":
                post_angle = data["angle_rad"][data["time_s"] >= cfg["schedule"]["disturbance_s"]]
                tol = numerics["monotonic_tolerance_rad"]
                base_checks["type_i_monotone_no_overshoot"] = bool(np.min(np.diff(post_angle)) >= -tol and np.max(post_angle) <= post_i + tol)
            if name == "type_ii_uncleared":
                post_rate = data["angle_rate_rad_s"][data["time_s"] >= cfg["schedule"]["disturbance_s"]]
                base_checks["uncleared_positive_phase_drift"] = bool(np.min(post_rate) > 0)
            file_stem = f"{branch}_{name}"
            np.savez_compressed(args.output / f"{file_stem}.npz", **data)
            np.savetxt(args.output / f"{file_stem}.csv", np.column_stack(list(data.values())), delimiter=",",
                       header=",".join(data), comments="", fmt="%.12g")
            case["npz"] = f"{file_stem}.npz"
            summary["cases"][name] = case
            runs.append((name, data, case))
        plot_trajectories(cfg, runs, args.figures, branch)
        checks[branch] = {k: bool(v) for k, v in base_checks.items()}
        results[branch] = summary
    result = {
        "doi": cfg["doi"], "scope": cfg["scope"], "source_sha256": source["sha256"],
        "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {name: version(name) for name in ("numpy", "scipy", "matplotlib")},
        "started_utc": start_time.isoformat(), "finished_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": perf_counter() - timer, "gain_per_s": gain, "impedance_base_ohm": impedance_base,
        "current_base_RMS_A": base["power_W"]/(3*base["phase_RMS_voltage_V"]),
        "checks": checks, "numerical_checks_passed": all(all(v.values()) for v in checks.values()),
        "branches": results}
    (args.output / "run_manifest.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("numerical_checks_passed", "wall_seconds", "checks")}))
    if not result["numerical_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
