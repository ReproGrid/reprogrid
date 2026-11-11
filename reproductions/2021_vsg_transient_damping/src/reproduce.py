"""Rebuild the VSG power-angle curves, trajectories and damping design regions."""

import argparse
import csv
from dataclasses import replace
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

import numpy as np
import scipy
from scipy.integrate import cumulative_simpson, quad
from scipy.optimize import brentq, minimize_scalar

from model import CAPTURED, CROSSED, UNRESOLVED, Potential, VSG, sweep, waveforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "verification"))
from independent import phasor_power, reference_run


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2) + "\n")


def read_figures(path):
    groups = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            groups.setdefault((row["panel"], row["series"]), []).append((float(row["x"]), float(row["y"])))
    return {key: np.asarray(values) for key, values in groups.items()}


def inclusive_grid(start, stop, step):
    return start + step * np.arange(round((stop - start) / step) + 1)


def increasing_part(x, y):
    stops = np.flatnonzero(np.diff(x) < -1e-10)
    end = stops[0] + 1 if len(stops) else len(x)
    xx, indices = np.unique(x[:end], return_index=True)
    return xx, y[:end][indices]


def phase_rmse(source, modeled_x, modeled_y, bounds):
    source_x, source_y = increasing_part(*source.T)
    model_x, model_y = increasing_part(modeled_x, modeled_y)
    low = max(bounds[0], source_x[0], model_x[0])
    high = min(bounds[1], source_x[-1], model_x[-1])
    if high <= low:
        raise ValueError("The first-swing traces do not share the requested range")
    x = np.linspace(low, high, 201)
    difference = np.interp(x, model_x, model_y) - np.interp(x, source_x, source_y)
    return float(np.sqrt(np.mean(difference**2)))


def sampled_crossing(time, angle, saddle):
    hit = np.flatnonzero(angle >= saddle)
    if not len(hit):
        return None
    i = hit[0]
    return float(np.interp(saddle, angle[i - 1:i + 1], time[i - 1:i + 1]))


def power_angle_evidence(model, paper, setup):
    angle = np.linspace(-.12, 3.7, 801)
    voltage = setup["figure2_grid_voltage_v"]
    records, traces = [], []
    for resistance, reactance in paper["figure2_resistance_reactance_pairs_pu"]:
        branch = replace(model, resistance=resistance * model.impedance_base, reactance=reactance * model.impedance_base)
        stable, saddle = branch.equilibria(voltage)
        peak = minimize_scalar(lambda d: -branch.power(d, voltage)[1], bounds=(0., np.pi), method="bounded")
        records.append({"r_pu": resistance, "x_pu": reactance, "max_power_w": float(-peak.fun),
                        "stable_rad": stable, "saddle_rad": saddle, "equilibrium_span_rad": saddle - stable})
        traces.append(branch.power(angle, voltage)[1])
    return records, angle, np.asarray(traces)


def check_waveforms(model, time, state, initial, stable, saddle, potential, gains_pu, alpha, setup, figures):
    def complex_power(angle, vg):
        return phasor_power(angle, vg, model.v_ref, model.q_ref, model.voltage_droop,
                            complex(model.resistance, model.reactance))

    def active_power(angle):
        return complex_power(angle, model.v_sag)[1]

    probe_angles = np.linspace(-np.pi, np.pi, 81)
    max_power_error = 0.
    for vg in (model.v_before, model.v_sag):
        direct = np.array([complex_power(d, vg) for d in probe_angles])
        max_power_error = max(max_power_error, float(np.max(abs(direct - np.asarray(model.power(probe_angles, vg)).T))))
    pre = brentq(lambda d: complex_power(d, model.v_before)[1] - model.p_ref, 0, 1)
    sep = brentq(lambda d: active_power(d) - model.p_ref, .9, 1.46)
    uep = brentq(lambda d: active_power(d) - model.p_ref, 1.48, 2.2)
    potential_probe = np.linspace(potential.left_saddle, saddle, 61)
    direct_potential = [quad(lambda d: active_power(d) - model.p_ref, stable, p, epsabs=1e-8)[0] for p in potential_probe]
    potential_error = float(np.max(abs(np.interp(potential_probe, potential.angle, potential.energy) - direct_potential)))
    stiffness = (active_power(sep + 1e-5) - active_power(sep - 1e-5)) / 2e-5
    records = []
    for i, gain_pu in enumerate(gains_pu):
        gain = model.gain_unit * gain_pu
        angle, speed, h = state[:, :, i].T
        hpf, x2 = gain * (speed - h), -gain * h
        cross = sampled_crossing(time, angle, saddle)
        ref, ref_outcome, ref_cross = reference_run(active_power, model.p_ref, model.inertia, model.damping,
                                                    alpha, gain, pre, sep, uep, time, time[-1])
        stop = np.searchsorted(time, cross) if cross is not None else len(time)
        power_loss = model.damping * speed[:stop]**2 + gain * (speed[:stop] - h[:stop])**2
        energy = potential.total(model, state[:stop, :, i].T, gain / (2 * alpha))
        energy += cumulative_simpson(power_loss, x=time[:stop], initial=0.)
        poles = np.roots([model.inertia, model.inertia * alpha + model.damping + gain,
                          model.damping * alpha + stiffness, stiffness * alpha])
        key = str(int(gain_pu))
        records.append({
            "gain_pu": gain_pu, "crossed_saddle": cross is not None, "first_crossing_s": cross,
            "one_positive_slip": bool((angle > stable + 2 * np.pi).any()),
            "reference_outcome": ref_outcome, "reference_crossing_s": ref_cross,
            "solver_errors": np.max(abs(np.column_stack((angle, speed, hpf)) - ref), axis=0).tolist(),
            "energy_balance_error_w_rad": float(np.max(abs(energy - energy[0]))),
            "local_rightmost_pole_real_per_s": float(poles.real.max()),
            "first_swing_fig3a_speed_rmse_rad_s": phase_rmse(figures["3a", key], angle, speed, setup["first_swing_angle_band_rad"]),
        })
        if ("3b", key) in figures:
            records[-1]["first_swing_fig3b_x2_rmse_w"] = phase_rmse(figures["3b", key], speed, x2, [.05, 1.0])
    return {"phasor_max_error": max_power_error, "equilibrium_max_error_rad": max(abs(pre - initial), abs(sep - stable), abs(uep - saddle)),
            "potential_interp_error_w_rad": potential_error, "stiffness_w_rad": stiffness, "cases": records}, active_power


def gain_intervals(gains, outcome):
    good = np.flatnonzero(outcome == CAPTURED)
    segments = np.split(good, np.flatnonzero(np.diff(good) > 1) + 1)
    return [[float(gains[s[0]]), float(gains[s[-1]])] for s in segments if len(s)]


def sweep_evidence(model, initial, stable, saddle, potential, active_power, setup, paper, alphas, gains, published):
    step, horizon, guard = (setup[key] for key in ("sweep_step_s", "sweep_horizon_s", "potential_guard_w_rad"))
    grid, stop_time = sweep(model, initial, potential, alphas[:, None], gains[None, :], step, horizon, guard)
    refine_alphas = np.array(setup["refinement_alphas_rad_s"])
    fine_gains = inclusive_grid(gains[0], gains[-1], setup["refinement_gain_step_pu"])
    fine_grid, _ = sweep(model, initial, potential, refine_alphas[:, None], fine_gains[None, :], step / 2, horizon, guard)
    coarse_step_grid, _ = sweep(model, initial, potential, refine_alphas[:, None], fine_gains[None, :], step, horizon, guard)
    slices, checks = {}, []
    for alpha, row in zip(refine_alphas, fine_grid, strict=True):
        intervals = gain_intervals(fine_gains, row)
        key = str(alpha)
        slices[key] = {"capture_intervals_pu": intervals, "n_captured": int((row == CAPTURED).sum()),
                       "n_crossed": int((row == CROSSED).sum()), "n_unresolved": int((row == UNRESOLVED).sum())}
        probes = set(paper["figure4_reported_slices_pu"].get(key, []))
        for low, high in intervals:
            probes.update([max(0., low - .2), low + .2, high - .2, min(gains[-1], high + .2)])
        if not probes:
            probes.add(30.)
        for gain_pu in sorted(probes):
            index = int(round(gain_pu / setup["refinement_gain_step_pu"]))
            expected = int(row[index])
            _, actual, crossing = reference_run(active_power, model.p_ref, model.inertia, model.damping,
                                                alpha, fine_gains[index] * model.gain_unit, initial, stable, saddle, None, horizon)
            checks.append({"alpha_rad_s": float(alpha), "gain_pu": float(fine_gains[index]),
                           "rk4_outcome": expected, "dop853_outcome": actual, "dop853_crossing_s": crossing})
    captured = grid == CAPTURED
    has_capture = alphas[captured.any(axis=1)]
    return grid, stop_time, fine_grid, {
        "captured": int(captured.sum()), "crossed": int((grid == CROSSED).sum()), "unresolved": int((grid == UNRESOLVED).sum()),
        "published_region_points": int(published.sum()), "captured_inside_published_region": int((captured & published).sum()),
        "alpha_max_captured_rad_s": float(has_capture[-1]) if len(has_capture) else None,
        "rk4_step_half_mismatches": int((coarse_step_grid != fine_grid).sum()),
        "slices": slices, "independent_slice_checks": checks,
    }


def numeric_checks(metrics, gates):
    waves, sweeps = metrics["waveforms"], metrics["sweeps"]
    cases = [c for branch in waves.values() for c in branch["cases"]]
    boundary = [p for branch in sweeps.values() for p in branch["independent_slice_checks"]]
    solver_max = np.max([c["solver_errors"] for c in cases], axis=0)
    return {
        "phasor_algebra": max(r["phasor_max_error"] for r in waves.values()) < gates["phasor_max_error_w_var"],
        "equilibria": max(r["equilibrium_max_error_rad"] for r in waves.values()) < gates["equilibrium_max_error_rad"],
        "potential_quadrature": max(r["potential_interp_error_w_rad"] for r in waves.values()) < gates["potential_interp_error_w_rad"],
        "cross_solver_angles": solver_max[0] < gates["cross_solver_angle_max_error_rad"],
        "cross_solver_speeds": solver_max[1] < gates["cross_solver_speed_max_error_rad_s"],
        "cross_solver_hpf_outputs": solver_max[2] < gates["cross_solver_hpf_max_error_w"],
        "cross_solver_crossing_flags": all(c["crossed_saddle"] == (c["reference_outcome"] == CROSSED) for c in cases),
        "cross_solver_crossing_times": all(abs(c["first_crossing_s"] - c["reference_crossing_s"]) < gates["cross_solver_crossing_time_error_s"]
                                           for c in cases if c["crossed_saddle"] and c["reference_outcome"] == CROSSED),
        "energy_identity": max(c["energy_balance_error_w_rad"] for c in cases) < gates["energy_balance_error_w_rad"],
        "half_step_slice_convergence": all(b["rk4_step_half_mismatches"] == 0 for b in sweeps.values()),
        "independent_slice_classification": all(c["rk4_outcome"] == c["dop853_outcome"] for c in boundary),
        "stable_equilibrium_local_poles": all(c["local_rightmost_pole_real_per_s"] < 0 for c in cases),
    }


def compare_paper(metrics, paper, gates):
    phase = {}
    for name, waveform in metrics["waveforms"].items():
        cases = waveform["cases"]
        phase[name] = {
            "crossing_pattern_matches": [c["crossed_saddle"] for c in cases] == paper["figure3_expected_crossing"],
            "first_swing_rmse_within_gate": max(c["first_swing_fig3a_speed_rmse_rad_s"] for c in cases) < gates["fig3_first_swing_max_rmse_rad_s"],
        }
    region = {}
    for name, region_data in metrics["sweeps"].items():
        matches = {}
        for alpha, target in paper["figure4_reported_slices_pu"].items():
            intervals = region_data["slices"][alpha]["capture_intervals_pu"]
            matches[alpha] = len(intervals) == 1 and bool(np.allclose(intervals[0], target, rtol=0,
                                                                     atol=gates["fig4_slice_gain_tolerance_pu"]))
        max_alpha = region_data["alpha_max_captured_rad_s"]
        region[name] = {"slice_matches": matches, "alpha_max_matches": max_alpha is not None
                        and abs(max_alpha - paper["figure4_reported_alpha_max_rad_s"]) <= gates["fig4_max_alpha_tolerance_rad_s"]}
    return {"fig3": phase, "fig4": region}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--figures", type=Path)
    args = parser.parse_args()
    start, clock = datetime.now(timezone.utc), perf_counter()
    out = args.out
    figures_dir = args.figures or out / "figures"
    out.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    config_path = ROOT / "configs/protocol.json"
    source_path = ROOT / "verification/fig3_4_readback.csv"
    cfg = json.loads(config_path.read_text())
    paper, setup, gates = (cfg[k] for k in ("paper", "reconstruction", "gates"))
    figures = read_figures(source_path)
    alphas = inclusive_grid(*setup["sweep_alpha_range_rad_s"])
    gains = inclusive_grid(*setup["sweep_gain_range_pu"])
    from matplotlib.path import Path as Polygon
    xy = np.stack(np.meshgrid(alphas, gains, indexing="ij"), axis=-1)
    fills = [values for (panel, _), values in figures.items() if panel == "4"]
    published = np.logical_or.reduce([Polygon(v).contains_points(xy.reshape(-1, 2)).reshape(xy.shape[:2]) for v in fills])
    phase_gains = np.asarray(paper["figure3_gains_pu"])
    metrics = {"doi": cfg["doi"], "sweep_grid_shape": list(published.shape),
               "published_fill_max_alpha_rad_s": float(max(np.max(v[:, 0]) for v in fills)),
               "waveforms": {}, "sweeps": {}, "parameters": {}}
    traces, maps = {"gain_pu": phase_gains}, {"alpha_rad_s": alphas, "gain_pu": gains, "paper_region": published}
    maps["refine_alphas_rad_s"] = np.asarray(setup["refinement_alphas_rad_s"])
    maps["refine_gains_pu"] = inclusive_grid(gains[0], gains[-1], setup["refinement_gain_step_pu"])
    for name, branch in setup["branches"].items():
        model = VSG.from_protocol(paper, setup, branch)
        initial = model.equilibria(model.v_before)[0]
        stable, saddle = model.equilibria(model.v_sag)
        potential = Potential.build(model, stable, saddle, setup["potential_nodes"])
        metrics["parameters"][name] = {"r_ohm": model.resistance, "x_ohm": model.reactance,
                                        "pref_w": model.p_ref, "gain_unit": model.gain_unit, "J": model.inertia,
                                        "Dp": model.damping, "Dq": model.voltage_droop, "initial_rad": initial,
                                        "stable_rad": stable, "saddle_rad": saddle, "barrier_w_rad": potential.barrier}
        time, state = waveforms(model, initial, paper["figure3_alpha_rad_s"], phase_gains, setup["waveform_step_s"],
                                setup["waveform_duration_s"])
        traces["time_s"], traces[name] = time, state
        metrics["waveforms"][name], active_power = check_waveforms(model, time, state, initial, stable, saddle, potential,
                                                                   phase_gains.tolist(), paper["figure3_alpha_rad_s"], setup, figures)
        maps[name], maps[name + "_stop_s"], maps[name + "_refined"], metrics["sweeps"][name] = sweep_evidence(
            model, initial, stable, saddle, potential, active_power, setup, paper, alphas, gains, published
        )
        print(f"{name}: {metrics['sweeps'][name]['captured']} captures, {metrics['sweeps'][name]['unresolved']} unresolved", flush=True)
    base = VSG.from_protocol(paper, setup, setup["branches"]["table_si"])
    metrics["fig2"], angles, power = power_angle_evidence(base, paper, setup)
    metrics["paper_comparisons"] = compare_paper(metrics, paper, gates)
    checks = numeric_checks(metrics, gates)
    for key in ("max_power_w", "equilibrium_span_rad"):
        checks["fig2_decreasing_" + key] = bool(np.all(np.diff([r[key] for r in metrics["fig2"]]) < 0))
    checks = {key: bool(val) for key, val in checks.items()}
    metrics.update(checks=checks, checks_passed=sum(checks.values()), checks_total=len(checks))
    save_json(out / "metrics.json", metrics)
    np.savez_compressed(out / "phase_traces.npz", **traces)
    np.savez_compressed(out / "stability_maps.npz", **maps)
    np.savez_compressed(out / "power_angle_curves.npz", angle_rad=angles, power_w=power)
    from plots import draw
    plots = draw(figures_dir, metrics, figures, traces, maps, angles, power)
    inputs = ["src/model.py", "src/reproduce.py", "src/plots.py", "verification/independent.py", "verification/fig3_4_readback.csv"]
    outputs = ("metrics.json", "phase_traces.npz", "stability_maps.npz", "power_angle_curves.npz")
    save_json(out / "run_manifest.json", {
        "paper_doi": cfg["doi"], "started_utc": start.isoformat(), "finished_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": perf_counter() - clock, "python": platform.python_version(), "platform": platform.platform(),
        "numpy": np.__version__, "scipy": scipy.__version__, "protocol_sha256": digest(config_path),
        "implementation_sha256": {name: digest(ROOT / name) for name in inputs},
        "data_sha256": {name: digest(out / name) for name in outputs},
        "figures_sha256": {path.name: digest(path) for path in plots},
        "checks_passed": metrics["checks_passed"], "checks_total": metrics["checks_total"],
    })
    print(f"{metrics['checks_passed']}/{metrics['checks_total']} numerical checks passed. {out}", flush=True)
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
