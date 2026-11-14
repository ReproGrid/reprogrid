"""Reproduce equilibrium limits, finite disturbances, and four-branch load steps."""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

import numpy as np
import scipy
from scipy.linalg import solve_continuous_lyapunov, solve_sylvester
from scipy.optimize import brentq, minimize_scalar

from model import Plant, integrate, reduced_limit, sampled, voltage_extrema, voltage_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "verification"))
from independent import bus_equilibrium, decode, reference_trace, thresholds


QUADRATURE_NODES, QUADRATURE_WEIGHTS = np.polynomial.legendre.leggauss(8)


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def spectral_abscissa(plant, power):
    equilibrium = plant.equilibrium(power)
    return float(np.linalg.eigvals(plant.jacobian(equilibrium[-1], power)).real.max())


def local_limits(plant, bounds):
    # A Hopf boundary is a sign change on the high-voltage branch before the fold.
    near_fold = plant.fold * (1 - 1e-10)
    limit = plant.fold
    if spectral_abscissa(plant, near_fold) > 0:
        limit = brentq(lambda power: spectral_abscissa(plant, power), 0., near_fold, xtol=1e-9)
    ref, hopf_frequency = thresholds(plant.source, plant.resistance, plant.inductance, plant.capacitance)
    voltage = voltage_limits(plant.source, plant.r_eq, bounds)
    return {
        "r_eq_ohm": plant.r_eq, "l_eq_h": plant.l_eq, "total_capacitance_f": plant.capacitance,
        "saddle_node_w": plant.fold, "local_stability_limit_w": limit,
        "type": "Hopf" if limit < plant.fold else "fold",
        "hopf_angular_frequency_rad_s": hopf_frequency,
        "stability_threshold_cross_check_error_w": abs(limit - ref),
        "equation_48_reduced_w": reduced_limit(plant.source, plant.r_eq, plant.l_eq, plant.capacitance),
        "equilibrium_voltage_positive_load_w": float(voltage[0]),
        "equilibrium_voltage_negative_load_w": float(voltage[1]),
    }


def detail(plant, sol, power, end, setup, stop_pu, knee=None):
    rec = {"load_w": float(power), "start_s": float(sol.t[0]), "end_s": float(sol.t[-1]),
           "low_voltage_stop_pu": stop_pu, "stopped_on_low_voltage": bool(len(sol.t_events[0])),
           "min_voltage_v": None, "max_voltage_v": None,
           "terminal_voltage_v": float(sol.y[-1, -1]), "equilibrium_v": None}
    rec["min_voltage_v"], rec["max_voltage_v"] = voltage_extrema(sol)
    initial = sol.y[:, 0]
    eq = plant.equilibrium(power)
    ref_eq = bus_equilibrium(plant.source, plant.resistance, power)
    rec["equilibrium_reference_error_v"] = 0.
    rec["converged_to_high_voltage_equilibrium"] = False
    if eq is not None:
        if ref_eq is None:
            raise RuntimeError("Independent load-flow solution was missing")
        rec["equilibrium_v"] = float(eq[-1])
        rec["equilibrium_reference_error_v"] = abs(eq[-1] - ref_eq)
        rec["local_rightmost_real_per_s"] = spectral_abscissa(plant, power)
        rec["terminal_relative_state_error"] = float(np.max(abs(sol.y[:, -1] - eq) /
                                                               np.r_[np.full(len(eq) - 1, plant.source / plant.r_eq), plant.source]))
        rec["converged_to_high_voltage_equilibrium"] = rec["terminal_relative_state_error"] < 1e-5
    elif ref_eq is not None:
        raise RuntimeError("Implementations disagree on existence of the high-voltage solution")
    ref_sol, base = reference_trace(plant.source, plant.resistance, plant.inductance, plant.capacitance, power, initial,
                                    (sol.t[0], end), setup["reference_solver"], stop_pu * plant.source, knee)
    t_stop = min(sol.t[-1], ref_sol.t[-1])
    comparison_t = np.unique(np.r_[np.linspace(sol.t[0], t_stop, 1001), sol.t[sol.t <= t_stop]])
    measured = sol.sol(comparison_t).T
    reference = decode(plant.source, base, ref_sol.sol(comparison_t))
    absolute_error = np.max(abs(measured - reference), axis=0)
    ref_stopped = bool(len(ref_sol.t_events[0]))
    rec["reference_voltage_error_v"] = float(absolute_error[-1])
    rec["reference_current_error_a"] = float(max(absolute_error[:-1]))
    rec["reference_stop_agrees"] = rec["stopped_on_low_voltage"] == ref_stopped
    rec["reference_stop_time_error_s"] = float(abs(sol.t[-1] - ref_sol.t[-1])) if ref_stopped else 0.
    if knee is None:
        half_steps = np.diff(sol.t) / 2
        nodes = sol.t[:-1, None] + half_steps[:, None] * (1 + QUADRATURE_NODES)
        # Each DOP853 dense segment is degree seven, so eight-point
        # Gauss–Legendre quadrature integrates the quadratic power exactly.
        power_at_nodes = plant.power_balance(sol.sol(nodes.ravel()).T, power).reshape(nodes.shape)
        net_energy = half_steps @ (power_at_nodes @ QUADRATURE_WEIGHTS)
        rec["energy_balance_error_j"] = float(abs(plant.stored_energy(sol.y[:, -1]) - plant.stored_energy(initial) - net_energy))
    return rec


def trajectory(plant, power, initial, start, end, setup, stop_pu, knee=None):
    sol = integrate(plant, power, initial, (start, end), setup["solver"], plant.source * stop_pu, knee)
    rec = detail(plant, sol, power, end, setup, stop_pu, knee)
    t, states = sampled(sol, setup["waveform_sample_s"])
    return rec, t, states


def cold_step(plant, power, setup, stop_pu):
    initial = np.r_[np.zeros(len(plant.resistance)), plant.source]
    return integrate(plant, power, initial, (0., setup["cold_step_duration_s"]), setup["solver"], plant.source * stop_pu)


def monotone_crossing(predicate, low, high, tolerance):
    if predicate(low) or not predicate(high):
        raise RuntimeError("Boundary endpoints must bracket a change from false to true")
    while high - low > tolerance:
        middle = (high + low) / 2
        if predicate(middle):
            high = middle
        else:
            low = middle
    return [low, high]


def step_limits(plant, bounds, setup, shutdown_limits, local):
    low_v, high_v = np.asarray(bounds) * plant.source
    step_horizon = setup["cold_step_duration_s"]
    stop = setup["stop_voltage_pu"]
    tolerance = setup["dynamics_boundary_tolerance_w"]
    upper_p = local["equilibrium_voltage_positive_load_w"]
    upper_injection = -local["equilibrium_voltage_negative_load_w"]
    voltage_boundaries = {
        "positive_load_w": monotone_crossing(
            lambda power: voltage_extrema(cold_step(plant, power, setup, stop))[0] < low_v,
            0., 1.01 * upper_p, tolerance),
        "negative_load_magnitude_w": monotone_crossing(
            lambda injection: voltage_extrema(cold_step(plant, -injection, setup, stop))[1] > high_v,
            0., 1.01 * upper_injection, tolerance),
    }
    # Check the scan before bracketing. The result is a finite-horizon crossing
    # threshold for this initial state and tested power interval.
    load_grid = np.linspace(0., 1.01 * plant.fold, 37)
    collapse_grid = [bool(len(cold_step(plant, power, setup, stop).t_events[0])) for power in load_grid]
    if any(first and not second for first, second in zip(collapse_grid, collapse_grid[1:])):
        raise RuntimeError("The coarse scan is not monotone; report disconnected crossing intervals")
    brackets = {
        str(fraction): monotone_crossing(
            lambda power: bool(len(cold_step(plant, power, setup, fraction).t_events[0])),
            0., 1.01 * plant.fold, tolerance)
        for fraction in shutdown_limits
    }
    initial = np.r_[np.zeros(len(plant.resistance)), plant.source]
    reference_checks = {}
    for name, interval, fraction in (
        ("positive_voltage", voltage_boundaries["positive_load_w"], stop),
        ("negative_voltage", voltage_boundaries["negative_load_magnitude_w"], stop),
        *((f"collapse_{frac}", brackets[str(frac)], frac) for frac in shutdown_limits),
    ):
        passed = []
        for power in interval:
            signed_power = -power if name == "negative_voltage" else power
            result, current_base = reference_trace(
                plant.source, plant.resistance, plant.inductance, plant.capacitance, signed_power,
                initial, (0., step_horizon), setup["reference_solver"], fraction * plant.source
            )
            if name.startswith("collapse"):
                crossed = bool(len(result.t_events[0]))
            else:
                def voltage_at(time):
                    return decode(plant.source, current_base, result.sol(np.atleast_1d(time)))[0, -1]

                # Refine the extremum around the independent solver's best node.
                sign = 1 if name == "positive_voltage" else -1
                node = int(np.argmin(sign * result.y[-1]))
                lo = result.t[max(0, node - 1)]
                hi = result.t[min(node + 1, len(result.t) - 1)]
                extrema = minimize_scalar(lambda time: sign * voltage_at(time), bounds=(lo, hi),
                                          method="bounded", options={"xatol": 1e-12})
                value = sign * min(extrema.fun, sign * voltage_at(lo), sign * voltage_at(hi))
                crossed = value < low_v if name == "positive_voltage" else value > high_v
            passed.append(bool(crossed))
        reference_checks[name] = passed == [False, True]
    return {"voltage_band_limits": voltage_boundaries, "low_voltage_crossing_limits_w": brackets,
            "scan_load_w": load_grid.tolist(), "scan_events": collapse_grid, "horizon_s": step_horizon,
            "reference_bracket_endpoints_agree": reference_checks}


def phase_portraits(paper, setup, bounds):
    u = paper["source_voltage_v"]
    r, l = paper["droop_resistance_ohm"], paper["inductance_h"]
    data, arrays = {}, {}
    for tag, case in paper["cases"].items():
        c = case["capacitance_f"]
        plant = Plant(u, np.array([r]), np.array([l]), c)
        limits = local_limits(plant, bounds)
        r_intersection = np.sqrt(l / c * (1 - bounds[0]) / bounds[0])
        limits["intersection_resistance_ohm"] = r_intersection
        limits["intersection_residual_w"] = abs(reduced_limit(u, r_intersection, l, c) -
                                                voltage_limits(u, r_intersection, bounds)[0])
        loads = case["phase_portrait_loads_w"]
        initial = np.array([0., u])
        tests = []
        for power in loads:
            rec, t, states = trajectory(plant, power, initial, 0., setup["phase_portrait_duration_s"],
                                         setup, setup["stop_voltage_pu"])
            tests.append(rec)
            arrays[f"{tag}_{power}"] = np.column_stack((t, states))
        p = setup["within_both_static_limits_example_w"]
        rec, t, state = trajectory(plant, p, initial, 0., setup["cold_step_duration_s"], setup, setup["stop_voltage_pu"])
        arrays[f"{tag}_within_static_limits"] = np.column_stack((t, state))
        preload = min(p for p in loads if p > 0)
        warm, t, state = trajectory(plant, p, plant.equilibrium(preload), 0.,
                                    setup["cold_step_duration_s"], setup, setup["stop_voltage_pu"])
        arrays[f"{tag}_preloaded"] = np.column_stack((t, state))
        data[tag] = {"limits": limits, "fig7_8": tests, "within_static_limits": rec,
                     "preload_sensitivity": {"previous_load_w": preload, **warm},
                     "independent_step_sensitivity": step_limits(plant, bounds, setup, setup["cold_step_voltage_guard_stop_pu"], limits)}
    return data, arrays


def schedule(plant, case, setup, knee=None):
    time, loads = [0., *case["change_times_s"], case["end_s"]], [0., *case["load_levels_w"]]
    state = np.r_[np.zeros(len(plant.resistance)), plant.source]
    outputs, stages = [], []
    for start, end, power in zip(time[:-1], time[1:], loads, strict=True):
        rec, t, trace = trajectory(plant, power, state, start, end, setup, setup["stop_voltage_pu"], knee)
        stages.append(rec)
        outputs.append(np.column_stack((t, np.full(len(t), power), trace)))
        if rec["stopped_on_low_voltage"]:
            break
        state = trace[-1]
    return stages, np.concatenate(outputs)


def four_converter(paper, setup, bounds):
    u = paper["source_voltage_v"]
    r, l = np.asarray(paper["resistances_ohm"]), np.asarray(paper["inductances_h"])
    cases, arrays = {}, {}
    for label, spec in paper["cases"].items():
        c = sum(spec["branch_capacitances_f"])
        full = Plant(u, r, l, c)
        equivalent = full.parallel_equivalent()
        rounded = Plant(u, np.array([paper["printed_equivalent_resistance_ohm"]]),
                        np.array([paper["printed_equivalent_inductance_h"]]), c)
        data = {}
        for branch, plant in (("full", full), ("parallel_equivalent", equivalent), ("printed_equivalent", rounded)):
            limits = local_limits(plant, bounds)
            phases, arrays[f"{label}_{branch}"] = schedule(plant, spec, setup)
            data[branch] = {"limits": limits, "stages": phases}
        phases, arrays[f"{label}_hypothetical_current_limited_load"] = schedule(full, spec, setup, knee=bounds[0] * u)
        data["hypothetical_current_limited_load"] = {"stages": phases}
        cases[label] = data
    return cases, arrays


def lyapunov_probe(plant, settings, setup):
    power = settings["active_power_w"]
    state = np.array([settings["initial_current_a"], settings["initial_bus_voltage_v"]])
    v = state[-1]

    def matrix(voltage):
        j = plant.jacobian(voltage, power)
        h_printed = (np.eye(2) + np.linalg.det(j) * np.linalg.inv(j @ j.T)) / (-2 * np.trace(j))
        return j, h_printed

    j, h = matrix(v)
    jv = np.zeros_like(j)
    jv[-1, -1] = -2 * power / (plant.capacitance * v**3)
    hv = solve_sylvester(j.T, j, -(jv.T @ h + h @ jv))
    step = settings["voltage_gradient_step_v"]
    hv_difference = (matrix(v + step)[1] - matrix(v - step)[1]) / (2 * step)
    phi = plant.rhs(state, power)
    rhs_paper = -float(phi @ phi)
    extra = float(phi @ hv @ phi * phi[-1])
    derivative = rhs_paper + extra
    numerical = rhs_paper + float(phi @ hv_difference @ phi * phi[-1])
    short = integrate(plant, power, state, (0., settings["short_trace_duration_s"]), setup["solver"],
                      plant.source * setup["stop_voltage_pu"])
    values = []
    for row in short.y.T:
        d = plant.rhs(row, power)
        values.append(float(d @ matrix(row[-1])[1] @ d))
    return {
        "state": state.tolist(), "active_power_w": power,
        "jacobian_rightmost_real_per_s": float(np.linalg.eigvals(j).real.max()),
        "lyapunov_matrix_min_eigenvalue": float(np.linalg.eigvalsh(h).min()),
        "printed_matrix_vs_scipy_solver_max_error": float(np.max(abs(h - solve_continuous_lyapunov(j.T, -np.eye(2))))),
        "equation_20_fixed_matrix_residual": float(np.max(abs(j.T @ h + h @ j + np.eye(2)))),
        "equation_19_derivative": rhs_paper, "extra_chain_rule_term": extra, "actual_gradient_derivative": derivative,
        "finite_difference_derivative": numerical, "value_start": values[0], "value_end": values[-1],
        "trace_duration_s": settings["short_trace_duration_s"]
    }


def review_counterexample(paper, setup):
    cfg = setup["counterexample"]
    u, r, l = [paper[name] for name in ("source_voltage_v", "droop_resistance_ohm", "inductance_h")]
    c = paper["cases"][cfg["case"]]["capacitance_f"]
    model = Plant(u, np.array([r]), np.array([l]), c)
    probe = setup["lyapunov_derivative_probe"]
    if probe["case"] != cfg["case"]:
        raise ValueError("The Lyapunov probe and large load step must name the same configured plant")
    initial = np.array([0., u])
    tests, arrays = [], {}
    for fraction in cfg["retest_stop_voltage_pu"]:
        rec, t, x = trajectory(model, cfg["active_power_w"], initial, 0., setup["cold_step_duration_s"], setup, fraction)
        tests.append(rec)
        arrays[str(fraction)] = np.column_stack((t, x))
    p = cfg["active_power_w"]
    unstable_v = (u - np.sqrt(u**2 - 4 * r * p)) / 2
    stable_v = model.equilibrium(p)[-1]
    # Finite difference in physically scaled coordinates around the equilibrium.
    scale = np.array([u / r, u])
    eq = model.equilibrium(p)
    derivatives = []
    for unit in np.eye(len(eq)):
        delta = 1e-5 * scale * unit
        derivatives.append((model.rhs(eq + delta, p) - model.rhs(eq - delta, p)) / (2e-5 * scale))
    fd = np.column_stack(derivatives)
    anal = model.jacobian(stable_v, p) * scale / scale[:, None]
    return {"finite_step": tests, "high_equilibrium_v": float(stable_v), "saddle_voltage_v": float(unstable_v),
            "current_upper_bound_a": u / r, "voltage_below_which_zero_is_unavoidable_v": r * p / u,
            "saddle_max_eigenvalue_per_s": spectral_abscissa_at(model, unstable_v, p),
            "linearization_scaled_max_error": float(np.max(abs(fd - anal) / np.maximum(1, abs(anal)))),
            "lyapunov_probe": lyapunov_probe(model, probe, setup)}, arrays


def spectral_abscissa_at(model, voltage, power):
    return float(np.linalg.eigvals(model.jacobian(voltage, power)).real.max())


def check_results(data, paper, gates):
    equivalent, full = data["equivalent"], data["four_converter"]
    probes = data["counterexample"]
    all_records = [*probes["finite_step"]]
    limits = []
    for case in equivalent.values():
        all_records.extend([*case["fig7_8"], case["within_static_limits"], case["preload_sensitivity"]])
        limits.append(case["limits"])
    for case in full.values():
        for branch in case.values():
            all_records.extend(branch["stages"])
            if "limits" in branch:
                limits.append(branch["limits"])
    probe = probes["lyapunov_probe"]
    by_table = paper["equivalent_simulation"]["cases"]
    by_lab = paper["four_converter_system"]["cases"]
    source_voltage_limits = paper["equivalent_simulation"]["paper_lower_and_upper_voltage_limits_w"]
    source_comparison = {
        "table_iii_stability_limit": all(abs(case["limits"]["local_stability_limit_w"] - by_table[key]["paper_stability_limit_w"]) <
                                         gates["table_iii_stability_error_w"] for key, case in equivalent.items()),
        "table_iii_intersections": all(abs(case["limits"]["intersection_resistance_ohm"] - by_table[key]["paper_intersection_ohm"]) <
                                        gates["table_iii_intersection_tolerance_ohm"] for key, case in equivalent.items()),
        "table_iii_voltage_limits_within_display_precision": all(
            abs(case["limits"]["equilibrium_voltage_positive_load_w"] - source_voltage_limits[0]) <= gates["table_iii_voltage_error_w"]
            and abs(case["limits"]["equilibrium_voltage_negative_load_w"] - source_voltage_limits[1]) <= gates["table_iii_voltage_error_w"]
            for case in equivalent.values()
        ),
        "table_v_stability_limit_using_component_values": all(abs(case["full"]["limits"]["local_stability_limit_w"] -
                                                                  by_lab[key]["paper_stability_limit_w"]) <
                                                               gates["table_v_actual_components_limit_error_w"] for key, case in full.items()),
        "table_v_voltage_limit_using_component_values": all(abs(case["full"]["limits"]["equilibrium_voltage_positive_load_w"] -
                                                               by_lab[key]["paper_voltage_limit_w"]) < 1e-8 for key, case in full.items()),
    }
    numeric = {
        "independent_admittance_thresholds": max(p["stability_threshold_cross_check_error_w"] for p in limits) <
                                             gates["independent_stability_boundary_error_w"],
        "equation_48_and_52_intersection": max(e["limits"]["intersection_residual_w"] for e in equivalent.values()) < 1e-7,
        "equation_48_limit": all(abs(e["limits"]["equation_48_reduced_w"] - e["limits"]["local_stability_limit_w"]) < 1e-7
                                 for e in equivalent.values()),
        "independent_step_limit_brackets": all(all(e["independent_step_sensitivity"]["reference_bracket_endpoints_agree"].values())
                                                for e in equivalent.values()),
        "equilibrium_voltages": max(p["equilibrium_reference_error_v"] for p in all_records) < gates["independent_equilibrium_error_v"],
        "scaled_linearization": probes["linearization_scaled_max_error"] < gates["linearization_max_scaled_error"],
        "energy_balance_cpl": max(p.get("energy_balance_error_j", 0.) for p in all_records) < gates["energy_balance_error_j"],
        "independent_bus_voltage": max(p["reference_voltage_error_v"] for p in all_records) < gates["independent_bus_waveform_error_v"],
        "independent_branch_currents": max(p["reference_current_error_a"] for p in all_records) < gates["independent_current_waveform_error_a"],
        "independent_voltage_guard_crossings": all(p["reference_stop_agrees"] and p["reference_stop_time_error_s"] <
                                                   gates["independent_crossing_time_error_s"] for p in all_records),
        "local_but_not_global_counterexample": all(p["local_rightmost_real_per_s"] < 0 and p["stopped_on_low_voltage"]
                                                   and p["terminal_voltage_v"] < probes["voltage_below_which_zero_is_unavoidable_v"]
                                                   for p in probes["finite_step"]),
        "lyapunov_derivative_audit": probe["jacobian_rightmost_real_per_s"] < 0 < probe["actual_gradient_derivative"]
        and abs(probe["finite_difference_derivative"] / probe["actual_gradient_derivative"] - 1) <
        gates["counterexample_derivative_check_relative_error"] and probe["value_end"] > probe["value_start"],
    }
    return {k: bool(v) for k, v in numeric.items()}, {k: bool(v) for k, v in source_comparison.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--figures", type=Path)
    args = parser.parse_args()
    out, figures = args.out, args.figures or args.out / "figures"
    out.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    protocol = ROOT / "configs/protocol.json"
    cfg = json.loads(protocol.read_text())
    paper, setup, gates = (cfg[k] for k in ("paper", "reconstruction", "gates"))
    bounds = paper["voltage_bounds_pu"]
    started, clock = datetime.now(timezone.utc), perf_counter()
    data, phase_arrays = phase_portraits(paper["equivalent_simulation"], setup, bounds)
    four, step_arrays = four_converter(paper["four_converter_system"], setup, bounds)
    counterexamples, counter_arrays = review_counterexample(paper["equivalent_simulation"], setup)
    metrics = {"doi": cfg["doi"], "equivalent": data, "four_converter": four, "counterexample": counterexamples}
    checks, comparison = check_results(metrics, paper, gates)
    metrics.update(checks=checks, checks_passed=sum(checks.values()), checks_total=len(checks),
                   paper_comparisons=comparison)
    write_json(out / "metrics.json", metrics)
    np.savez_compressed(out / "phase_portraits.npz", **phase_arrays)
    np.savez_compressed(out / "four_converter_steps.npz", **step_arrays)
    np.savez_compressed(out / "large_step_probe.npz", **counter_arrays)
    from plots import draw
    paths = draw(figures, cfg, metrics, phase_arrays, step_arrays, counter_arrays)
    inputs = ("src/model.py", "src/reproduce.py", "src/plots.py", "verification/independent.py")
    outputs = ("metrics.json", "phase_portraits.npz", "four_converter_steps.npz", "large_step_probe.npz")
    write_json(out / "run_manifest.json", {
        "schema_version": 1, "paper_doi": cfg["doi"], "started_utc": started.isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(), "wall_seconds": perf_counter() - clock,
        "python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__, "scipy": scipy.__version__,
        "protocol_sha256": digest(protocol), "implementation_sha256": {name: digest(ROOT / name) for name in inputs},
        "data_sha256": {name: digest(out / name) for name in outputs}, "figures_sha256": {path.name: digest(path) for path in paths},
        "checks_passed": metrics["checks_passed"], "checks_total": metrics["checks_total"],
    })
    print(f"{metrics['checks_passed']}/{metrics['checks_total']} numerical checks. "
          f"{sum(comparison.values())}/{len(comparison)} source comparisons.", flush=True)
    if not all(checks.values()):
        print(f"Failed: {[key for key, passed in checks.items() if not passed]}", flush=True)
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
