"""Reconstruct VSG validation cases and compare equations, dq states and figures."""

from argparse import ArgumentParser
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
from numpy.polynomial import polynomial as poly
from scipy.optimize import brentq, linear_sum_assignment
from scipy.signal import find_peaks, step

from model import (close_loop, closed_poles, evaluate, inferred_case2_models, margins, nyquist,
                   printed_line, published_model, reduced_crossover, sign_crossings, transfer_functions)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "verification"))
from dq_reference import closed_response, line_response, linear_step, nonlinear_step, state_matrices


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def now():
    return datetime.now(timezone.utc).isoformat()


def read_figures(path):
    grouped = {}
    with path.open() as handle:
        for row in csv.DictReader(handle):
            key = (int(row["figure"]), row["axis"], row["series"])
            grouped.setdefault(key, []).append((float(row["abscissa"]), float(row["value"])))
    return {key: np.array(sorted(points)).T for key, points in grouped.items()}


def complex_rows(poles):
    return [[float(p.real), float(p.imag)] for p in sorted(poles, key=lambda p: (p.real, p.imag))]


def pole_state(poles):
    max_real = max(poles.real)
    if max_real > 1e-7:
        return "unstable"
    if max_real >= -1e-7:
        return "marginal"
    return "asymptotically_stable"


def frequency_checks(model, functions, frequency, gates):
    jw = 1j * frequency
    omega = model.omega
    error = 0.
    for a, b in (
        (evaluate(close_loop(functions["full"]), jw, omega), closed_response(model, jw)),
        (evaluate(functions["line"], jw, omega), line_response(jw, model)),
    ):
        error = max(error, float(np.max(abs(a - b) / np.maximum(1e-9, abs(a)))))
    full_poles = closed_poles(functions["full"], omega)
    reduced_poles = closed_poles(functions["reduced"], omega)
    state_poles = np.linalg.eigvals(state_matrices(model)[0])
    distances = abs(full_poles[:, None] - state_poles)
    left, right = linear_sum_assignment(distances)
    pole_error = float(max(distances[left, right]))
    winding, phase_step = nyquist(functions["full"], frequency, omega)
    rhp = int(np.count_nonzero(full_poles.real > 1e-7))
    return {
        "equation_vs_dq_max_frequency_relative_error": error,
        "equation_vs_dq_max_pole_error_rad_s": pole_error,
        "full_closed_poles_rad_s": complex_rows(full_poles),
        "reduced_closed_poles_rad_s": complex_rows(reduced_poles),
        "full_closed_classification": pole_state(full_poles),
        "reduced_closed_classification": pole_state(reduced_poles),
        "full_right_half_plane_pole_count": rhp,
        "full_dominant_real_pole_rad_s": float(max(full_poles.real)),
        "full_clockwise_rhp_nyquist_winding": winding,
        "nyquist_max_phase_step_rad": phase_step,
        "checks": {
            "independent_dq_frequency": error < gates["same_model_frequency_relative_error"],
            "independent_dq_poles": pole_error < gates["same_model_pole_error_rad_s"],
            "nyquist_matches_full_pole_count": bool(-winding == rhp and phase_step < .2),
        },
    }


def bode_readback(model, functions, figure, source):
    comparison = {}
    for name, pair in functions.items():
        if not np.any(pair[0]):
            continue
        for axis in ("magnitude_db", "phase_deg"):
            w, y = source[figure, axis, name]
            h = evaluate(pair, 1j * w, model.omega)
            calculated = 20 * np.log10(abs(h)) if axis == "magnitude_db" else np.degrees(np.unwrap(np.angle(h)))
            rms = np.sqrt(np.mean((calculated - y)**2))
            comparison[f"{name}_{axis}_rmse"] = float(rms)
    return comparison


def evaluate_case(case_id, model, meta, source, frequency, gates):
    functions = transfer_functions(model)
    record = frequency_checks(model, functions, frequency, gates)
    reduced_margin = margins(functions["reduced"], frequency, model.omega)
    full_margin = margins(functions["full"], frequency, model.omega)
    h0 = float(functions["reduced"][0][0])
    reduced_omega = reduced_crossover(h0, model)
    if reduced_omega is None:
        eq7_check = not reduced_margin["gain_crossings"]
        eq8 = None
    else:
        eq7_check = abs(reduced_omega - reduced_margin["gain_crossings"][0]["omega_rad_s"]) / reduced_omega
        eq7_check = bool(eq7_check <= gates["equation7_crossover_check_fraction"])
        eq8 = [bool(reduced_omega <= .1 * model.omega), bool(reduced_omega <= model.damping / model.inertia)]
    record.update({
        "id": case_id, "source_figure": meta["figure"],
        "parameter_branch": "published_values" if case_id == meta["id"] else "inferred_from_reported_reduced_crossover",
        "virtual_inductance_h": model.virtual_inductance, "virtual_resistance_ohm": model.virtual_resistance,
        "total_series_inductance_h": model.inductance, "total_series_resistance_ohm": model.resistance,
        "current_loop_time_constant_s": model.current_tau,
        "line_dc_power_angle_gain_w_per_rad": h0,
        "damping_to_inertia_rad_s": model.damping / model.inertia,
        "reduced_crossover_eq7_rad_s": reduced_omega, "eq8_conditions_in_order": eq8,
        "paper_metrics": {k: v for k, v in meta.items() if k.startswith("reported_")},
        "full_margins": full_margin, "reduced_margins": reduced_margin,
        "readback": bode_readback(model, functions, meta["figure"], source),
    })
    record["checks"]["equation7_matches_frequency_crossing"] = eq7_check
    return record, functions


def dense_signal(time, values):
    # The vector coordinates are quantized. Average the ordinates of
    # repeated abscissae before making an interpolant.
    x, inv = np.unique(time, return_inverse=True)
    return x, np.bincount(inv, weights=values) / np.bincount(inv)


def source_step_metrics(case_id, data, step_start):
    time, values = dense_signal(*data)
    post = time >= step_start
    basic = {"post_step_peak_mw": float(values[post].max()), "final_trace_mw": float(values[-1]),
             "pre_step_max_abs_mw": float(max(abs(values[~post])))}
    if case_id == "case1":
        return basic
    uniform = np.linspace(0., 1., 4001)
    v = np.interp(uniform, time, values)
    spacing_s = .012 if case_id == "case2" else .12
    indices, _ = find_peaks(v, distance=round(spacing_s / (uniform[1] - uniform[0])), prominence=.05)
    peaks = indices[(uniform[indices] >= .20) & (uniform[indices] <= .94)]
    times = uniform[peaks]
    retained = peaks[uniform[peaks] >= step_start]
    return {**basic, "number_of_peaks_used_for_frequency": len(times),
            "median_peak_spacing_frequency_hz": float(1 / np.median(np.diff(times))),
            "post_step_peaks_mw": [[float(uniform[k]), float(v[k])] for k in retained]}


def waveform_comparison(model, case, functions, time, source, protocol):
    settings = protocol["inferences_and_settings"]
    small_w = settings["small_signal_reference_step_w"]
    reference_w = protocol["published_common"]["fig5_active_power_step_w"]
    rtol, atol = settings["nonlinear_integrator_rtol"], settings["nonlinear_integrator_atol"]
    index = int(np.searchsorted(time, settings["fig5_step_start_s"]))
    step_time = time[index:] - time[index]
    numerator, denominator = close_loop(functions["full"])
    _, canonical = step((poly.polytrim(numerator)[::-1], denominator[::-1]), T=step_time * model.omega)
    physical_linear = linear_step(model, step_time, reference_w)
    nonlinear_w, angle = nonlinear_step(model, step_time, reference_w, rtol, atol)
    small_nonlinear, _ = nonlinear_step(model, step_time, small_w, rtol, atol)
    linear_disagreement = float(np.max(abs(physical_linear / reference_w - canonical)))
    local_agreement = float(np.max(abs(small_nonlinear / small_w - canonical)))
    x, p_source = dense_signal(*source[5, "power_step_mw", case])
    mask = (x >= time[index]) & (x <= time[-1])
    diff = np.interp(x[mask], time[index:], nonlinear_w) / 1e6 - p_source[mask]
    return {
        "canonical_vs_dq_linear_max_step_fraction": linear_disagreement,
        "nonlinear_small_step_vs_linear_max_step_fraction": local_agreement,
        "fig5_post_inferred_step_power_rmse_mw": float(np.sqrt(np.mean(diff**2))),
        "average_model_peak_mw": float(nonlinear_w.max() / 1e6),
        "average_model_final_mw": float(nonlinear_w[-1] / 1e6),
        "max_abs_terminal_phase_rad": float(max(abs(angle))),
    }, {tag: np.pad(values, (index, 0)) for tag, values in
        (("linear_w", physical_linear), ("average_w", nonlinear_w), ("terminal_phase_rad", angle))}


def damping_boundary(model):
    def real_part(damping):
        candidate = replace(model, damping=damping)
        return float(max(closed_poles(transfer_functions(candidate)["full"], model.omega).real))

    critical = brentq(real_part, 0., model.damping, xtol=1e-7)
    at_boundary = replace(model, damping=critical)
    poles = np.linalg.eigvals(state_matrices(at_boundary)[0])
    mode = poles[np.argmax(poles.real)]
    gain = transfer_functions(model)["reduced"][0][0]
    bandwidth = .1 * model.omega
    bandwidth_damping = np.sqrt(max(0., (gain / bandwidth)**2 - (model.inertia * bandwidth)**2))
    slope_damping = np.sqrt(model.inertia * gain / np.sqrt(2))
    return {"critical_full_model_damping": float(critical), "critical_pair_imag_rad_s": float(abs(mode.imag)),
            "independent_dq_real_part_at_boundary_rad_s": float(mode.real),
            "full_model_real_part_at_0p99_boundary_rad_s": real_part(.99 * critical),
            "full_model_real_part_at_1p01_boundary_rad_s": real_part(1.01 * critical),
            "min_damping_for_eq8_frequency_separation": float(bandwidth_damping),
            "min_damping_for_eq8_slope_condition": float(slope_damping)}


def time_constant_sweep(models, frequency, settings):
    tau = np.linspace(*settings["time_constant_scan_s"], settings["time_constant_scan_count"])
    records = []
    for value in tau:
        changes = {case: replace(models[case], current_tau=float(value)) for case in ("case1", "case3")}
        first = changes["case1"]
        converter = transfer_functions(first)["converter"]
        stable = bool(np.max(poly.polyroots(converter[1]).real) < 0.)

        def threshold(w):
            return abs(evaluate(converter, 1j * w, first.omega)) - 1 / np.sqrt(2)

        cutoff = sign_crossings(frequency, threshold(frequency), threshold)[0] / (2 * np.pi)
        records.append({"current_time_constant_s": float(value), "converter_3db_hz": float(cutoff),
                        "converter_stable": stable, "converter_condition_passed": bool(stable and cutoff > 100.),
                        **{case + "_dominant_real_rad_s": float(max(
                            closed_poles(transfer_functions(model)["full"], model.omega).real))
                           for case, model in changes.items()}})
    return records


def csv_table(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--figures", type=Path)
    args = parser.parse_args()
    out = args.out.resolve()
    fig_dir = (args.figures or out / "figures").resolve()
    if (out / "metrics.json").exists() or (out / "run_manifest.json").exists():
        parser.error("Use a new output directory for each run.")
    out.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    t0, started = perf_counter(), now()
    protocol_path = ROOT / "configs/protocol.json"
    readback_path = ROOT / "verification/fig2_to_5_readback.csv"
    protocol = json.loads(protocol_path.read_text())
    common, design, gates = (protocol[k] for k in ("published_common", "inferences_and_settings", "gates"))
    specs = {case["id"]: case for case in protocol["published_cases"]}
    models = {tag: published_model(common, spec) for tag, spec in specs.items()}
    models.update(inferred_case2_models(models["case2"], specs["case2"]["reported_crossover_rad_s"]))
    source = read_figures(readback_path)
    frequency = np.geomspace(*design["frequency_scan_rad_s"], design["frequency_scan_count"])
    time = np.arange(0., design["fig5_comparison_s"] + design["waveform_sample_step_s"] / 2,
                     design["waveform_sample_step_s"])
    rows, responses, waveforms = {}, {}, {}
    for case, model in models.items():
        meta = specs.get(case, specs["case2"])
        row, functions = evaluate_case(case, model, meta, source, frequency, gates)
        responses[case] = {key: evaluate(value, 1j * frequency, model.omega)
                           for key, value in functions.items()}
        if case in ("case1", "case3"):
            metrics, waveforms[case] = waveform_comparison(model, case, functions, time, source, protocol)
            row["step_comparison"] = metrics
            row["checks"]["independent_linear_step"] = (
                metrics["canonical_vs_dq_linear_max_step_fraction"] < gates["same_model_step_error_fraction"])
            row["checks"]["small_step_tends_to_linear_model"] = (
                metrics["nonlinear_small_step_vs_linear_max_step_fraction"] < gates["nonlinear_small_signal_step_error_fraction"])
            row["checks"]["published_bode_full_magnitude_and_phase"] = (
                row["readback"]["full_magnitude_db_rmse"] <= gates["readback_bode_magnitude_rmse_db"]
                and row["readback"]["full_phase_deg_rmse"] <= gates["readback_bode_phase_rmse_deg"])
        rows[case] = row
        print(case, row["full_closed_classification"], row["full_dominant_real_pole_rad_s"], flush=True)
    completed_at = perf_counter()
    hi = rows["case2_inferred_high_x"]
    lo = rows["case2_inferred_low_x"]
    gm = hi["full_margins"]["negative_real_crossings"][0]["gain_margin_db"]
    source_steps = {case: source_step_metrics(case, source[5, "power_step_mw", case], design["fig5_step_start_s"])
                    for case in specs}
    case1 = rows["case1"]
    ref_co = specs["case1"]["reported_crossover_rad_s"]
    tau_sweep = time_constant_sweep(models, frequency[::4], design)
    eligible = [r for r in tau_sweep if r["converter_condition_passed"]]
    boundary = damping_boundary(models["case3"])
    off_nominal_angle = 0.2
    printed = evaluate(printed_line(models["case1"], off_nominal_angle), 0., models["case1"].omega)
    independent = line_response(0., models["case1"], off_nominal_angle)[0]
    off_nominal = {"initial_phase_rad": off_nominal_angle, "printed_eq3_dc_w_per_rad": float(printed),
                   "ideal_series_dq_dc_w_per_rad": float(independent),
                   "relative_difference": float(abs(printed - independent) / abs(independent))}
    global_checks = {
        "case1_reduced_crossover_matches_reported": abs(case1["reduced_crossover_eq7_rad_s"] / ref_co - 1) < gates["readback_case1_crossover_relative_error"],
        "case2_inferred_high_branch_gain_margin_matches": abs(gm - specs["case2"]["reported_gain_margin_db"]) < gates["case2_inferred_gain_margin_abs_error_db"],
        "case2_inferred_high_branch_better_explains_line_plot": (
            hi["readback"]["line_phase_deg_rmse"] < lo["readback"]["line_phase_deg_rmse"]
            and hi["readback"]["line_magnitude_db_rmse"] < lo["readback"]["line_magnitude_db_rmse"]),
        "case2_inferred_high_branch_matches_full_bode": (
            hi["readback"]["full_magnitude_db_rmse"] < gates["readback_bode_magnitude_rmse_db"]
            and hi["readback"]["full_phase_deg_rmse"] < gates["readback_bode_phase_rmse_deg"]),
        "independent_dq_verifies_case3_damping_boundary": (
            abs(boundary["independent_dq_real_part_at_boundary_rad_s"]) < gates["same_model_pole_error_rad_s"]
            and boundary["full_model_real_part_at_0p99_boundary_rad_s"] > 0
            and boundary["full_model_real_part_at_1p01_boundary_rad_s"] < 0),
    }
    passed = [ok for row in rows.values() for ok in row["checks"].values()] + list(global_checks.values())
    metrics = {"doi": protocol["paper_doi"], "grid_omega_rad_s": models["case1"].omega,
               "eq8_tenth_of_grid_omega_rad_s": .1 * models["case1"].omega, "cases": rows,
               "fig5_vector_trace_metrics": source_steps,
               "fig5_inferred_step_start_s": design["fig5_step_start_s"],
               "case3_damping_boundary_at_fixed_other_parameters": boundary,
               "off_nominal_ideal_series_model_diagnostic": off_nominal,
               "time_constant_sweep": {"points": len(tau_sweep), "eligible_points": len(eligible),
                                       "max_case3_real_pole_over_eligible_rad_s": max(r["case3_dominant_real_rad_s"] for r in eligible)},
               "checks_passed": int(sum(passed)), "checks_total": len(passed), "global_checks": global_checks}
    dump(out / "metrics.json", metrics)
    csv_table(out / "time_constant_sweep.csv", tau_sweep)
    np.savez_compressed(out / "frequency_responses.npz", omega_rad_s=frequency,
                        **{f"{case}__{tag}": h for case, data in responses.items() for tag, h in data.items()})
    np.savez_compressed(out / "step_responses.npz", time_s=time,
                        **{f"{case}__{tag}": a for case, data in waveforms.items() for tag, a in data.items()})
    from plots import plot
    figures = plot(fig_dir, frequency, time, rows, responses, waveforms, source, design["fig5_step_start_s"])
    outputs = ["metrics.json", "time_constant_sweep.csv", "frequency_responses.npz", "step_responses.npz"]
    numerical_sources = ["src/model.py", "src/reproduce.py", "src/plots.py", "verification/dq_reference.py",
                         "verification/fig2_to_5_readback.csv"]
    dump(out / "run_manifest.json", {
        "schema_version": 1, "paper_doi": protocol["paper_doi"], "started_utc": started, "finished_utc": now(),
        "wall_seconds": perf_counter() - t0, "main_cases_wall_seconds": completed_at - t0,
        "python": platform.python_version(), "platform": platform.platform(),
        "numpy": np.__version__, "scipy": scipy.__version__, "protocol_sha256": digest(protocol_path),
        "implementation_sha256": {rel: digest(ROOT / rel) for rel in numerical_sources},
        "data_sha256": {name: digest(out / name) for name in outputs},
        "figures_sha256": {path.name: digest(path) for path in figures},
        "checks_passed": metrics["checks_passed"], "checks_total": metrics["checks_total"],
    })
    print(f"{metrics['checks_passed']}/{metrics['checks_total']} checks passed. {out}")


if __name__ == "__main__":
    main()
