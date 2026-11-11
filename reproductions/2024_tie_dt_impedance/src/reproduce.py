"""Reproduce the TIE identification equations on a balanced three-phase benchmark."""

from argparse import ArgumentParser
import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

import numpy as np
import scipy

from circuit import (J, continuous_closed_matrix, continuous_impedance, make_circuit,
                     periodic_records, sampled_circuit, sampled_frequency_response)
from identify import (band_metrics, complex_average, error_statistics, exact_common_bin_impedance,
                      from_records, loop_eigenvalues, median_magnitude_phase, perturbations,
                      spectral_leakage)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "verification"))
from sideband import balanced_abc_round_trip, sideband_reference


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def utc():
    return datetime.now(timezone.utc).isoformat()


def file_hash(path):
    return sha256(path.read_bytes()).hexdigest()


def gnc_reference(circuit, frequency, gates):
    z_source, y_grid = continuous_impedance(circuit, frequency)
    loop = z_source @ y_grid
    det = np.linalg.det(np.eye(2) + loop)

    # Along the imaginary axis from -infinity to +infinity, the RHP
    # contour is clockwise. The analytic tail has a finite positive limit.
    tail = (1 + circuit.grid_c / circuit.filter_c) ** 2
    contour = np.r_[tail, det[:0:-1].conj(), det, tail]
    phase_steps = np.angle(contour[1:] / contour[:-1])
    winding = int(np.rint(phase_steps.sum() / (2 * np.pi)))
    poles_twin = np.linalg.eigvals(continuous_closed_matrix(circuit, True))
    poles_grid = -circuit.grid_r / circuit.grid_l + np.linalg.eigvals(-circuit.omega * J)
    poles_open = np.r_[poles_twin, poles_grid]
    poles_closed = np.linalg.eigvals(continuous_closed_matrix(circuit, False))
    open_rhp = int((poles_open.real > 0).sum())
    closed_rhp = int((poles_closed.real > 0).sum())
    if np.min(abs(np.r_[poles_open.real, poles_closed.real])) < 1e-8:
        raise ValueError("An imaginary-axis pole requires an indented contour.")

    s = 2j * np.pi * frequency[:, None]
    det_from_poles = tail * np.prod(s - poles_closed, axis=1) / np.prod(s - poles_open, axis=1)
    determinant_error = float(max(abs(det - det_from_poles) / np.maximum(abs(det_from_poles), 1e-12)))
    z_ref, y_ref = sideband_reference(circuit, frequency)
    _, ez = error_statistics(z_source[1:], z_ref[1:])
    _, ey = error_statistics(y_grid, y_ref)
    sideband_error = max(ez["max"], ey["max"])
    return {
        "winding_ccw_clockwise_rhp_contour": winding,
        "open_loop_rhp_poles": open_rhp,
        "closed_loop_rhp_poles_gnc": open_rhp - winding,
        "closed_loop_rhp_poles_direct": closed_rhp,
        "closed_loop_max_real_pole_per_s": float(max(poles_closed.real)),
        "closed_loop_poles_per_s": [[float(p.real), float(p.imag)] for p in poles_closed],
        "max_contour_phase_step_rad": float(max(abs(phase_steps))),
        "endpoint_relative_error_to_infinity": float(abs(det[-1] - tail) / tail),
        "determinant_vs_pole_product_max_relative": determinant_error,
        "matrix_vs_scalar_sidebands_max_relative": sideband_error,
        "reference_checks": {
            "gnc_equals_direct_pole_count": open_rhp - winding == closed_rhp,
            "full_contour_resolves_argument": bool(max(abs(phase_steps)) < gates["full_contour_max_phase_step_rad"]),
            "determinant_matches_pole_product": determinant_error <= gates["determinant_vs_poles_max_relative"],
            "independent_scalar_sidebands": sideband_error <= gates["matrix_response_from_abc_max_relative"],
        },
    }, loop, det


def acquire(physical, twin, physical_drive, virtual_drive):
    measured, closure_p = periodic_records(physical, physical_drive)
    voltage, current_out = measured[:, :2], measured[:, 2:]
    # The measured PCC current is an input to the synchronized twin. A
    # separate artificial current is injected only into the virtual model.
    virtual_current_in = -current_out + virtual_drive
    virtual_voltage, closure_t = periodic_records(twin, virtual_current_in)
    return voltage, current_out, virtual_voltage, virtual_current_in, max(closure_p, closure_t)


def equation9_reference(physical, twin, drive, bins, physical_scale, virtual_scale):
    data = []
    for axis in range(2):
        test = drive[:, :1] * np.eye(2)[axis]
        voltage, current, twin_voltage, twin_current, closure = acquire(
            physical, twin, physical_scale * test, virtual_scale * test)
        spectra = [np.fft.rfft(a, axis=0)[bins] for a in (voltage, current, twin_voltage, twin_current)]
        data.append((*spectra, closure))
    v, i, vt, it = (np.stack((data[0][k], data[1][k]), axis=-1) for k in range(4))
    return (exact_common_bin_impedance(i, v), exact_common_bin_impedance(it, vt),
            max(data[0][-1], data[1][-1]), float(max(np.linalg.cond(i).max(), np.linalg.cond(it).max())))


def evaluate_case(case, protocol, excitation, bins, f_hz, full_frequency):
    journal, thesis = protocol["journal"], protocol["thesis"]
    model, gates = protocol["independent_benchmark"], protocol["gates"]
    ts, plant_step = journal["acquisition_step_s"], journal["average_dt_step_s"]
    circuit = make_circuit(protocol, case)
    gnc, loop_full, determinant = gnc_reference(circuit, full_frequency, gates)
    physical = sampled_circuit(circuit, False, ts, plant_step)
    twin = sampled_circuit(circuit, True, ts, plant_step)
    radius = float(max(abs(np.linalg.eigvals(physical[0]))))
    radius_twin = float(max(abs(np.linalg.eigvals(twin[0]))))
    if max(radius, radius_twin) >= 1.:
        raise ValueError(f"{case['id']}: the sampled benchmark cannot yield a stable identification record.")

    nominal_v = model["line_to_line_rms_v"] * np.sqrt(2 / 3)
    nominal_i = np.sqrt(2 / 3) * thesis["base_power_w"] / model["line_to_line_rms_v"]
    trial_physical_scale = thesis["real_disturbance_fraction"] * nominal_v
    virtual_scale = thesis["virtual_disturbance_fraction"] * nominal_i
    trial, _ = periodic_records(physical, trial_physical_scale * excitation)
    trial_voltage = float(np.linalg.norm(trial[:, :2], axis=1).max() / nominal_v)
    trial_current = float(np.linalg.norm(trial[:, 2:], axis=1).max() / nominal_i)
    response_limit = journal["reported_max_perturbation_response_fraction"]
    physical_scale = trial_physical_scale * min(
        1., response_limit * model["physical_perturbation_limit_headroom"] / max(trial_voltage, trial_current))
    v, i, vt, it, closure = acquire(physical, twin, physical_scale * excitation, virtual_scale * excitation)
    grid, cond_grid = from_records(i, v, bins, protocol)
    source, cond_source = from_records(it, vt, bins, protocol)

    grid_eq9, source_eq9, closure_ref, cond_eq9 = equation9_reference(
        physical, twin, excitation, bins, physical_scale, virtual_scale)
    transfer_physical = sampled_frequency_response(physical, f_hz, ts)
    source_exact = sampled_frequency_response(twin, f_hz, ts)
    grid_exact = exact_common_bin_impedance(transfer_physical[:, 2:], transfer_physical[:, :2])
    _, check_grid = error_statistics(grid_eq9, grid_exact)
    _, check_twin = error_statistics(source_eq9, source_exact)
    eq9_error = max(check_grid["max"], check_twin["max"])
    err_grid, stats_grid = error_statistics(grid, grid_eq9)
    err_source, stats_source = error_statistics(source, source_eq9)

    z_cont, y_cont = continuous_impedance(circuit, f_hz)
    loop_continuous = z_cont @ y_cont
    loop_estimate = np.full_like(source, np.nan + 1j * np.nan)
    good = np.isfinite(source).all(axis=(1, 2)) & np.isfinite(grid).all(axis=(1, 2))
    loop_estimate[good] = source[good] @ np.linalg.inv(grid[good])
    loop_eq9 = source_eq9 @ np.linalg.inv(grid_eq9)
    loop_error, loop_stats = error_statistics(loop_estimate, loop_eq9)
    _, sample_model_stats = error_statistics(loop_eq9, loop_continuous)

    # The estimator uses dq; retain a real abc record with zero sequence
    # suppressed and verify the inverse transform and phase sum separately.
    t = np.arange(len(v)) * ts
    fundamental = thesis["fundamental_hz"]
    v_abc, v_round_trip, v_balance = balanced_abc_round_trip(v, t, fundamental)
    i_abc, i_round_trip, i_balance = balanced_abc_round_trip(i, t, fundamental)
    abc_error = max(v_round_trip, v_balance, i_round_trip, i_balance)
    peak_voltage = float(np.linalg.norm(v, axis=1).max() / nominal_v)
    peak_current = float(np.linalg.norm(i, axis=1).max() / nominal_i)
    row = {
        **case, "grid_r_ohm": circuit.grid_r, "grid_l_h": circuit.grid_l,
        "sampled_physical_spectral_radius": radius, "sampled_twin_spectral_radius": radius_twin,
        "grid_equation10_vs_equation9": stats_grid, "source_equation10_vs_equation9": stats_source,
        "loop_equation10_vs_equation9": loop_stats,
        "sampled_equation9_vs_continuous_loop": sample_model_stats,
        "equation9_vs_sampled_transfer_max_relative": eq9_error,
        "equation9_current_matrix_max_condition": cond_eq9,
        "equation10_max_row_condition": {"grid": float(np.max(cond_grid)), "source": float(np.max(cond_source))},
        "periodic_state_relative_closure": max(closure, closure_ref),
        "abc_balance_and_round_trip_max_relative": abc_error,
        "physical_drive_amplitude_v": physical_scale,
        "physical_drive_trial_amplitude_v": trial_physical_scale,
        "virtual_current_drive_amplitude_a": virtual_scale,
        "physical_trial_max_dq_perturbation_fraction": {"voltage": trial_voltage, "current": trial_current},
        "physical_max_dq_perturbation_fraction": {"voltage": peak_voltage, "current": peak_current},
        "continuous_full_contour": gnc,
        "continuous_band": band_metrics(loop_continuous, f_hz),
        "sampled_equation9_band": band_metrics(loop_eq9, f_hz),
        "identified_band": band_metrics(loop_estimate, f_hz),
        "checks": {
            "sampled_reference_is_stable": max(radius, radius_twin) < 1.,
            "coherent_periodic_closure": max(closure, closure_ref) <= gates["periodic_state_relative_error"],
            "abc_balanced_and_invertible": abc_error <= gates["matrix_response_from_abc_max_relative"],
            "physical_record_within_five_percent": max(peak_voltage, peak_current) <= response_limit,
            "equation9_matches_sampled_transfer": eq9_error <= gates["direct_equation9_vs_sampled_transfer_max_relative"],
            "all_impedance_bins_usable": bool(good.all()),
            "equation10_median_error_within_gate": max(stats_grid["median"], stats_source["median"]) <= gates["equation10_median_relative"],
            "equation10_p95_error_within_gate": max(stats_grid["p95"], stats_source["p95"]) <= gates["equation10_p95_relative"],
            **{k: bool(v) for k, v in gnc["reference_checks"].items()},
        },
    }
    curves = {
        "grid_eq10": grid, "source_eq10": source, "grid_eq9": grid_eq9, "source_eq9": source_eq9,
        "grid_continuous": np.linalg.inv(y_cont), "source_continuous": z_cont,
        "loop_eq10": loop_estimate, "loop_eq9": loop_eq9, "loop_continuous": loop_continuous,
        "grid_relative_error": err_grid, "source_relative_error": err_source, "loop_relative_error": loop_error,
        "grid_ratio_condition": cond_grid, "source_ratio_condition": cond_source,
        "continuous_full_eigenvalues": loop_eigenvalues(loop_full), "continuous_full_determinant": determinant,
    }
    waveforms = {
        "t_s": t, "physical_modulation_dq_v": physical_scale * excitation, "virtual_injection_dq_a": virtual_scale * excitation,
        "pcc_voltage_dq_v": v, "pcc_outgoing_current_dq_a": i,
        "pcc_voltage_abc_v": v_abc, "pcc_outgoing_current_abc_a": i_abc,
        "twin_voltage_dq_v": vt, "twin_total_incoming_current_dq_a": it,
    }
    return row, curves, waveforms


def noise_experiment(waveforms, curves, bins, protocol):
    """Replay independent noisy acquisition frames of one stationary case."""
    design, journal = protocol["independent_benchmark"], protocol["journal"]
    rng = np.random.default_rng(design["noise_seed"])
    frames = design["noise_average_complete_irs_frames"]
    measurements = {}
    for label, current, voltage in (
        ("grid", "pcc_outgoing_current_dq_a", "pcc_voltage_dq_v"),
        ("source", "twin_total_incoming_current_dq_a", "twin_voltage_dq_v"),
    ):
        i, v = waveforms[current], waveforms[voltage]
        estimates = np.array([
            from_records(i + rng.normal(0., design["noise_current_rms_a"], i.shape),
                         v + rng.normal(0., design["noise_voltage_rms_v"], v.shape), bins, protocol)[0]
            for _ in range(frames)
        ])
        reference = curves[f"{label}_eq9"]
        avg = complex_average(estimates)
        filt = median_magnitude_phase(avg, journal["median_width_bins"])
        single_p95 = [error_statistics(z, reference)[1]["p95"] for z in estimates]
        measurements[label] = {
            "median_single_frame_p95": float(np.median(single_p95)),
            "first_frame": error_statistics(estimates[0], reference)[1],
            "geometric_phase_average": error_statistics(avg, reference)[1],
            "geometric_phase_average_plus_median": error_statistics(filt, reference)[1],
        }
    return {"case": design["noise_case"], "full_common_frames": frames,
            "voltage_noise_rms_v": design["noise_voltage_rms_v"], "current_noise_rms_a": design["noise_current_rms_a"],
            "seed": design["noise_seed"], **measurements}


def zero_coupling_example(excitation, bins, protocol):
    # Fully identifiable with two same-frequency columns, but Eq. (10)
    # requires both excitation responses to have nonzero row voltage.
    impedance = np.diag([2., 3.])
    estimate, _ = from_records(excitation, excitation @ impedance.T, bins, protocol)
    response_i = np.broadcast_to(np.eye(2), estimate.shape)
    exact = exact_common_bin_impedance(response_i, response_i @ impedance.T)
    return {"diagonal_ohm": [2., 3.], "tested_bins": len(bins),
            "equation10_complete_bins": int(np.isfinite(estimate).all(axis=(1, 2)).sum()),
            "same_frequency_current_matrix_condition": 1.,
            "equation9_max_absolute_error_ohm": float(np.max(abs(exact - impedance)))}


def save_csv(path, frequency, curves):
    with path.open("w", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(("case", "frequency_hz", "grid_relative_frobenius_error",
                         "source_relative_frobenius_error", "loop_relative_frobenius_error",
                         "min_distance_eq10_to_minus_one", "min_distance_eq9_to_minus_one"))
        for case, data in curves.items():
            distance = []
            for tag in ("loop_eq10", "loop_eq9"):
                loop = data[tag]
                valid = np.isfinite(loop).all(axis=(1, 2))
                values = np.full(len(loop), np.nan)
                values[valid] = np.min(abs(1 + np.linalg.eigvals(loop[valid])), axis=1)
                distance.append(values)
            for values in zip(frequency, data["grid_relative_error"], data["source_relative_error"],
                              data["loop_relative_error"], *distance, strict=True):
                writer.writerow((case, *values))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/protocol.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--figures", type=Path)
    args = parser.parse_args()
    out = args.out.resolve()
    figures = (args.figures or out / "figures").resolve()
    if (out / "run_manifest.json").exists() or (out / "case_metrics.json").exists():
        parser.error("Use a new output directory for each run.")
    out.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(args.protocol.read_text())
    journal, thesis, design = (protocol[k] for k in ("journal", "thesis", "independent_benchmark"))
    started, start = utc(), perf_counter()
    excitation = perturbations(protocol)
    ts = journal["acquisition_step_s"]
    f_grid = np.fft.rfftfreq(len(excitation), ts)
    lo, hi = journal["assessment_band_hz"]
    bins = np.flatnonzero((np.arange(len(f_grid)) % 2 == 0) & (f_grid >= lo) & (f_grid <= hi))
    frequency = f_grid[bins]
    full_frequency = np.r_[0., np.geomspace(design["gnc_min_positive_hz"], design["gnc_max_positive_hz"],
                                            design["gnc_positive_points"])]
    leakage = spectral_leakage(excitation)
    frame_s = len(excitation) * ts
    checks = {"stimuli_have_disjoint_parity": max(leakage.values()) <= protocol["gates"]["off_parity_spectral_energy_fraction"]}
    sequence = {
        "mlbs_chips": 2**journal["mlbs_order"] - 1, "irs_chips": 2 * (2**journal["mlbs_order"] - 1),
        "samples_per_common_frame": len(excitation), "common_frame_s": frame_s, "mlbs_period_s": frame_s / 2,
        "common_dft_spacing_hz": float(f_grid[1]), "mlbs_spacing_hz": float(2 * f_grid[1]),
        "analysis_bandwidth_from_generation_hz": journal["generation_hz"] * journal["practical_band_fraction_of_generation"],
        "selected_bin_count": len(bins), "first_selected_frequency_hz": float(frequency[0]), "last_selected_frequency_hz": float(frequency[-1]),
        "journal_as_mlbs_periods_s": journal["reported_average_periods"] * frame_s / 2,
        "journal_as_full_irs_frames_s": journal["reported_average_periods"] * frame_s,
        "thesis_as_mlbs_periods_s": thesis["reported_periods"] * frame_s / 2,
        "thesis_as_full_irs_frames_s": thesis["reported_periods"] * frame_s,
        "off_parity_energy_fraction": leakage,
    }
    results, curves, records, seconds = [], {}, None, {}
    for case in thesis["cases_table_5_3"]:
        t_case = perf_counter()
        row, data, waves = evaluate_case(case, protocol, excitation, bins, frequency, full_frequency)
        case_id = case["id"]
        seconds[case_id] = perf_counter() - t_case
        results.append(row)
        curves[case_id] = data
        if case_id == design["noise_case"]:
            records = waves
        print(f"{case_id} sampled radius={row['sampled_physical_spectral_radius']:.8f}"
              f" grid p95={row['grid_equation10_vs_equation9']['p95']:.3g}"
              f" twin p95={row['source_equation10_vs_equation9']['p95']:.3g}", flush=True)
    noise_start = perf_counter()
    noise = noise_experiment(records, curves[design["noise_case"]], bins, protocol)
    noise["synthetic_acquisition_duration_s"] = noise["full_common_frames"] * frame_s
    diagonal = zero_coupling_example(excitation, bins, protocol)
    checks["equation10_zero_denominator_is_detected"] = (
        diagonal["equation10_complete_bins"] == 0 and diagonal["equation9_max_absolute_error_ohm"] == 0.)
    seconds["noise_and_degeneracy"] = perf_counter() - noise_start
    all_checks = list(checks.values()) + [ok for row in results for ok in row["checks"].values()]
    metrics = {"sequence": sequence, "cases": results, "noise_replay": noise, "zero_coupling_example": diagonal,
               "global_checks": checks, "checks_passed": int(sum(all_checks)), "checks_total": len(all_checks)}
    dump(out / "case_metrics.json", metrics)
    save_csv(out / "identification_summary.csv", frequency, curves)
    np.savez_compressed(out / "impedance_curves.npz", frequency_hz=frequency, full_frequency_hz=full_frequency,
                        **{f"{case}__{key}": array for case, data in curves.items() for key, array in data.items()})
    np.savez_compressed(out / "example_waveforms.npz", **records)

    from plot_results import draw_figures
    figure_start = perf_counter()
    images = draw_figures(figures, results, curves, frequency, excitation, ts)
    seconds["figures"] = perf_counter() - figure_start
    data_files = ["case_metrics.json", "identification_summary.csv", "impedance_curves.npz", "example_waveforms.npz"]
    source_files = [ROOT / path for path in ("src/reproduce.py", "src/circuit.py", "src/identify.py",
                                            "src/plot_results.py", "verification/sideband.py")]
    dump(out / "run_manifest.json", {
        "schema_version": 1, "paper_doi": protocol["paper_doi"], "started_utc": started,
        "finished_utc": utc(), "wall_seconds": perf_counter() - start, "stage_seconds": seconds,
        "python": platform.python_version(), "platform": platform.platform(),
        "numpy": np.__version__, "scipy": scipy.__version__, "protocol_sha256": file_hash(args.protocol),
        "implementation_sha256": {str(path.relative_to(ROOT)): file_hash(path)
                                  for path in source_files},
        "data_sha256": {name: file_hash(out / name) for name in data_files},
        "figures_sha256": {path.name: file_hash(path) for path in images},
        "checks_passed": metrics["checks_passed"], "checks_total": metrics["checks_total"],
    })
    print(f"{metrics['checks_passed']}/{metrics['checks_total']} checks passed. {out / 'run_manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
