"""Reproduce observer frequency responses, pulse normalization and q-axis lock."""

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
from scipy.optimize import brentq

from observer import OBSERVER_NAMES, bandwidth, gains, poles, state_matrix, transfer
from standstill import (Motor, direction, fixed_rotor_vectors, held_voltage, into_frame, out_of_frame,
                        printed_detector, pulse_pair, rtvi_steady_error, run_puvi, switching_interval)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "verification"))
from independent import held_alpha_beta, observer_response


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def now():
    return datetime.now(timezone.utc).isoformat()


def read_figures(path):
    data = {}
    with path.open() as handle:
        for row in csv.DictReader(handle):
            key = (int(row["figure"]), row["axis"], row["series"])
            data.setdefault(key, []).append((float(row["abscissa"]), float(row["value"])))
    return {key: np.array(sorted(values)).T for key, values in data.items()}


def observer_evidence(paper, figures, gates):
    design = paper["fig8_design_bandwidth_rad_s"]
    damping = paper["fig8_damping"]
    f = np.geomspace(1e-2, 1e4, 3001)
    results, responses, checks = {}, {}, {}
    for kind in OBSERVER_NAMES:
        gain = gains(kind, damping, design)
        h = transfer(gain, 1j * f)
        responses[kind] = h
        ref = observer_response(*state_matrix(gain), f)
        response_error = float(max(abs(ref - h)))
        record = {"gains": gain.tolist(), "actual_3db_rad_s": bandwidth(gain),
                  "max_real_pole_rad_s": float(max(poles(gain).real)),
                  "state_space_vs_formula_max_absolute_error": response_error}
        for axis, tolerance in (("gain_db", "source_bode_magnitude_rmse_db"),
                                ("phase_deg", "source_bode_phase_rmse_deg")):
            omega, target = figures[8, axis, kind]
            modeled = transfer(gain, 1j * omega)
            values = 20 * np.log10(abs(modeled)) if axis == "gain_db" else np.degrees(np.angle(modeled))
            rmse = float(np.sqrt(np.mean((values - target)**2)))
            record[f"fig8_{axis}_rmse"] = rmse
            checks[f"{kind}_fig8_{axis}"] = rmse < gates[tolerance]
        checks[f"{kind}_independent_state_frequency"] = response_error < gates["algebraic_max_absolute_error"]
        results[kind] = record
    threshold = float(9**(-1 / 3))
    locations = [threshold - .001, threshold, threshold + .001]
    boundary = [{"damping": z, "rightmost_pole_real_rad_s": float(max(poles(gains("ESO_C2", z, design)).real))}
                for z in locations]
    checks["eso_c2_routh_boundary"] = boundary[0]["rightmost_pole_real_rad_s"] > 0 > boundary[-1]["rightmost_pole_real_rad_s"]
    checks["pi_design_bandwidth"] = abs(results["PI"]["actual_3db_rad_s"] - design) < 1e-10
    return {"fig8_damping": damping, "design_parameter_rad_s": design, "methods": results,
            "eso_c2_stability_threshold": threshold, "poles_near_boundary": boundary}, f, responses, checks


def normalization_evidence(motor, paper, settings, gates):
    ideal = replace(motor, resistance=0.)
    voltage = settings["injection_vector_amplitude_v"]
    angles = np.arange(-180., 181.)
    estimated = direction(np.zeros(len(angles)))
    rotor = fixed_rotor_vectors(-angles)
    _, measured = pulse_pair(ideal, np.zeros_like(estimated), rotor, estimated, voltage, 0)
    printed = printed_detector(np.radians(angles), ideal.inductances)
    discrepancy = float(max(abs(printed - measured)))
    factor_error = float(max(abs(printed - np.sqrt(2) * measured)))
    threshold = float(motor.saliency_factor * np.sin(np.radians(2 * paper["convergence_nominal_electrical_deg"])))
    lhs = brentq(lambda e: printed_detector(e, ideal.inductances) / np.sqrt(2) - threshold, 0., np.pi / 4)
    rhs = brentq(lambda e: printed_detector(e, ideal.inductances) - threshold, 0., np.pi / 4)
    rtvi = np.array([rtvi_steady_error(.41, .41 + e, ideal, voltage,
                                      2 * np.pi * settings["rtvi_analytical_probe_frequency_hz"])
                     for e in np.radians(angles)])
    rtvi_error = float(max(abs(rtvi - np.sin(np.radians(2 * angles)))))
    evidence = {"left_vs_printed_max_absolute_difference": discrepancy,
                "sqrt2_times_left_vs_printed_max_absolute_error": factor_error,
                "fig4_small_error_detector_gain_left": float(1 / np.sqrt(2)),
                "fig4_small_error_detector_gain_right": 1.,
                "paper_normalized_detector_threshold": threshold,
                "ideal_near_d_axis_limit_from_left_deg": float(np.degrees(lhs)),
                "ideal_near_d_axis_limit_from_right_deg": float(np.degrees(rhs)),
                "rtvi_eq8_vs_independent_steady_dq_max_error": rtvi_error}
    checks = {"eq15_sqrt2_factor": factor_error < gates["algebraic_max_absolute_error"] and discrepancy > .1,
              "eq8_rotating_steady_state": rtvi_error < gates["algebraic_max_absolute_error"]}
    return evidence, {"angle_deg": angles, "left": measured, "right": printed}, checks


def circuit_evidence(motor, settings, gates):
    rotor = fixed_rotor_vectors(np.arange(0., 360., 15.))
    estimated = direction(np.linspace(-1.3, 4.6, len(rotor)))
    voltage = settings["injection_vector_amplitude_v"] * estimated
    current = .11 * estimated
    trial = held_voltage(motor, into_frame(current, rotor), into_frame(voltage, rotor), motor.sample_period)
    direct = np.array([held_alpha_beta(motor.resistance, motor.inductances, axis, i, v, motor.sample_period)
                       for axis, i, v in zip(rotor, current, voltage, strict=True)])
    err_current = float(np.max(abs(out_of_frame(trial, rotor) - direct)))
    ideal = replace(motor, resistance=0.)
    averaged = held_voltage(ideal, into_frame(current, rotor), into_frame(voltage, rotor), ideal.sample_period)
    switched = [switching_interval(ideal, into_frame(current, rotor), voltage, rotor,
                                   settings["ideal_dc_bus_v"], rising) for rising in (True, False)]
    err_switch = float(np.max(abs(np.array(switched) - averaged)))
    end_a, eps_a = pulse_pair(motor, np.zeros_like(rotor), rotor, estimated,
                              settings["injection_vector_amplitude_v"], 0)
    end_b, eps_b = pulse_pair(motor, np.zeros_like(rotor), -rotor, estimated,
                              settings["injection_vector_amplitude_v"], 0)
    err_polarity = float(max(np.max(abs(out_of_frame(end_a, rotor) - out_of_frame(end_b, -rotor))),
                             max(abs(eps_a - eps_b))))
    evidence = {"stationary_matrix_vs_rotor_frame_max_current_error_a": err_current,
                "ideal_switch_vs_held_voltage_endpoint_error_a": err_switch,
                "polarity_reversal_measurement_difference": err_polarity}
    checks = {"independent_matrix_current": err_current < gates["independent_matrix_current_error_a"],
              "same_input_switch_endpoint": err_switch < gates["same_input_ideal_switch_endpoint_error_a"],
              "linear_motor_cannot_resolve_polarity": err_polarity < gates["algebraic_max_absolute_error"]}
    return evidence, checks


def axis_error(theta, degrees):
    return np.degrees((theta - np.radians(degrees) + np.pi / 2) % np.pi - np.pi / 2)


def first_held(valid, time, count):
    lengths = np.zeros(valid.shape[1], dtype=int)
    first = np.full(valid.shape[1], -1, dtype=int)
    for k, good in enumerate(valid, 1):
        lengths = (lengths + 1) * good
        first[(first < 0) & (lengths >= count)] = k
    return first, np.where(first >= 0, time[first], np.nan)


def summarize_sweep(degrees, time, angle, detector, paper, eps_threshold):
    tolerance = paper["convergence_nominal_electrical_deg"]
    hold_groups = int(np.ceil(paper["convergence_hold_s"] / time[1]))
    error = axis_error(angle, degrees)
    detector_valid = abs(detector) <= eps_threshold
    flag_index, flag_time = first_held(detector_valid, time, hold_groups)
    _, joint_time = first_held(detector_valid & (abs(error[:-1]) <= tolerance)
                              & (abs(error[1:]) <= tolerance), time, hold_groups)
    flagged = flag_index >= 0
    error_on_flag = np.where(flagged, error[flag_index, np.arange(len(degrees))], np.nan)
    accepted = np.isfinite(joint_time)
    record = {"angles": len(degrees), "detector_flagged_angles": int(flagged.sum()),
              "flagged_with_axis_error_over_nominal_at_declaration": int((abs(error_on_flag) > tolerance).sum()),
              "joint_converged_angles": int(accepted.sum()), "hold_duration_implemented_s": hold_groups * time[1],
              "median_joint_time_s": float(np.median(joint_time[accepted])) if accepted.any() else None,
              "max_joint_time_s": float(max(joint_time[accepted])) if accepted.any() else None,
              "terminal_max_abs_axis_error_deg": float(max(abs(error[-1])))}
    return record, {"angle_deg": degrees, "time_s": time, "estimate_rad": angle, "raw_detector": detector,
                    "axis_error_deg": error, "flag_time_s": flag_time, "joint_time_s": joint_time,
                    "error_on_flag_deg": error_on_flag}


def simulate(paper, motor, settings, threshold):
    pi = gains("PI", settings["simulation_observer_damping"], paper["fig13_design_bandwidth_rad_s"]["puvi_pi"])
    eso = gains("ESO_C2", settings["simulation_observer_damping"], paper["fig13_design_bandwidth_rad_s"]["puvi_eso_c2"])
    ideal_switch_bus = settings["ideal_dc_bus_v"]
    fine = np.arange(0., 360., settings["rotor_angle_increment_deg"])
    coarse = np.arange(0., 360., settings["switched_angle_increment_deg"])
    axes = np.array([0., 90., 180., 270.])
    seeded = settings["initial_speed_seed_rad_s"]
    setups = (
        ("pi_averaged", fine, pi, seeded, None, 1.),
        ("eso_averaged", fine, eso, seeded, None, 1.),
        ("pi_switched", coarse, pi, seeded, ideal_switch_bus, 1.),
        ("eso_switched", coarse, eso, seeded, ideal_switch_bus, 1.),
        ("pi_zero_seed", axes, pi, 0., None, 1.),
        ("eso_zero_seed", axes, eso, 0., None, 1.),
        ("pi_scaled_sqrt2", fine, pi, seeded, None, np.sqrt(2)),
        ("eso_scaled_sqrt2", fine, eso, seeded, None, np.sqrt(2)),
    )
    stats, traces = {}, {}
    for tag, angles, gain, seed, bus, scaling in setups:
        t, state, error = run_puvi(motor, angles, gain, settings, seed, bus=bus, detector_scale=scaling)
        stats[tag], traces[tag] = summarize_sweep(angles, t, state, error, paper, threshold)
        print(f"{tag}: {stats[tag]['joint_converged_angles']}/{len(angles)} joint converged.", flush=True)
    seed_sweep = []
    near_q = np.array([89., 90., 91., 269., 270., 271.])
    for tag, gain in (("PI", pi), ("ESO_C2", eso)):
        for seed in settings["seed_sweep_rad_s"]:
            t, state, error = run_puvi(motor, near_q, gain, settings, seed)
            row, _ = summarize_sweep(near_q, t, state, error, paper, threshold)
            seed_sweep.append({"observer": tag, "initial_speed_rad_s": seed, **row})
    return stats, traces, seed_sweep


def save_angle_csv(path, traces):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("run", "rotor_electrical_angle_deg", "terminal_axis_error_deg",
                         "detector_flag_s", "error_at_flag_deg", "joint_axis_detector_flag_s"))
        for tag, record in traces.items():
            values = zip(record["angle_deg"], record["axis_error_deg"][-1], record["flag_time_s"],
                         record["error_on_flag_deg"], record["joint_time_s"], strict=True)
            writer.writerows([tag, *("" if np.isnan(v) else v for v in row)] for row in values)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--figures", type=Path)
    args = parser.parse_args()
    out = args.out.resolve()
    figures_path = (args.figures or out / "figures").resolve()
    if (out / "metrics.json").exists() or (out / "run_manifest.json").exists():
        parser.error("Use a new output directory for each run.")
    out.mkdir(parents=True, exist_ok=True)
    figures_path.mkdir(parents=True, exist_ok=True)
    t0, started = perf_counter(), now()
    config_path = ROOT / "configs/protocol.json"
    source_path = ROOT / "verification/fig8_9_readback.csv"
    config = json.loads(config_path.read_text())
    paper, settings, gates = (config[k] for k in ("paper", "reconstruction_settings", "gates"))
    motor = Motor(paper["standstill_resistance_ohm"], np.array([paper["d_inductance_h"], paper["q_inductance_h"]]),
                  paper["sampling_period_s"])
    source = read_figures(source_path)
    analytical, frequency, response, obs_checks = observer_evidence(paper, source, gates)
    normalization, detector_curves, norm_checks = normalization_evidence(motor, paper, settings, gates)
    circuit, circuit_checks = circuit_evidence(motor, settings, gates)
    stats, traces, seed_sweep = simulate(paper, motor, settings, normalization["paper_normalized_detector_threshold"])
    t, angle = source[9, "angle_deg", "estimated_rotor"]
    true_t, true_angle = source[9, "angle_deg", "true_rotor"]
    end = t >= .03
    paper_axis_error = axis_error(np.radians(angle[end]), np.interp(t[end], true_t, true_angle))
    paper_q_axis_deviation = float(max(90. - abs(paper_axis_error)))
    evidence = {"paper_last_20ms_max_deviation_from_q_axis_deg": paper_q_axis_deviation,
                "paper_last_20ms_mean_axis_error_deg": float(np.mean(paper_axis_error)),
                "q_axis_classification_tolerance_from_paper_deg": paper["convergence_nominal_electrical_deg"]}
    check_lock = {"fig9_source_trace_identifies_q_axis": paper_q_axis_deviation < paper["convergence_nominal_electrical_deg"]}
    for name in ("pi", "eso"):
        run = traces[f"{name}_zero_seed"]
        exact_q = np.isin(run["angle_deg"], [90., 270.])
        end_error = float(min(abs(run["axis_error_deg"][-1, exact_q])))
        evidence[f"{name}_zero_seed_min_terminal_axis_error_deg"] = end_error
        check_lock[f"{name}_reproduces_exact_q_equilibrium_and_false_flag"] = (
            abs(90. - end_error) < gates["exact_model_q_axis_equilibrium_tolerance_deg"]
            and np.isfinite(run["flag_time_s"][exact_q]).all() and np.isnan(run["joint_time_s"][exact_q]).all())
        for route in ("averaged", "switched"):
            seeded = traces[f"{name}_{route}"]
            q = np.isin(seeded["angle_deg"], [90., 270.])
            check_lock[f"{name}_{route}_speed_seed_escapes_q_axis"] = bool(np.isfinite(seeded["joint_time_s"][q]).all())
    checks = {k: bool(v) for k, v in {**obs_checks, **norm_checks, **circuit_checks, **check_lock}.items()}
    metrics = {"doi": config["doi"], "observer_analysis": analytical, "normalization": normalization,
               "circuit_verification": circuit, "figure9_q_lock": evidence, "sweeps": stats, "seed_sweep": seed_sweep,
               "checks": checks, "checks_passed": sum(checks.values()), "checks_total": len(checks)}
    dump(out / "metrics.json", metrics)
    save_angle_csv(out / "angle_sweep.csv", traces)
    np.savez_compressed(out / "observer_curves.npz", omega_rad_s=frequency, **response, **detector_curves)
    np.savez_compressed(out / "pulse_traces.npz",
                        **{f"{tag}__{field}": value for tag, row in traces.items() for field, value in row.items()})
    from plots import draw
    figures = draw(figures_path, frequency, response, detector_curves, normalization, traces, source)
    outputs = ["metrics.json", "angle_sweep.csv", "observer_curves.npz", "pulse_traces.npz"]
    sources = ["src/observer.py", "src/standstill.py", "src/reproduce.py", "src/plots.py",
               "verification/independent.py", "verification/fig8_9_readback.csv"]
    dump(out / "run_manifest.json", {
        "schema_version": 1, "paper_doi": config["doi"], "started_utc": started, "finished_utc": now(),
        "wall_seconds": perf_counter() - t0, "python": platform.python_version(), "platform": platform.platform(),
        "numpy": np.__version__, "scipy": scipy.__version__, "protocol_sha256": digest(config_path),
        "implementation_sha256": {rel: digest(ROOT / rel) for rel in sources},
        "data_sha256": {rel: digest(out / rel) for rel in outputs},
        "figures_sha256": {path.name: digest(path) for path in figures},
        "checks_passed": metrics["checks_passed"], "checks_total": metrics["checks_total"],
    })
    print(f"{metrics['checks_passed']}/{metrics['checks_total']} checks passed. {out}", flush=True)


if __name__ == "__main__":
    main()
