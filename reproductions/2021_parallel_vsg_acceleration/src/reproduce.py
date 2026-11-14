"""Reconstruct the Bode curves and pole loci of the two-VSG system."""

import argparse
from collections import defaultdict
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
from scipy.linalg import expm
from scipy.optimize import brentq, linear_sum_assignment
from scipy.signal import find_peaks

from model import augmented, operating_point, response, rightmost

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "verification"))
from independent import nodal_response, phasor_slopes, reference_step, swing_filter


LABELS = ("traditional", "power_only", "acceleration_only", "combined")
BODE_CSV = "verification/fig5_6_bode.csv"
POLES_CSV = "verification/fig8_9_poles.csv"


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def get_csv(path, fields, values):
    data = defaultdict(list)
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            data[tuple(row[key] for key in fields)].append(tuple(float(row[key]) for key in values))
    return {key: np.asarray(rows) for key, rows in data.items()}


def grid(bounds, step):
    low, high = bounds
    return low + step * np.arange(round((high - low) / step) + 1)


def controllers(paper):
    gains = np.asarray(paper["nominal_controller_by_converter"])
    bank = {}
    for label, off in zip(LABELS, [(0, 2), (0,), (2,), ()], strict=True):
        current = gains.copy()
        current[:, off] = 0
        bank[label] = current
    return bank


def solver_error(first, second):
    distances = abs(first[:, None] - second[None, :])
    a, b = linear_sum_assignment(distances)
    return distances[a, b]


def summarize_operating_point(paper, bus_voltage, line, active):
    point = operating_point(paper, bus_voltage, line, active)
    phasor_k, phasor_angle, fd_k = phasor_slopes(paper, bus_voltage, line, active)
    return point, {
        "bus_voltage_pu": bus_voltage, "line_inductance_mh": list(line), "active_power_by_converter_pu": list(active),
        "base_impedance_ohm": point.base_impedance_ohm, "reactance_by_converter_ohm": point.reactance_ohm.tolist(),
        "angle_by_converter_rad": point.angle_rad.tolist(), "stiffness_by_converter_pu_per_rad": point.stiffness.tolist(),
        "phasor_angle_error_rad": float(np.max(abs(point.angle_rad - phasor_angle))),
        "phasor_slope_error": float(np.max(abs(point.stiffness - phasor_k))),
        "finite_difference_slope_error": float(np.max(abs(point.stiffness - fd_k))),
    }


def step_response(system, time, size):
    a, b, c, d = system
    n = len(a)
    block = np.zeros((n + 1, n + 1))
    block[:n, :n], block[:n, n] = a, size * b
    transition = expm((time[1] - time[0]) * block)
    state = np.zeros((len(time), n))
    for i in range(1, len(time)):
        state[i] = transition[:n, :n] @ state[i - 1] + transition[:n, n]
    return state @ c.T + size * d


def bode_result(point, inertia, droop, gain, frequency, reference):
    system = augmented(point.stiffness, inertia, droop, point.omega_n, gain)
    other = swing_filter(point.stiffness, inertia, droop, point.omega_n, gain)
    omega = 2j * np.pi * frequency
    values = response(system, omega)
    from_blocks = response(other, omega)
    from_network = nodal_response(point.stiffness, inertia, droop, point.omega_n, gain, omega)
    magnitude = 20 * np.log10(abs(values[:, 0]))
    peaks = find_peaks(magnitude)[0]
    data = {
        "local_maxima": [{"frequency_hz": float(frequency[i]), "gain_db": float(magnitude[i])} for i in peaks],
        "gain_100hz_db": float(magnitude[-1]),
        "dc_share": response(system, 0.).tolist(),
        "high_frequency_share": system[3].tolist(),
        "high_frequency_share_max_error": float(np.max(abs(system[3] - point.stiffness / point.stiffness.sum()))),
        "swing_filter_max_complex_error": float(np.max(abs(values - from_blocks))),
        "nodal_max_complex_error": float(np.max(abs(values - from_network))),
        "power_balance_max_error": float(np.max(abs(values.sum(axis=1) - 1))),
        "eigenvalue_max_pair_error_per_s": float(np.max(solver_error(np.linalg.eigvals(system[0]), np.linalg.eigvals(other[0])))),
    }
    if reference:
        for quantity, points in reference.items():
            plot_frequency, target = points.T
            z = response(system, 2j * np.pi * plot_frequency)[:, 0]
            error = 20 * np.log10(abs(z)) - target if quantity == "magnitude_db" else np.angle(z, deg=True) - target
            data[quantity + "_rmse"] = float(np.sqrt(np.mean(error**2)))
            data[quantity + "_max_absolute_error"] = float(np.max(abs(error)))
    return system, other, values, data


def run_bode(paper, setup, frequency, source):
    inertia, droop = (np.asarray(paper[key]) for key in ("inertia_s", "droop_pu"))
    bank = controllers(paper)
    point, point_metrics = summarize_operating_point(
        paper, setup["bus_voltage_pu"], paper["line_inductance_mh"], paper["active_reference_pu"]
    )
    fig5, fig6, step_data = {}, {}, {}
    values = {"frequency_hz": frequency}
    step = setup["practical_step"]
    time = grid((0., step["duration_s"]), step["sample_s"])
    step_values = {"time_s": time}
    for name, gains in bank.items():
        refs = {key[2]: source[key] for key in source if key[:2] == ("5", name)}
        system, other, values["fig5_" + name], fig5[name] = bode_result(point, inertia, droop, gains, frequency, refs)
        primary = step_response(system, time, step["amplitude_pu"])
        reference = reference_step(other, time, step["amplitude_pu"])
        steady = step["amplitude_pu"] * response(system, 0.)
        above_band = np.flatnonzero(abs(primary[:, 0] - steady[0]) > .02 * abs(steady[0]))
        first_settled_index = above_band[-1] + 1 if len(above_band) else 0
        settle = float(time[first_settled_index]) if first_settled_index < len(time) else None
        step_values[name] = primary
        step_data[name] = {"peak_increment_p1_pu": float(primary[:, 0].max()), "steady_increment_p1_pu": float(steady[0]),
                           "settling_2_percent_p1_s": settle,
                           "reference_max_power_error_pu": float(np.max(abs(reference - primary)))}
    for inductance in paper["figure6_l_s1_mh"]:
        name = str(inductance)
        line = [inductance, paper["line_inductance_mh"][1]]
        case, record = summarize_operating_point(paper, setup["bus_voltage_pu"], line, paper["active_reference_pu"])
        refs = {key[2]: source[key] for key in source if key[:2] == ("6", name)}
        _, _, values["fig6_" + name], fig6[name] = bode_result(case, inertia, droop, bank["combined"], frequency, refs)
        fig6[name]["operating_point"] = record
    inputs = setup["sensitivity"]
    alternatives = {
        "aggregate_preload": (setup["bus_voltage_pu"], [inputs["alternative_aggregate_preload_pu"] / 2] * 2),
        **{f"bus_voltage_{v:.2f}": (v, paper["active_reference_pu"]) for v in inputs["bus_voltages_pu"]},
    }
    sensitivity = {}
    for name, (v, active) in alternatives.items():
        case, record = summarize_operating_point(paper, v, paper["line_inductance_mh"], active)
        record["bode"] = {}
        for label, gains in bank.items():
            _, _, values[f"sensitivity_{name}_{label}"], record["bode"][label] = bode_result(case, inertia, droop, gains, frequency, {})
        sensitivity[name] = record
    return point, {"operating_point": point_metrics, "fig5": fig5, "fig6": fig6, "step": step_data,
                   "operating_point_sensitivity": sensitivity}, values, step_values


def pole_sweep(point, inertia, droop, control, values, parameter, count, source):
    poles, errors, plot_errors = [], [], []
    for value in values:
        h, gains = inertia.copy(), control.copy()
        if parameter == "H1":
            h[0] = value
        else:
            gains[0, 0] = value
        a = augmented(point.stiffness, h, droop, point.omega_n, gains)[0]
        ref = swing_filter(point.stiffness, h, droop, point.omega_n, gains)[0]
        roots, reference_roots = np.linalg.eigvals(a), np.linalg.eigvals(ref)
        roots = rightmost(roots, len(roots))
        poles.append(roots)
        errors.extend(solver_error(roots, reference_roots))
        if value in source:
            allowed = roots[:count] if parameter == "H1" else roots
            plot_errors.extend(solver_error(source[value], allowed))
    poles = np.asarray(poles, dtype=complex)
    visible = poles[:, :count]
    report = {
        "reference_max_error_per_s": float(max(errors)),
        "rightmost_real_part_max_per_s": float(visible[:, 0].real.max()),
        "rightmost_real_parts_endpoints_per_s": visible[[0, -1], 0].real.tolist(),
        "imaginary_magnitude_max_of_three_rightmost_per_s": float(abs(visible.imag).max()),
        "plotted_points_compared": len(plot_errors),
    }
    if len(plot_errors):
        report.update(plot_rmse_per_s=float(np.sqrt(np.mean(np.square(plot_errors)))),
                      plot_max_error_per_s=float(max(plot_errors)))
    return poles, report


def run_loci(paper, setup, point, source):
    inertia, droop = (np.asarray(paper[key]) for key in ("inertia_s", "droop_pu"))
    bank = controllers(paper)
    count = setup["dominant_pole_count"]
    h_values = grid(paper["figure8_h1_s"], setup["figure8_inertia_step_s"])
    k_values = grid(paper["figure9_k11"], setup["figure9_k11_step"])
    dense_k = grid(paper["figure9_k11"], setup["dense_figure9_k11_step"])
    arrays = {"fig8_h1_s": h_values, "fig9_k11": k_values, "fig9_dense_k11": dense_k}
    metrics = {"fig8": {}, "fig9": {}}
    for panel, key in (("8a", "traditional"), ("8b", "combined")):
        markers = {float(k[2]): p[:, 0] + 1j * p[:, 1] for k, p in source.items() if k[:2] == (panel, key)}
        arrays["fig8_" + key], data = pole_sweep(point, inertia, droop, bank[key], h_values, "H1", count, markers)
        metrics["fig8"][key] = data
    for k12 in paper["figure9_k12"]:
        key = str(k12)
        gains = bank["combined"].copy()
        gains[0, 1] = k12
        markers = {float(k[2]): p[:, 0] + 1j * p[:, 1] for k, p in source.items() if k[:2] == ("9", key)}
        arrays["fig9_" + key], data = pole_sweep(point, inertia, droop, gains, k_values, "k11", count, markers)
        all_poles, dense_data = pole_sweep(point, inertia, droop, gains, dense_k, "k11", count, {})
        arrays["fig9_dense_" + key] = all_poles
        nonreal = np.max(abs(all_poles.imag), axis=1) > 1e-6
        near_axis = abs(all_poles.real) < 45
        nonreal_visible = ((abs(all_poles.imag) > 1e-6) & near_axis).any(axis=1)
        if nonreal_visible.any() and not nonreal_visible[-1]:
            last = np.flatnonzero(nonreal_visible)[-1]
            data["near_axis_transition_k11_bracket"] = dense_k[last:last + 2].tolist()
        data["k11_with_complex_poles_anywhere"] = dense_k[nonreal][[0, -1]].tolist() if nonreal.any() else None
        data["k11_with_complex_poles_within_plot_width"] = dense_k[nonreal_visible][[0, -1]].tolist() if nonreal_visible.any() else None
        data["dense_reference_max_error_per_s"] = dense_data["reference_max_error_per_s"]
        data["rightmost_real_part_max_per_s"] = dense_data["rightmost_real_part_max_per_s"]
        metrics["fig9"][key] = data
    return metrics, arrays


def comparison_checks(paper, gates, data, values):
    plotted = list(data["fig5"].values()) + list(data["fig6"].values())
    variants = [spec for v in data["operating_point_sensitivity"].values() for spec in v["bode"].values()]
    records = plotted + variants
    poles = list(data["fig8"].values()) + list(data["fig9"].values())
    pole_errors = [r["eigenvalue_max_pair_error_per_s"] for r in records]
    pole_errors += [max(r["reference_max_error_per_s"], r.get("dense_reference_max_error_per_s", 0.)) for r in poles]
    points = [data["operating_point"], *[r["operating_point"] for r in data["fig6"].values()],
              *data["operating_point_sensitivity"].values()]
    droop = 1 / np.asarray(paper["droop_pu"])
    ideal_dc = droop / droop.sum()
    check = {
        "synchronizing_slope_phasor_derivative": max(p["phasor_slope_error"] for p in points) < gates["phasor_slope_max_error"],
        "synchronizing_slope_finite_difference": max(p["finite_difference_slope_error"] for p in points) < gates["phasor_slope_max_error"],
        "operating_angle_against_explicit_phasor": max(p["phasor_angle_error_rad"] for p in points) < 1e-11,
        "figure7_swing_filter_frequency_response": max(v["swing_filter_max_complex_error"] for v in records) < gates["transfer_max_complex_error"],
        "network_frequency_response": max(v["nodal_max_complex_error"] for v in records) < gates["transfer_max_complex_error"],
        "swing_filter_eigenvalues": max(pole_errors) < gates["pole_max_pair_error_per_s"],
        "output_power_balance": max(v["power_balance_max_error"] for v in records) < gates["power_balance_error_pu"],
        "low_frequency_droop_sharing": all(np.max(abs(v["dc_share"] - ideal_dc)) < gates["dc_and_high_frequency_limit_error"] for v in records),
        "high_frequency_algebraic_sharing": max(v["high_frequency_share_max_error"] for v in records) < gates["dc_and_high_frequency_limit_error"],
        "independent_load_step": max(v["reference_max_power_error_pu"] for v in data["step"].values()) < gates["step_solver_max_power_error_pu"],
        "sampled_pole_stability": all(v["rightmost_real_part_max_per_s"] < 0 for v in poles),
        "figure6_nominal_consistency": np.array_equal(values["fig5_combined"], values["fig6_1.0"]),
    }
    source = {
        "bode_magnitude": max(v["magnitude_db_rmse"] for v in plotted) < gates["bode_magnitude_rmse_db"],
        "bode_phase": max(v["phase_deg_rmse"] for v in plotted) < gates["bode_phase_rmse_deg"],
        "reported_resonances": all(any(abs(peak["frequency_hz"] - f) / f < gates["resonance_relative_error"]
                                       for peak in data["fig5"][name]["local_maxima"])
                                   for name, f in paper["figure5_reported_resonance_hz"].items()),
        "figure8_loci": max(v["plot_rmse_per_s"] for v in data["fig8"].values()) < gates["pole_locus_rmse_per_s"],
        "figure9_loci": max(v["plot_rmse_per_s"] for v in data["fig9"].values()) < gates["pole_locus_rmse_per_s"],
    }
    return {k: bool(v) for k, v in check.items()}, {k: bool(v) for k, v in source.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--figures", type=Path)
    args = parser.parse_args()
    out = args.out
    figures = args.figures or out / "figures"
    out.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    started, clock = datetime.now(timezone.utc), perf_counter()
    protocol = ROOT / "configs/protocol.json"
    cfg = json.loads(protocol.read_text())
    paper, setup, gates = (cfg[key] for key in ("paper", "reconstruction", "gates"))
    lo, hi, size = setup["frequency_hz"]
    frequency = np.geomspace(lo, hi, size)
    bode_source = get_csv(ROOT / BODE_CSV, ("figure", "series", "quantity"), ("frequency_hz", "value"))
    poles_source = get_csv(ROOT / POLES_CSV, ("figure", "series", "sweep_value"), ("real_per_s", "imag_per_s"))
    point, metrics, bode, step = run_bode(paper, setup, frequency, bode_source)
    locus_metrics, loci = run_loci(paper, setup, point, poles_source)
    metrics.update(doi=cfg["doi"], **locus_metrics)
    checks, comparison = comparison_checks(paper, gates, metrics, bode)
    metrics.update(checks=checks, checks_passed=sum(checks.values()), checks_total=len(checks), paper_comparisons=comparison)
    write_json(out / "metrics.json", metrics)
    np.savez_compressed(out / "frequency_responses.npz", **bode)
    np.savez_compressed(out / "pole_loci.npz", **loci)
    np.savez_compressed(out / "load_step.npz", **step)
    from plots import draw
    plots = draw(figures, bode, loci, bode_source, poles_source, step)
    inputs = ("src/model.py", "src/reproduce.py", "src/plots.py", "verification/independent.py", BODE_CSV, POLES_CSV)
    outputs = ("metrics.json", "frequency_responses.npz", "pole_loci.npz", "load_step.npz")
    write_json(out / "run_manifest.json", {
        "schema_version": 1, "paper_doi": cfg["doi"], "started_utc": started.isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(), "wall_seconds": perf_counter() - clock,
        "python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__, "scipy": scipy.__version__,
        "protocol_sha256": digest(protocol), "implementation_sha256": {name: digest(ROOT / name) for name in inputs},
        "data_sha256": {name: digest(out / name) for name in outputs}, "figures_sha256": {path.name: digest(path) for path in plots},
        "checks_passed": metrics["checks_passed"], "checks_total": metrics["checks_total"],
    })
    print(f"{metrics['checks_passed']}/{metrics['checks_total']} numerical checks. "
          f"{sum(comparison.values())}/{len(comparison)} figure checks.", flush=True)
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
