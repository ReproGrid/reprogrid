"""Audit Fig. 3 vector curves and replay the scope power deficit through Eq. (4)."""

import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.integrate import cumulative_trapezoid

from reproduce import CASE, Circuit, PERIOD, plt


def load_json(path):
    return json.loads(path.read_text())


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def polyline(drawings, number):
    path = drawings[number]["items"]
    if any(item[0] != "l" for item in path):
        raise ValueError(f"Expected a vector polyline at PDF path {number}")
    return np.asarray([tuple(path[0][1]), *(tuple(item[2]) for item in path)])


def calibrated_curve(drawings, path, cfg):
    points = polyline(drawings, path)
    left, right = cfg["right_panel_x_pt"]
    angle = (points[:, 0]-left)/(right-left) * (9*np.pi/4)
    power = np.interp(points[:, 1], [cfg["top_y_pt"], cfg["bottom_y_pt"]],
                     [cfg["power_top_pu"], cfg["power_bottom_pu"]])
    return np.column_stack((angle, power))


def vector_audit(page, config, model_summary, table):
    drawings = {draw["seqno"]: draw for draw in page.get_drawings()}
    traces = {
        "normal": calibrated_curve(drawings, config["normal_power_path"], config),
        "sag": calibrated_curve(drawings, config["sag_power_path"], config),
    }
    ref = table["active_reference_pu"]
    nominal = traces["normal"]
    first_period = nominal[(nominal[:, 0] >= -1e-6) & (nominal[:, 0] < PERIOD)]
    angle, power = first_period.T
    crossings = np.flatnonzero(np.signbit(power[:-1]-ref) != np.signbit(power[1:]-ref))
    roots = []
    for i in crossings:
        frac = (ref-power[i])/(power[i+1]-power[i])
        roots.append(float(angle[i] + frac*(angle[i+1]-angle[i])))
    sep, uep = roots
    sag_angles, sag_power = traces["sag"].T
    if np.any(np.diff(sag_angles) <= 0):
        raise ValueError("Sag polyline is not strictly ordered in angle")
    droop = model_summary["si_derived"]["droop_rad_s_per_pu"]

    def integrated_time(final_angle):
        mesh = np.r_[sep, sag_angles[(sep < sag_angles) & (sag_angles < final_angle)], final_angle]
        deficit = ref - np.interp(mesh, sag_angles, sag_power)
        if np.any(deficit <= 0):
            raise ValueError("The integration interval contains a fault-on equilibrium")
        # Integrate reciprocal power deficit exactly along each linear vector segment.
        fractions = np.diff(deficit)/deficit[:-1]
        ratio = np.ones_like(fractions)
        np.divide(np.log1p(fractions), fractions, out=ratio, where=fractions != 0)
        return float(np.sum(np.diff(mesh)/deficit[:-1]*ratio)/droop)

    plotted_clearances = {}
    for example, panel, end in (("short", "left", np.pi/8), ("long", "right", 9*np.pi/4)):
        vertices = polyline(drawings, config[f"{example}_clearing_path"])
        if np.ptp(vertices[:, 0]) > 1e-3:
            raise ValueError(f"Expected a vertical clearing jump for {example}")
        left, right = config[f"{panel}_panel_x_pt"]
        clear = float((vertices[0, 0]-left)/(right-left)*end)
        plotted_clearances[example] = {"angle_rad": clear, "time_implied_at_table_i_gain_s": integrated_time(clear),
                                      "beyond_uep": clear > uep}

    errors = {}
    for name, model in model_summary.items():
        circuit = Circuit(model["grid_resistance_pu"], model["grid_reactance_pu"],
                          table["voltage_reference_pu"], table["current_limit_pu"])
        errors[name] = {}
        for branch, v in (("normal", 1.), ("sag", .5)):
            curve = traces[branch]
            angle, power = curve[(curve[:, 0] >= -1e-6) & (curve[:, 0] < PERIOD)].T
            error = circuit.observe(angle, v)[0]-power
            errors[name][branch] = {"points": len(angle), "rmse_pu": float(np.sqrt(np.mean(error**2))),
                                    "max_abs_pu": float(max(abs(error)))}
    comparison = {"sep_rad": sep, "uep_rad": uep, "clearance_from_vector_seconds": integrated_time(uep),
                  "plotted_examples": plotted_clearances, "curve_errors": errors}
    return comparison, traces


def scope_audit(document, spec, config, model_trace, model):
    with Image.open(BytesIO(document.extract_image(spec["xref"])["image"])) as image:
        red, green, blue = np.moveaxis(np.asarray(image.convert("RGB"), dtype=np.int16), -1, 0)
    y0, y1 = config["power_row_window"]
    onset, clear = spec["sag_edges_x"]
    t_per_pixel = spec["seconds_per_div"]/config["pixels_per_horizontal_div"]
    scale = config["power_pu_per_div"]/config["pixels_per_vertical_div"]
    t = np.arange(clear-onset+1) * t_per_pixel
    power_deficits, excursions = [], []
    for threshold in config["blue_thresholds"]:
        color = (blue-red > threshold) & (blue-green > threshold)
        first = min(window[0] for window in config["baseline_windows_x_inclusive"])
        rows = []
        for x in range(first, clear):
            pixels = np.flatnonzero(color[y0:y1, x])
            if pixels.size == 0:
                raise ValueError(f"Power trace is missing in Fig. {spec['figure']} at x={x}")
            rows.append(np.median(pixels)+y0)
        rows = np.asarray(rows)
        for left, right in config["baseline_windows_x_inclusive"]:
            baseline = np.median(rows[left-first:right-first+1])
            for offset in config["baseline_pixel_offset_sensitivity"]:
                # The final pixel interval uses the last fault-on power sample.
                fault_rows = rows[onset-first:]
                deficit = (baseline+offset-np.r_[fault_rows, fault_rows[-1]]) * scale
                power_deficits.append(deficit)
                excursions.append(cumulative_trapezoid(deficit, t, initial=0) * model["droop_rad_s_per_pu"])
    excursions = np.stack(excursions)
    lo, hi = excursions.min(axis=0), excursions.max(axis=0)
    duration = t[-1]
    modeled_clearance = float(np.interp(spec["duration_ms"]/1000, model_trace[:, 0], model_trace[:, 1]))
    angle_boundary = model["postfault_uep_rad"]-model["postfault_sep_rad"]
    info = {
        "figure": spec["figure"], "nominal_duration_s": spec["duration_ms"]/1000, "raster_duration_s": duration,
        "baseline_method_variants": len(excursions),
        "replayed_fault_angle_increase_rad_range": [float(lo[-1]), float(hi[-1])],
        "nominal_model_fault_angle_increase_rad": modeled_clearance-model["postfault_sep_rad"],
        "original_basin_angle_increase_rad": angle_boundary,
        "range_contains_boundary": bool(lo[-1] <= angle_boundary <= hi[-1]),
        "mean_replayed_power_deficit_pu_range": [float(v/(model["droop_rad_s_per_pu"]*duration))
                                                 for v in (lo[-1], hi[-1])],
    }
    return info, np.column_stack((t, lo, hi, np.min(power_deficits, axis=0), np.max(power_deficits, axis=0)))


def plot(curves, scope, time_traces, model, table, destination):
    fig, axes = plt.subplots(1, 2, figsize=(10.25, 3.95), layout="constrained")
    circuit = Circuit(model["grid_resistance_pu"], model["grid_reactance_pu"],
                      table["voltage_reference_pu"], table["current_limit_pu"])
    for name, volts, color, label in (("normal", 1., "#2166ac", "1 pu"), ("sag", .5, "#a94c83", "0.5 pu")):
        trace = curves[name]
        interval = (trace[:, 0] >= -1e-6) & (trace[:, 0] <= .6)
        angle = trace[interval, 0]
        axes[0].plot(angle, trace[interval, 1], color=color, label=f"Fig. 3 · {label}")
        axes[0].plot(angle, circuit.observe(angle, volts)[0], color=".15", ls=":", lw=1.)
    axes[0].plot([], [], color=".15", ls=":", label="Table I model")
    axes[0].axhline(table["active_reference_pu"], color=".55", lw=.7)
    axes[0].set(xlabel="Converter-grid angle (rad)", ylabel="Terminal active power (pu)", title="Fig. 3 vector comparison")
    for audits, trace, color in zip(scope, time_traces, ("#2166ac", "#a94c21"), strict=True):
        duration = audits["nominal_duration_s"]
        axes[1].fill_between(trace[:, 0]*1000, trace[:, 1], trace[:, 2], color=color, alpha=.2)
        axes[1].plot(trace[:, 0]*1000, (trace[:, 1]+trace[:, 2])/2, color=color,
                     label=f"Fig. {audits['figure']} power replay")
        axes[1].scatter(duration*1000, audits["nominal_model_fault_angle_increase_rad"],
                        c=color, marker="x", zorder=4)
    axes[1].scatter([], [], c=".2", marker="x", label="Table I clearing angle increase")
    axes[1].axhline(model["postfault_uep_rad"]-model["postfault_sep_rad"], ls=":", color=".33", label="Original-basin boundary")
    axes[1].set(xlabel="Time from sag onset (ms)", ylabel="Fault angle increase (rad)", title="Eq. (4) with digitized scope power")
    for axis in axes:
        axis.legend(fontsize=7.5)
        axis.grid(alpha=.2)
    fig.savefig(destination, dpi=170)
    plt.close(fig)


def main():
    import pymupdf

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=CASE / "results")
    args = parser.parse_args()
    source = load_json(CASE / "sources/source_manifest.json")
    if digest(args.pdf) != source["sha256"]:
        raise ValueError("PDF differs from the version used for axis calibration")
    config_path = CASE / "configs/figure_audit.json"
    cfg = load_json(config_path)
    archive_path = CASE / "results/trajectories.npz"
    primary_path = CASE / "results/run_manifest.json"
    models = load_json(primary_path)["models"]
    si = models["si_derived"]
    table = load_json(CASE / "configs/protocol.json")["table_i"]
    with pymupdf.open(args.pdf) as document, np.load(archive_path, allow_pickle=False) as primary:
        vector, curves = vector_audit(document[cfg["pdf_page_one_based"]-1], cfg["fig3"], models, table)
        results = [scope_audit(document, spec, cfg["scope"], primary[f"si_derived_{spec['duration_ms']}ms"], si)
                   for spec in cfg["scope"]["panels"]]
    metrics, time_traces = zip(*results, strict=True)
    checks = {
        "vector_curves_match_si_model": all(row["max_abs_pu"] <= cfg["fig3"]["max_si_curve_abs_error_pu"]
                                           for row in vector["curve_errors"]["si_derived"].values()),
        "vector_clearance_agrees_with_si_model": (
            abs(vector["clearance_from_vector_seconds"]-si["same_basin_clearing_boundary_s"])
            <= cfg["fig3"]["max_clearance_time_difference_s"]
        ),
        "fig3_clearing_examples_straddle_boundary": (
            not vector["plotted_examples"]["short"]["beyond_uep"] and vector["plotted_examples"]["long"]["beyond_uep"]
        ),
        "scope_timebase_agrees_with_captions": all(
            abs(row["raster_duration_s"]-row["nominal_duration_s"]) <= cfg["scope"]["maximum_duration_readback_error_s"]
            for row in metrics
        ),
    }
    provenance = {"source_pdf_sha256": source["sha256"], "figure_protocol_sha256": digest(config_path),
                  "primary_manifest_sha256": digest(primary_path), "primary_trajectories_sha256": digest(archive_path),
                  "implementation_sha256": digest(Path(__file__)), "pymupdf_version": pymupdf.VersionBind}
    summary = {"fig3": vector, "oscilloscope": metrics, "checks": checks, "provenance": provenance,
               "trace_columns": {
                   "normal_and_sag": ["angle_rad", "power_pu"],
                   "scope_5_and_scope_6": ["time_s", "angle_increase_min_rad", "angle_increase_max_rad",
                                            "power_deficit_min_pu", "power_deficit_max_pu"],
               },
               "passed": all(checks.values())}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "figure_audit.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    np.savez_compressed(args.output / "figure_audit_traces.npz", **curves,
                        scope_5=time_traces[0], scope_6=time_traces[1])
    plot(curves, metrics, time_traces, si, table, args.output / "figure_audit.png")
    print(json.dumps(summary, allow_nan=False))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
