"""Independent Fig. 1/2 checks, Fig. 3 readback, and measured-input droop replay."""

from argparse import ArgumentParser
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import brentq

from phasor_circuit import PhasorCircuit, scalar_bound
from reproduce import Circuit
from sampled_circuit import DigitalGFM, Electrical
from plot_extended import render


ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def load_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def get_electrical(table, filter_values):
    omega = 2 * np.pi * table["frequency_hz"]
    s_base = table["active_reference_w"] / table["active_reference_pu"]
    voltage = table["voltage_phase_peak_v"]
    z_base = 1.5 * voltage**2 / s_base
    si = Electrical(omega, table["grid_resistance_ohm"]/z_base, omega*table["grid_inductance_h"]/z_base,
                    omega*filter_values["converter_inductance_h"]/z_base, omega*filter_values["capacitor_f"]*z_base)
    printed = Electrical(omega, table["grid_resistance_printed_pu"], table["grid_reactance_printed_pu"],
                         filter_values["converter_inductance_printed_pu"], filter_values["capacitor_printed_pu"])
    return {"si": si, "printed_pu": printed}


def phasor_studies(protocol, extension, electrical, readback, arrays):
    table = protocol["table_i"]
    sag = protocol["fault_grid_voltage_pu"]
    samples = [item["duration_s"] for item in protocol["paper_test_s"]]
    reference, limit, setpoint = table["voltage_reference_pu"], table["current_limit_pu"], table["active_reference_pu"]
    paper_curve, fine_curve = readback["fig3"], readback["fig3_fine"]
    main_angle = paper_curve[:, 0]
    pdf_sep, pdf_uep = [brentq(lambda a: np.interp(a, fine_curve[:, 0], fine_curve[:, 1])-setpoint, lo, hi)
                        for lo, hi in ((.1, .25), (.30, .39))]
    model = {}
    for name, circuit in electrical.items():
        reduced = Circuit(circuit.r_grid, circuit.x_grid, reference, limit)

        def reduced_power(angle, voltage):
            return reduced.observe(angle, voltage)[0]

        cap = PhasorCircuit(circuit.r_grid, circuit.x_grid, circuit.b_cap, reference, limit)
        no_cap = PhasorCircuit(circuit.r_grid, circuit.x_grid, 0., reference, limit)
        gain = table["power_droop_pu"]*circuit.omega
        row = {"electrical_pu": {"r_grid": circuit.r_grid, "x_grid": circuit.x_grid, "x_filter": circuit.x_filter, "b_cap": circuit.b_cap},
               "without_capacitor": scalar_bound(reduced_power, sag, setpoint, gain, samples),
               "with_capacitor": scalar_bound(cap.power, sag, setpoint, gain, samples),
               "capacitor_removal_power_error_pu": max(float(np.max(np.abs(no_cap.power(main_angle, v)-reduced_power(main_angle, v)))) for v in (1., sag)),
               "capacitor_full_kcl_error_pu": max(cap.independent_error(v) for v in (1., sag)),
               "fig3_pdf_global_curve_max_error_pu": [float(np.max(np.abs(paper_curve[:, j]-reduced_power(main_angle, v))))
                                                       for j, v in enumerate((1., sag), 1)],
               "fig3_pdf_enlarged_curve_max_error_pu": [float(np.max(np.abs(fine_curve[:, j]-reduced_power(fine_curve[:, 0], v))))
                                                          for j, v in enumerate((1., sag), 1)]}
        model[name] = row
        for suffix, fun in (("without_cap", reduced_power), ("with_cap", cap.power)):
            arrays[f"fig3_{name}_{suffix}"] = np.column_stack([main_angle, fun(main_angle, 1.), fun(main_angle, sag)])
            fine = fine_curve[:, 0]
            arrays[f"fig3_fine_{name}_{suffix}"] = np.column_stack([fine, fun(fine, 1.), fun(fine, sag)])
        if name == "si":
            angle = np.linspace(pdf_sep, pdf_uep, 2049)
            pp = np.interp(angle, fine_curve[:, 0], fine_curve[:, 2])
            pdf_cct = float(np.trapezoid(1/(gain*(setpoint-pp)), angle))
            row["voltage_sag_sensitivity"] = [{"sag_pu": v, **scalar_bound(reduced_power, v, setpoint, gain, samples)}
                                              for v in extension["sag_amplitude_sensitivity_pu"]]
    return {"published_fig3": {"stable_angle_rad": pdf_sep, "first_unstable_angle_rad": pdf_uep,
                               "same_basin_boundary_s_from_drawn_curves": pdf_cct,
                               "maximum_drawn_sag_power_pu": float(np.max(paper_curve[:, 2]))},
            "models": model}


def classify(trace, init, ref, angle_tolerance):
    distance = trace[-1, 1] - init.delta
    turns = round(distance / (2*np.pi))
    error = abs(distance - turns*2*np.pi)
    settled = error <= angle_tolerance and abs(trace[-1, 2]-ref) < .005 and not bool(trace[-1, 8])
    return (turns if settled else None), error


def digital_record(trace, idx, dt, init, ref, tolerance):
    angles = trace[:, 1]
    offset, residual = classify(trace, init, ref, tolerance)
    sag = trace[:idx]
    post = trace[idx:]
    inactive = np.where(post[:, 8] == 0)[0]
    active = np.where(post[:, 8] == 1)[0]
    return {"duration_s": float(trace[idx, 0]), "clear_angle_rad": float(angles[idx]),
            "drift_to_clear_rad": float(angles[idx]-init.delta), "final_basin_offset_turns": offset,
            "final_angle_residual_rad": float(residual), "fault_mean_active_power_pu": float(np.mean(sag[:, 2])),
            "fault_current_peak_pu": float(np.max(sag[:, 4])), "fault_current_peak_after_initial_5ms_pu": float(np.max(sag[sag[:, 0] >= .005, 4])),
            "first_10ms_peak_sampled_current_pu": float(np.max(sag[sag[:, 0] < .010, 4])),
            "max_projected_reference_pu": float(np.max(trace[:, 7])),
            "recovery_current_peak_pu": float(np.max(post[:, 4])),
            "post_first_unclipped_sample_s": float(inactive[0]*dt) if inactive.size else None,
            "last_postfault_clipped_sample_s": float(active[-1]*dt) if active.size else None,
            "max_command_pu": float(np.max(trace[:, 9])), "max_limited_command_pu": float(np.max(trace[:, 11])), "pref_pu": ref}


def locate_transition(model, init, protocol, resolution, angle_tol):
    low = round(protocol["diagnostic_duration_s"]/model.dt)
    high = round(protocol["paper_test_s"][0]["duration_s"]/model.dt)
    trials = []

    def evaluate(ticks):
        duration = ticks*model.dt
        trace, _, diverged = model.simulate(init, duration, protocol["postfault_stop_s"], protocol["fault_grid_voltage_pu"])
        turns, error = classify(trace, init, model.p_ref, angle_tol)
        if diverged or turns is None:
            trace, _, diverged = model.simulate(init, duration, 2*protocol["postfault_stop_s"], protocol["fault_grid_voltage_pu"])
            turns, error = classify(trace, init, model.p_ref, angle_tol)
        if diverged or turns not in (0, 1):
            raise RuntimeError(f"The local clearing bracket is unclassified at {duration:g} s")
        trials.append({"duration_s": duration, "offset_turns": turns, "residual_rad": error})
        return turns

    if evaluate(low) != 0 or evaluate(high) != 1:
        raise RuntimeError("The nominal digital cases no longer bracket one recovery transition")
    while (high-low)*model.dt > resolution:
        midpoint = (high+low)//2
        if evaluate(midpoint) == 0:
            low = midpoint
        else:
            high = midpoint
    return {"observed_local_boundary_s": [low*model.dt, high*model.dt], "tested_clearing_times": trials}


def digital_studies(protocol, extension, electrical, arrays):
    table = protocol["table_i"]
    output = {}
    for scenario in extension["sampled_runs"]:
        key = scenario["key"]
        print(f"Sampled circuit {key}", flush=True)
        model = DigitalGFM(electrical[scenario["electrical_units"]], table, extension["filter_table_i"], scenario)
        init = model.equilibrium()
        residual, modes = model.linearize(init)
        dominant = max(modes, key=abs)
        radius = abs(dominant)
        data = {"scenario": scenario, "prefault_fixed_point_angle_rad": init.delta, "prefault_fixed_point_residual": residual,
                "prefault_max_discrete_pole_modulus": float(radius),
                "prefault_dominant_discrete_pole": [float(dominant.real), float(dominant.imag)], "checks": model.plant.check(), "tests": {}}
        if np.isfinite(model.bridge_limit):
            data["bridge_limit_pu"] = float(model.bridge_limit)
        if radius >= 1:
            data["state"] = "prefault_locally_unstable"
        else:
            data["state"] = "events_simulated"
            for published in protocol["paper_test_s"]:
                duration = published["duration_s"]
                out, idx, diverged = model.simulate(init, duration, protocol["postfault_stop_s"], protocol["fault_grid_voltage_pu"])
                if diverged:
                    raise RuntimeError(f"The healthy pre-fault case diverged for {key}, {duration:g} s")
                name = f"fig{published['figure']}"
                data["tests"][name] = digital_record(out, idx, model.dt, init, model.p_ref, extension["settled_angle_tolerance_rad"])
                arrays[f"digital_{key}_{name}"] = out
            if key == "si_nominal":
                data["clearing_boundary"] = locate_transition(model, init, protocol, extension["same_basin_limit_resolution_s"],
                                                               extension["settled_angle_tolerance_rad"])
        output[key] = data
    return output


def evaluate_scope(readback_meta, author_traces, protocol, extension, electrical, arrays):
    reference = protocol["table_i"]["active_reference_pu"]
    k = electrical.omega*protocol["table_i"]["power_droop_pu"]
    results = {}
    for figure, row in readback_meta["scope"].items():
        t, p, theta, voltage = author_traces[f"fig{figure}"].T
        during = t >= 0
        theta = np.unwrap(theta)
        angle_from_power = np.zeros_like(theta)
        angle_from_power[during] = k*cumulative_trapezoid(reference - p[during], t[during], initial=0)
        duration = row["sag_duration_from_image_s"]
        sag = (0 <= t) & (t <= duration)
        clear_read = (duration-.020 <= t) & (t <= duration-.006)
        after = (t >= duration+.01) & (t <= duration+.10)
        uncertainty = extension["readback_uncertainty"]["relative_trace_px"]
        results[figure] = {"measured_duration_s": duration, "mean_fault_power_pu": float(np.trapezoid(p[sag], t[sag])/(t[sag][-1]-t[sag][0])),
                           "power_pixel_uncertainty_pu": uncertainty*row["power_pu_per_px"],
                           "estimated_preclear_angle_change_rad": float(np.median(theta[clear_read])),
                           "integrated_preclear_angle_change_rad": float(np.median(angle_from_power[clear_read])),
                           "measured_p_integral_at_clear_rad": float(angle_from_power[np.where(sag)[0][-1]]),
                           "fault_phase_rmse_with_recorded_power_rad": float(np.sqrt(np.mean((theta[sag]-angle_from_power[sag])**2))),
                           "angle_pixel_uncertainty_rad": uncertainty*row["angle_rad_per_px"],
                           "post_fault_first_100ms_power_95th_percentile_pu": float(np.quantile(p[after], .95)),
                           "ending_scope_unwrapped_angle_turns": float(np.median(theta[t >= t[-1]-.15])/(2*np.pi))}
        arrays[f"scope_fig{figure}_phase_replay"] = np.column_stack((t, p, theta, angle_from_power, voltage))
    return results


def checks(metrics, protocol, extension, baseline):
    phasors, digital, scope = (metrics[key] for key in ("phasor", "sampled", "figure_readback"))
    si = phasors["models"]["si"]
    a = si["without_capacitor"]
    digitized = phasors["published_fig3"]
    tol = extension["verification_algebraic_pu"]
    results = {"archived_scalar_cct_matches_independent_inverse": abs(a["same_basin_boundary_s"] - baseline["models"]["si_derived"]["same_basin_clearing_boundary_s"]) < 1e-7,
               "fig3_si_power_curve_both_levels": max(si["fig3_pdf_global_curve_max_error_pu"]) < .001,
               "fig3_equilibrium_angles": max(abs(a["stable_angle_rad"]-digitized["stable_angle_rad"]),
                                               abs(a["first_unstable_angle_rad"]-digitized["first_unstable_angle_rad"])) < .0001,
               "fig3_readback_clearing_integral": abs(digitized["same_basin_boundary_s_from_drawn_curves"] - a["same_basin_boundary_s"]) < .001,
               "fig5_phase_integral_from_visible_power": scope["5"]["fault_phase_rmse_with_recorded_power_rad"] < .015,
               "fig6_phase_integral_within_scope_resolution": scope["6"]["fault_phase_rmse_with_recorded_power_rad"] < scope["6"]["angle_pixel_uncertainty_rad"],
               "fig5_scope_returns_without_phase_wrap": abs(scope["5"]["ending_scope_unwrapped_angle_turns"]) < .025,
               "fig6_scope_unwraps_once": abs(scope["6"]["ending_scope_unwrapped_angle_turns"] - 1) < .05}
    for name, row in phasors["models"].items():
        results[f"{name}_full_kcl_at_both_grid_voltages"] = row["capacitor_full_kcl_error_pu"] < tol
        results[f"{name}_zero_capacitor_matches_original_equations"] = row["capacitor_removal_power_error_pu"] < tol
    for name, row in digital.items():
        results[f"{name}_exact_hold_matches_independent_ode"] = row["checks"]["held_sample_independent_ode_error_pu"] < extension["verification_sample_integration_atol_pu"]
        results[f"{name}_electrical_energy_balance"] = row["checks"]["instantaneous_energy_balance_error_pu"] < tol
        results[f"{name}_prefault_fixed_point"] = row["prefault_fixed_point_residual"] < tol
        if row["state"] == "events_simulated":
            for figure, outcome in row["tests"].items():
                results[f"{name}_{figure}_reference_limit"] = outcome["max_projected_reference_pu"] <= protocol["table_i"]["current_limit_pu"] + tol
                results[f"{name}_{figure}_settled_classification"] = (outcome["final_basin_offset_turns"] is not None)
                if "bridge_limit_pu" in row:
                    results[f"{name}_{figure}_bridge_voltage_limit"] = outcome["max_limited_command_pu"] <= row["bridge_limit_pu"] + tol
    for test in protocol["paper_test_s"]:
        figure = test["figure"]
        results[f"fig{figure}_visible_duration_within_4ms"] = abs(scope[figure]["measured_duration_s"]-test["duration_s"]) <= .004 + 1e-12
    width = np.ptp(digital["si_nominal"]["clearing_boundary"]["observed_local_boundary_s"])
    results["nominal_sampled_transition_bracket_resolved"] = width <= extension["same_basin_limit_resolution_s"]
    return {k:bool(v) for k,v in results.items()}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--reference", type=Path, default=ROOT / "results", help="Directory containing the two PDF readback artifacts")
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True, parents=True)
    original_path, extension_path = ROOT / "configs/protocol.json", ROOT / "verification/extended_protocol.json"
    protocol, extension = load_json(original_path), load_json(extension_path)
    baseline_path = ROOT / "results/run_manifest.json"
    baseline = load_json(baseline_path)
    reference_meta, reference_npz = (args.reference / f"published_readback.{ext}" for ext in ("json", "npz"))
    meta = load_json(reference_meta)
    started = datetime.now(timezone.utc).isoformat()
    start = perf_counter()
    electrical = get_electrical(protocol["table_i"], extension["filter_table_i"])
    arrays = {}
    with np.load(reference_npz) as readback:
        metrics = {"doi": protocol["doi"], "phasor": phasor_studies(protocol, extension, electrical, readback, arrays),
                   "sampled": digital_studies(protocol, extension, electrical, arrays),
                   "figure_readback": evaluate_scope(meta, readback, protocol, extension, electrical["si"], arrays),
                   "sampled_trace_columns": DigitalGFM.columns,
                   "readback_phase_replay_columns": ["time_s", "power_pu", "image_angle_rad", "integrated_from_power_rad", "relative_capacitor_v_pu"]}
        metrics["checks"] = checks(metrics, protocol, extension, baseline)
        filenames = render(metrics, arrays, readback, args.output)
    write_json(args.output / "extended_metrics.json", metrics)
    np.savez_compressed(args.output / "extended_traces.npz", **arrays)
    exposed_files = ["extended_metrics.json", "extended_traces.npz", *filenames]
    checked = sum(metrics["checks"].values())
    manifest = {"doi": protocol["doi"], "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
                "wall_seconds": perf_counter()-start, "checks_passed": checked, "checks_total": len(metrics["checks"]),
                "environment": {"python": platform.python_version(), "platform": platform.platform(),
                                **{pkg: version(pkg) for pkg in ("numpy", "scipy", "matplotlib")}},
                "input_sha256": {x.name: digest(x) for x in (original_path, extension_path, baseline_path, reference_meta, reference_npz)},
                "implementation_sha256": {name: digest(ROOT / "src" / name) for name in (
                    "run_extended.py", "phasor_circuit.py", "sampled_circuit.py", "plot_extended.py", "reproduce.py")},
                "file_sha256": {name: digest(args.output / name) for name in exposed_files}}
    write_json(args.output / "extended_manifest.json", manifest)
    print(f"Extended checks {checked}/{manifest['checks_total']}, {manifest['wall_seconds']:.1f} seconds", flush=True)
    failed = [key for key, passed in metrics["checks"].items() if not passed]
    if failed:
        print("Failed checks", failed, flush=True)
    return int(bool(failed))


if __name__ == "__main__":
    raise SystemExit(main())
