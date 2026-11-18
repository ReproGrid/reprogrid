"""Run the TPEL voltage-interpolation and three-phase harmonic comparison."""

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter

import matplotlib
import numpy as np
import scipy

from circuit import Circuit, PHASES, terminal_command
from plots import make_figures
from pwm import triangular_pwm
from reference import ac_response, measured_spectrum, nodal_spectrum, observables
from solvers import exact_switching, fixed_step


ROOT = Path(__file__).resolve().parents[1]
DOI = "10.1109/TPEL.2021.3117633"


def write_json(path, content):
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def circuit_from_protocol(config):
    p = config["cired_2019_tables_1_2"]
    a = config["assumptions"]
    if p["delta_delta_turns_ratio"] != 1:
        raise ValueError("This referred circuit expects a unity transformer ratio.")
    return Circuit(
        p["filter_inductance_h"], p["filter_capacitance_f"], p["filter_damping_ohm"],
        p["line_inverter_side_inductance_h"] + p["transformer_leakage_h"],
        p["line_grid_side_inductance_h"], p["line_midpoint_capacitance_f"],
        a["filter_inductor_series_ohm"], a["line_inverter_side_series_ohm"],
        a["line_grid_side_series_ohm"],
    )


def rms_bins(coefficients):
    return float(np.sqrt(2.0 * np.sum(np.abs(coefficients) ** 2)))


def band_summary(measured, expected):
    error = np.linalg.norm(measured - expected)
    amplitude = rms_bins(measured)
    exact = rms_bins(expected)
    return {
        "rms": amplitude,
        "reference_rms": exact,
        "amplitude_error_pct": 100 * (amplitude - exact) / exact if exact > 1e-12 else None,
        "complex_error_pct": float(100 * error / np.linalg.norm(expected)) if exact > 1e-12 else None,
    }


def carrier_band(carrier, grid_hz, halfwidth):
    return np.arange(carrier - halfwidth, carrier + halfwidth + 1, grid_hz)


def row_summary(wave, expected, frequencies, fundamental):
    actual = observables(measured_spectrum(wave, fundamental, frequencies))
    ref = observables(expected)
    return {k: band_summary(value, ref[k]) for k, value in actual.items()}


def named_wave(waves, key, wave):
    waves[key + "_time_s"] = wave.time
    waves.update({key + "_" + signal: value for signal, value in wave.signals.items()})
    if wave.gate is not None:
        waves[key + "_pole_duty"] = wave.gate


def add_check(checks, label, value, allowance):
    checks[label] = {"value": float(value), "max_allowed": float(allowance),
                     "passed": bool(np.isfinite(value) and value <= allowance)}


def frequency_response(circuit, sweep, nominal_pwm, carrier_hz, fundamental):
    on = np.ones((len(sweep), 3), dtype=complex)
    ans, _ = ac_response(circuit, sweep, on, np.zeros_like(on))
    kind = {"v_bridge_to_v_line": abs(ans["v_line"][:, 0]),
            "v_line_over_v_filter": abs(ans["v_line"][:, 0] / ans["v_filter"][:, 0]),
            "v_bridge_to_i_cable": abs(ans["i_cable"][:, 0])}
    center = nominal_pwm.fourier_voltage(np.array([carrier_hz]), differential=False)[0]
    neighbors = nominal_pwm.fourier_voltage(np.array([carrier_hz - 2 * fundamental,
                                                      carrier_hz + 2 * fundamental]))
    return kind, {"center_pole_rms_v": float(np.sqrt(2.0) * abs(center[0])),
                  "center_differential_rms_v": float(np.sqrt(2.0) * abs(center[0] - center[1])),
                  "neighbor_differential_rms_v": (np.sqrt(2.0) * abs(neighbors[:, 0] - neighbors[:, 1])).tolist()}


def trapezoidal_frequency(frequency, step):
    return np.tan(np.pi * step * frequency) / (np.pi * step)


def interpolation_error_budget(circuit, wave, expected, band, grid_hz, carrier, phase_band_order):
    """Separate PWM representation from trapezoidal frequency warping."""
    step = wave.time[1] - wave.time[0]
    measured = measured_spectrum(wave, grid_hz, band)
    zero = np.zeros_like(expected["u_bridge"])
    warped = trapezoidal_frequency(band, step)
    bridge = measured["u_bridge"]
    ideal_bridge = expected["u_bridge"]
    input_only, _ = ac_response(circuit, band, bridge, zero)
    integrator_only, _ = ac_response(circuit, warped, ideal_bridge, zero)
    together, _ = ac_response(circuit, warped, bridge, zero)
    readback = observables(measured)
    ref = observables(expected)
    out = {label: {key: band_summary(series[key], ref[key]) for key in ("v_line_ab", "i_cable_a")}
           for label, series in (("changed_source_only", observables(input_only)),
                                 ("warping_only", observables(integrator_only)),
                                 ("both", observables(together)))}
    predicted = observables(together)
    mismatch = max(np.linalg.norm(readback[k] - predicted[k]) / np.linalg.norm(ref[k])
                   for k in ("v_line_ab", "i_cable_a"))
    for tone in (carrier - phase_band_order * grid_hz, carrier + phase_band_order * grid_hz):
        idx = np.flatnonzero(band == tone)[0]
        out.setdefault("sidebands", []).append({
            "frequency_hz": int(tone), "integrator_effective_frequency_hz": float(warped[idx]),
            "exact_v_line_ab_rms_v": float(np.sqrt(2) * abs(ref["v_line_ab"][idx])),
            "vi10_v_line_ab_rms_v": float(np.sqrt(2) * abs(readback["v_line_ab"][idx])),
        })
    return out, mismatch


def fundamental_power(spectrum, grid_phase_peak):
    grid = grid_phase_peak * np.exp(1j * PHASES) / 2
    power = 2 * (grid @ spectrum["i_grid"][0].conj())
    return {"p_w": float(power.real), "q_var": float(power.imag)}


def experiment(config):
    p = config["cired_2019_tables_1_2"]
    a = config["analysis"]
    conditions = config["paper_conditions"]
    f = p["grid_frequency_hz"]
    carrier = conditions["carrier_hz"]
    probe = a["diagnostic_carrier_hz"]
    circuit = circuit_from_protocol(config)
    no_cable = replace(circuit, cl=0)
    command, grid = terminal_command(circuit, p["grid_line_to_line_rms_v"], f, p["published_output_w"])
    pwms = {freq: triangular_pwm(f, freq, p["dc_v"], command)
            for freq in a["carrier_hz_sweep"]}
    pwm = pwms[carrier]
    dt_exact = a["flat_wave_reference_step_us"] * 1e-6
    exact_grid = np.arange(round(pwm.period / dt_exact)) * dt_exact
    bands = {freq: carrier_band(freq, f, a["carrier_band_half_width_hz"])
             for freq in (carrier, probe)}

    def calculate(plant, switch, model, step_us, phase=0, fifth=0, solver="trapezoidal"):
        return fixed_step(plant, switch, model, step_us * 1e-6, f, grid, fifth, command, phase, solver)

    def spectral(plant, switch, freqs, fifth=0):
        return nodal_spectrum(plant, switch, freqs, f, grid, fifth)

    refs = {}
    checks = {}
    results = {}
    waves = {}
    run_times = {}
    error_budgets = {}
    principal = (("sw2", "SW", conditions["switching_step_us"]),
                 ("sw10", "SW", conditions["interpolation_step_us"]),
                 ("vi10", "VI", conditions["interpolation_step_us"]),
                 ("av100", "AV", conditions["average_step_us"]))
    for label, plant, freq in (("cable_4500", circuit, carrier), ("cable_6000", circuit, probe),
                               ("no_cable_6000", no_cable, probe)):
        switch = pwms[freq]
        freq_band = bands[freq]
        ref, mismatch = spectral(plant, switch, freq_band)
        refs[label] = ref
        add_check(checks, label + "_independent_kcl_a", mismatch, a["independent_kcl_allowance_a"])
        wave = exact_switching(plant, switch, exact_grid, f, grid, 0)
        if label != "no_cable_6000":
            named_wave(waves, label + "_event", wave)
        exact_summary = row_summary(wave, ref, freq_band, f)
        for obs in ("v_line_ab", "i_filter_a", "i_cable_a", "i_grid_a"):
            complex_error = exact_summary[obs]["complex_error_pct"]
            if complex_error is not None:
                add_check(checks, label + "_" + obs + "_event_matches_nodal_pct",
                          complex_error, 100 * a["relative_reference_dft_allowance"])
        variants = {}
        fundamental, _ = spectral(plant, switch, np.array([f]))
        for name, model, step in principal:
            start = perf_counter()
            sample = calculate(plant, switch, model, step)
            run_times[label + "_" + name] = perf_counter() - start
            if label != "no_cable_6000":
                named_wave(waves, label + "_" + name, sample)
            if np.max(freq_band) < f * len(sample.time) / 2:
                metrics = row_summary(sample, ref, freq_band, f)
            else:
                metrics = None
            if name == "vi10" and plant.cl > 0:
                budget, error = interpolation_error_budget(
                    plant, sample, ref, freq_band, f, freq, a["sideband_order"])
                error_budgets[label] = budget
                add_check(checks, label + "_tr_frequency_domain_agrees_with_trajectory_relative", error, 1e-9)
            variants[name] = {"step_us": step, "steps_per_period": sample.switching_steps,
                              "band": metrics,
                              "band_outside_discrete_nyquist": metrics is None,
                              "grid_fundamental_power": fundamental_power(
                                  measured_spectrum(sample, f, np.array([f])), grid)}
        results[label] = {
            "carrier_hz": freq,
            "band_hz": [int(freq_band[0]), int(freq_band[-1])],
            "reference": exact_summary,
            "reference_grid_fundamental_power": fundamental_power(fundamental, grid),
            "models": variants,
        }

    # Independent centered aperture integral matches each Eq. (1) endpoint,
    # including the coarse-step cases with more than one edge near a node.
    sweep = []
    max_node_error = 0.0
    max_area_error = 0.0
    min_gate, max_gate = 1.0, 0.0
    for label, freq in (("cable_4500", carrier), ("cable_6000", probe)):
        switch = pwms[freq]
        for step_us in a["steps_us"]:
            step = step_us * 1e-6
            for phase in a["step_phase_fraction"]:
                vi = calculate(circuit, switch, "VI", step_us, phase)
                sw = calculate(circuit, switch, "SW", step_us, phase)
                aperture = (switch.integral_gate(vi.time + step / 2)
                            - switch.integral_gate(vi.time - step / 2)) / step
                max_node_error = max(max_node_error, float(np.max(abs(aperture - vi.gate))))
                global_area = step * np.sum(vi.gate, axis=0)
                area_error = p["dc_v"] * np.max(abs(global_area - switch.integrated_gate[-1]))
                max_area_error = max(max_area_error, float(area_error))
                min_gate = min(min_gate, float(np.min(vi.gate)))
                max_gate = max(max_gate, float(np.max(vi.gate)))
                sweep.append({
                    "carrier_hz": freq, "step_us": step_us, "step_phase_fraction": phase,
                    "SW": row_summary(sw, refs[label], bands[freq], f),
                    "VI": row_summary(vi, refs[label], bands[freq], f),
                })
    add_check(checks, "interpolated_nodes_vs_exact_centered_area_max", max_node_error, 1e-9)
    add_check(checks, "maximum_leg_cycle_voltage_area_error_vs", max_area_error,
              a["exact_voltage_area_allowance_vs"])
    add_check(checks, "vi_gate_range_violation", max(-min_gate, max_gate - 1.0, 0.0), 1e-12)

    solver_results = {}
    for label, freq in (("cable_4500", carrier), ("cable_6000", probe)):
        solver_results[label] = {}
        for name, model, step in principal:
            if model == "AV" and freq > 0.5e6 / step:
                continue
            sample = calculate(circuit, pwms[freq], model, step, solver="sdirk2")
            solver_results[label][name] = row_summary(sample, refs[label], bands[freq], f)

    # Difference the with/without fifth-harmonic cases with identical PWM.
    fifth_ratio = conditions["grid_fifth_relative_peak"]
    fifth_freq = np.array([5 * f])
    fifth_reference, _ = spectral(circuit, pwm, fifth_freq, fifth_ratio)
    clean_reference, _ = spectral(circuit, pwm, fifth_freq)
    fifth_exact = {k: v - clean_reference[k] for k, v in fifth_reference.items()}
    fifth_results = {}
    for name, model, step in principal:
        states = [calculate(circuit, pwm, model, step, fifth=h) for h in (0, fifth_ratio)]
        baseline, injected = [measured_spectrum(w, f, fifth_freq) for w in states]
        fifth_model = {k: v - baseline[k] for k, v in injected.items()}
        fifth_results[name] = {k: band_summary(v, observables(fifth_exact)[k])
                               for k, v in observables(fifth_model).items()}
        named_wave(waves, "grid_fifth_" + name, states[1])

    sweep_plant = []
    for freq, switch in pwms.items():
        pairs = np.array([freq - a["sideband_order"] * f, freq + a["sideband_order"] * f])
        for label, plant in (("cable", circuit), ("open", no_cable)):
            response, _ = spectral(plant, switch, pairs)
            value = observables(response)
            sweep_plant.append({
                "carrier_hz": freq, "line_cable": label,
                "v_line_ab_sideband_rms_v": rms_bins(value["v_line_ab"]),
                "v_filter_ab_sideband_rms_v": rms_bins(value["v_filter_ab"]),
                "i_cable_a_sideband_rms_a": rms_bins(value["i_cable_a"]),
                "v_line_over_v_filter": rms_bins(value["v_line_ab"]) / rms_bins(value["v_filter_ab"]),
            })

    sensitivity = []
    for c_multiplier in a["line_c_multiplier"]:
        for r_multiplier in a["resistance_multiplier"]:
            plant = replace(circuit, cl=circuit.cl * c_multiplier, rf=circuit.rf * r_multiplier,
                            r1=circuit.r1 * r_multiplier, r2=circuit.r2 * r_multiplier)
            for label, freq in (("cable_4500", carrier), ("cable_6000", probe)):
                frequency_band = bands[freq]
                ref, _ = spectral(plant, pwms[freq], frequency_band)
                models = {}
                for name, model, step in principal[:3]:
                    wave = calculate(plant, pwms[freq], model, step)
                    models[name] = row_summary(wave, ref, frequency_band, f)["v_line_ab"]
                sensitivity.append({
                    "carrier_hz": freq, "line_c_multiplier": c_multiplier,
                    "series_r_multiplier": r_multiplier, "v_line_ab_band_v": models,
                })

    frequency_grid = np.linspace(3000, 10000, 701)
    transfer, symmetry = frequency_response(circuit, frequency_grid, pwm, carrier, f)
    open_transfer, _ = frequency_response(no_cable, frequency_grid, pwm, carrier, f)
    tr_dt = conditions["interpolation_step_us"] * 1e-6
    shifted_hz = np.arctan(np.pi * tr_dt * frequency_grid) / (np.pi * tr_dt)
    analog_peak = int(frequency_grid[np.argmax(transfer["v_bridge_to_v_line"])])
    for key, value in transfer.items():
        waves["cable_transfer_" + key] = value
        waves["open_transfer_" + key] = open_transfer[key]
    waves["transfer_hz"] = frequency_grid
    waves["trapezoidal_transfer_hz"] = shifted_hz
    waves["pwm_edge_s"] = pwm.edges
    waves["pwm_edge_leg"] = pwm.legs
    waves["pwm_edge_direction"] = pwm.signs
    add_check(checks, "balanced_carrier_voltage_center_differential_rms_v",
              symmetry["center_differential_rms_v"], 1e-8)
    av = results["cable_4500"]["models"]["av100"]["band"]
    add_check(checks, "av_has_no_carrier_sideband_v_line_ab_rms", av["v_line_ab"]["rms"], 1e-8)
    # These are algebraic and cross-implementation checks. A model crossing
    # the engineering tolerance remains a measured result.
    step_allow = a["engineering_harmonic_amplitude_allowance_pct"]
    metrics = {
        "doi": DOI, "reference_circuit": "CIRED 2019 Figures 1–2 and Tables 1–2 (TPEL reference [3])",
        "command": {"phasor_peak_v": [float(command.real), float(command.imag)],
                    "max_modulation": float(2 * abs(command) / p["dc_v"]),
                    "grid_phase_peak_v": grid},
        "pwm_carrier_symmetry": symmetry,
        "harmonic_cases": results,
        "timestep_phase_sweep": sweep,
        "sdirk2_sensitivity": solver_results,
        "trapezoidal_error_budget": error_budgets,
        "frequency_response": {
            "bridge_to_line_peak_hz": analog_peak,
            "trapezoidal_10us_peak_hz": float(shifted_hz[np.argmax(transfer["v_bridge_to_v_line"])]),
        },
        "grid_fifth_delta": fifth_results,
        "carrier_sweep": sweep_plant,
        "parameter_sensitivity": sensitivity,
        "engineering_harmonic_amplitude_allowance_pct": step_allow,
        "checks": checks,
    }
    return metrics, waves, run_times


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/protocol.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--figures", type=Path)
    args = parser.parse_args()
    figures = args.figures or args.output / "figures"
    args.output.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    started, begin = utc_now(), perf_counter()
    config = json.loads(args.protocol.read_text())
    metrics, waves, run_times = experiment(config)
    write_json(args.output / "metrics.json", metrics)
    np.savez_compressed(args.output / "waveforms.npz", **waves)
    figure_files = make_figures(metrics, waves, figures)
    data_files = ("metrics.json", "waveforms.npz")
    write_json(args.output / "run_manifest.json", {
        "schema_version": 1, "doi": DOI, "scope": config["claim_scope"],
        "started_utc": started, "finished_utc": utc_now(), "elapsed_seconds": perf_counter() - begin,
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "numpy": np.__version__, "scipy": scipy.__version__, "matplotlib": matplotlib.__version__},
        "protocol_sha256": sha(args.protocol),
        "implementation_sha256": {p.name: sha(p) for p in sorted((ROOT / "src").glob("*.py"))
                                  if not p.name.startswith("._")},
        "data_sha256": {name: sha(args.output / name) for name in data_files},
        "figures_sha256": {p.name: sha(p) for p in figure_files},
        "checks_passed": sum(x["passed"] for x in metrics["checks"].values()),
        "checks_total": len(metrics["checks"]),
        "individual_fixed_step_runs_wall_seconds": run_times,
    })
    failures = [k for k, v in metrics["checks"].items() if not v["passed"]]
    print(f"{len(metrics['checks']) - len(failures)}/{len(metrics['checks'])} numerical checks passed.")
    if failures:
        print("Failing checks:", failures)
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
