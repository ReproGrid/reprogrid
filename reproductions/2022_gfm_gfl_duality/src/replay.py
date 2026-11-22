"""Independently simulate the GFL current loop, sampled PLL and switched ABC grid."""

from argparse import ArgumentParser
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile
import xml.etree.ElementTree as ET

import numpy as np
from scipy.io import loadmat
from scipy.optimize import brentq


HERE = Path(__file__).resolve().parent
COMMIT = "8f73752332d6c2144570ac101e556f1859c06930"
CASE = "SingleInverterCase_Simulation/Case2_SingleInverterInfiniteBus/"
RAW = f"https://raw.githubusercontent.com/Future-Power-Networks/Simplus-Grid-Tool/{COMMIT}/{CASE}"
ARCHIVE_HASH = "6553b973850f0c909a9888c7dc7c0ef62329797c022fd72230e7f600d845f945"
MODEL_HASH = "4ecf2088c71498f1bf44e5a77967a07512e299dbd3a3ee0e85a880ae57c9eab1"
TS = 1e-4
BASE_HZ = 50
W = 2 * np.pi * BASE_HZ
LF, CF = 0.05 / W, 0.02 / W
LG, LS = (0.4 + 1e-9) / W, 0.2 / W
RG, RS = 0.08 + 1e-9, 0.04
RON, SNUBBER = 0.01, 1e6
RF_SLX = 1e-9
IREF = 0.5
KP_I = 2 * np.pi * 250 * LF
KI_I = (2 * np.pi * 250)**2 * LF / 4
KP_PLL = 2 * np.pi * 15
KI_PLL = KP_PLL**2 / 4
ALPHA_PLL = 2 * np.pi * 1000 * TS
SCALE = np.sqrt(2 / 3)
PHASES = np.array([0.0, -2 * np.pi / 3, -4 * np.pi / 3])
START, OPEN, CLOSE, STOP = 18000, 20000, 22000, 24000

# Numerical comparison criteria scoped to this archive and this sampled model.
# No post hoc time shift, amplitude normalization or filtering is applied.
LIMITS = {"steady_dq_and_frequency_max_pu": 1e-4, "steady_abc_max_pu": 1e-4,
          "steady_phase_max_rad": 1e-4, "waveform_abc_dq_rms_pu": 0.01,
          "waveform_abc_dq_max_pu": 0.02, "frequency_rms_hz": 0.05,
          "frequency_max_hz": 0.15, "phase_max_rad": 0.02}


def park(angle):
    """Power-invariant transform. Rows are d and q, with a separate ABC plant."""
    shifts = np.asarray(angle)[..., None] + PHASES
    return SCALE * np.stack([np.cos(shifts), -np.sin(shifts)], axis=-2)


def wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def to_abc(phasor, angle):
    return SCALE * np.real(np.asarray(phasor)[..., None] * np.exp(1j * (np.asarray(angle)[..., None] + PHASES)))


def bypass_resistance(closed):
    return 1 / (1 / SNUBBER + (1 / RON if closed else 0))


def grid_angle(t):
    # The source angle has a trapezoidal integrator with Auto initialization
    # and a constant input, so its first output includes half a sample.
    return W * (t + TS / 2)


@dataclass
class OperatingPoint:
    v: float
    grid_current: complex
    extra_current: complex
    phase_offset: float
    command: complex


def steady_state(rf):
    # Use the exact frequency substitution for a trapezoidal RL/C update.
    w_discrete = 2 / TS * np.tan(W * TS / 2)
    z_extra = RS + 1j * w_discrete * LS
    r_bypass = bypass_resistance(True)
    z_parallel = r_bypass * z_extra / (r_bypass + z_extra)
    z_line = RG + 1j * w_discrete * LG + z_parallel

    def current(v):
        return IREF - 1j * w_discrete * CF * v

    def source(v):
        return v - z_line * current(v)

    v = brentq(lambda v: abs(source(v)) - 1, 0.8, 1.2, xtol=1e-14)
    actual_voltage = v + (rf + 1j * w_discrete * LF) * IREF
    # The controller leads the applied stationary voltage by its one-tick delay.
    command = actual_voltage * np.exp(1j * W * TS)
    return OperatingPoint(v, current(v), current(v) * z_parallel / z_extra,
                          -np.angle(source(v)), command)


def plant_step_matrix(rf, was_closed, is_closed, transition_rule):
    """Discretize the KCL/KVL equations for iL, vC, iGrid and iExtra."""
    def stationary_matrix(closed):
        r_bypass = bypass_resistance(closed)
        return np.array([[-rf / LF, -1 / LF, 0, 0],
                         [1 / CF, 0, -1 / CF, 0],
                         [0, 1 / LG, -(RG + r_bypass) / LG, r_bypass / LG],
                         [0, 0, r_bypass / LS, -(r_bypass + RS) / LS]])

    inputs = np.array([[1 / LF, 0], [0, 0], [0, -1 / LG], [0, 0]])
    identity = np.eye(4)
    old = stationary_matrix(was_closed)
    new = stationary_matrix(is_closed)
    if was_closed != is_closed and transition_rule == "two_half_backward_euler":
        half = TS / 2
        step = np.linalg.solve(identity - half * new, identity)
        gain = step @ (half * inputs)
        return np.column_stack([step @ step, step @ gain / 2, gain + step @ gain / 2])
    if was_closed != is_closed and transition_rule == "backward_euler":
        # A one-step backward Euler sensitivity check. The archived TBE
        # solver's exact switch-step update is not specified by this equation.
        return np.linalg.solve(identity - TS * new,
                               np.column_stack([identity, np.zeros_like(inputs), TS * inputs]))
    half = TS / 2
    return np.linalg.solve(identity - half * new,
                           np.column_stack([identity + half * old, half * inputs, half * inputs]))


def simulate(rf, transition_rule="trapezoid"):
    """The simulation uses parameters and commands only, with no archived signal inputs."""
    time = np.arange(START, STOP + 1) * TS
    source = to_abc(1, grid_angle(time))
    point = steady_state(rf)
    initial_angle = grid_angle(time[0]) + point.phase_offset
    initial_phasors = [IREF, point.v, point.grid_current, point.extra_current]
    electrical = to_abc(initial_phasors, initial_angle)
    output = np.empty((len(time), 4, 3))
    angles = np.empty(len(time))
    frequency = np.empty(len(time))
    output[0] = electrical
    current_integral = np.array([point.command.real, point.command.imag])
    phase, pll_integral, filtered_q = initial_angle, 0.0, 0.0
    closing = np.ones(3, dtype=bool)
    matrix = {(old, new): plant_step_matrix(rf, old, new, transition_rule)
              for old in (False, True) for new in (False, True)}
    last_inputs = np.stack([to_abc(point.command, initial_angle - W * TS), source[0]])
    physical_events = []
    for k in range(len(time)):
        if k:
            inputs = np.stack([command_abc, source[k]])
            terms = np.vstack([electrical, last_inputs, inputs])
            predicted = np.column_stack([matrix[old, old] @ terms[:, j] for j, old in enumerate(closing)])
            command_closed = closing.copy()
            tick = START + k
            # The close command arrives at the start of this interval.
            # Its first effect is in the sample after the command.
            if tick == CLOSE + 1:
                command_closed[:] = True
            if OPEN <= tick < CLOSE:
                old_breaker_current = electrical[2] - electrical[3]
                predicted_breaker_current = predicted[2] - predicted[3]
                command_closed[old_breaker_current * predicted_breaker_current <= 0] = False
            for j in np.flatnonzero(closing != command_closed):
                predicted[:, j] = matrix[closing[j], command_closed[j]] @ terms[:, j]
                physical_events.append({"time_s": float(time[k]), "phase": "ABC"[j], "closed": bool(command_closed[j])})
            electrical = predicted
            last_inputs, closing = inputs, command_closed
            output[k] = electrical

        angles[k] = phase % (2 * np.pi)
        transform = park(angles[k])
        current_error = np.array([IREF, 0]) - transform @ electrical[0]
        command_abc = transform.T @ (current_integral + KP_I * current_error)
        current_integral += TS * KI_I * current_error

        measured_q = transform[1] @ electrical[1]
        speed = W + KP_PLL * filtered_q + pll_integral
        phase += TS * speed
        pll_integral += TS * KI_PLL * filtered_q
        filtered_q = (1 - ALPHA_PLL) * filtered_q + ALPHA_PLL * measured_q
        frequency[k] = speed / (2 * np.pi)

    result = {"time": time, "phase_rad": angles, "frequency_hz": frequency}
    for idx, name in enumerate(("il", "vc", "ig")):
        result[name + "abc"] = output[:, idx]
        result[name + "dq"] = np.einsum("tij,tj->ti", park(angles), output[:, idx])
    return result, physical_events


def load_reference(payload, name="Case2_GFL_Y", digest=ARCHIVE_HASH):
    if sha256(payload).hexdigest() != digest:
        raise ValueError("Author MAT does not match the pinned SHA-256.")
    trace = loadmat(BytesIO(payload), simplify_cells=True)[name]
    time = trace["time"]
    assert time.shape == (30001,) and np.max(abs(time - np.arange(30001) * TS)) < 1e-12
    channels = {entry["label"].strip("<>"): entry["values"] for entry in trace["signals"]}
    return {"time": time, "phase_rad": channels.pop("theta"),
            "frequency_hz": channels.pop("w") * BASE_HZ, **channels}


def audit_model(payload):
    if sha256(payload).hexdigest() != MODEL_HASH:
        raise ValueError("Author SLX does not match the pinned SHA-256.")
    with ZipFile(BytesIO(payload)) as slx:
        systems = {name: ET.fromstring(slx.read(f"simulink/systems/system_{name}.xml"))
                   for name in ("root", "2374", "2368", "2387", "2455")}
        metadata = ET.fromstring(slx.read("metadata/coreProperties.xml"))
        modified = next(item.text for item in metadata if item.tag.endswith("}modified"))

    def fields(system, sid, path="P"):
        return {entry.get("Name"): entry.text for entry in systems[system].find(f'Block[@SID="{sid}"]').findall(path)}

    def mask(sid, case="2374"):
        block = systems[case].find(f'Block[@SID="{sid}"]')
        return {entry.get("Name"): entry.findtext("Value") for entry in block.findall("Mask/MaskParameter")}

    circuit, controller = mask("2374:2759"), mask("2374:2087")
    source_integrator = fields("2368", "2368:806")
    breaker = fields("root", "2349", "InstanceData/P")
    powergui = fields("root", "8", "InstanceData/P")
    assert float(circuit["Rf"]) == RF_SLX and float(circuit["Rc"]) == 1e-9
    assert circuit["Rd"] == "1e-20" and controller["Kcd"] == "0"
    assert controller["Kp"] == "w_i*Lf*(Vbase^2/Sbase)/Wbase"
    assert controller["Ki"] == "w_i*w_i/4*Lf*(Vbase^2/Sbase)/Wbase"
    assert source_integrator["IntegratorMethod"] == "Integration: Trapezoidal" and source_integrator["InitialConditionSetting"] == "Auto"
    assert float(breaker["BreakerResistance"]) == RON and float(breaker["SnubberResistance"]) == SNUBBER
    assert breaker["SnubberCapacitance"] == "inf" and breaker["External"] == "on"
    assert fields("2374", "2374:2751")["SampleTime"] == "-1"
    gain_masks = {case: mask(case + ":2759", case)["Rf"] for case in ("2387", "2455")}
    assert all(value == "Lf/5" for value in gain_masks.values())
    return {"rf_mask_ohm": float(circuit["Rf"]), "gain_case_rf_mask_expressions": gain_masks, "gain_case_rf_resolved_ohm_at_base_impedance_1": 0.01,
            "slx_modified_timestamp": modified, "current_controller_mask": controller,
            "grid_angle_integrator": source_integrator["IntegratorMethod"],
            "grid_angle_initial_condition_setting": source_integrator["InitialConditionSetting"],
            "powergui_solver": powergui["SolverType"], "powergui_sample_time": powergui["SampleTime"]}


def infer_resistance(reference, current_scale=None, train=slice(20100, 21999)):
    """Diagnose R from KVL and logged control input; never feed logs to simulate()."""
    il, vc, ig = (reference[key] for key in ("ilabc", "vcabc", "igabc"))
    error = np.array([IREF, 0]) - reference["ildq"]
    phase = reference["phase_rad"]
    transform = park(phase)
    scale = np.ones((len(error), 1)) if current_scale is None else np.asarray(current_scale)[:, None]
    integral_input = KI_I * TS * error * scale**2
    pi = KP_I * error * scale + np.vstack([np.zeros(2), np.cumsum(integral_input[:-1], axis=0)])
    command = np.einsum("tij,ti->tj", transform, pi)

    def mean_endpoints(x):
        return (x[1:] + x[:-1]) / 2

    def delayed(x):
        return np.concatenate([np.zeros_like(x[:1]), x[:-1]])

    target = LF * np.diff(il, axis=0) / TS + mean_endpoints(vc) - mean_endpoints(delayed(command))
    columns = np.stack([mean_endpoints(delayed(transform[:, 0])),
                        mean_endpoints(delayed(transform[:, 1])), -mean_endpoints(il)], axis=-1)
    cap_residual = CF * np.diff(vc, axis=0) / TS - mean_endpoints(il - ig)
    held_out = slice(22004, 24000)
    # These windows are outside the short switch-step solver transitions.
    assert np.max(abs(cap_residual[train])) < 1e-9 and np.max(abs(cap_residual[held_out])) < 1e-9
    fit, _, rank, _ = np.linalg.lstsq(columns[train].reshape(-1, 3), target[train].ravel(), rcond=None)
    if rank != 3:
        raise ValueError("The selected first-event window cannot identify the two PI offsets and resistance.")
    return {"effective_resistance_ohm": float(fit[2]), "integrator_initial_dq_fitted": fit[:2].tolist(),
            "training_line_intervals_s": [train.start * TS, train.stop * TS], "held_out_line_intervals_s": [2.2004, 2.4],
            "training_kvl_max_error_pu": float(np.max(abs(columns[train] @ fit - target[train]))),
            "held_out_kvl_max_error_pu": float(np.max(abs(columns[held_out] @ fit - target[held_out]))),
            "training_design_condition_number": float(np.linalg.cond(columns[train].reshape(-1, 3)))}


def compare(model, reference):
    reference = {name: values[START:STOP + 1] for name, values in reference.items()}
    channels = ("vcabc", "ilabc", "igabc", "vcdq", "ildq", "igdq", "frequency_hz", "phase_rad")
    errors = {name: model[name] - reference[name] for name in channels}
    errors["phase_rad"] = wrap(errors["phase_rad"])
    steady = slice(0, 1000)
    steady_max = {name: float(np.max(abs(error[steady]))) for name, error in errors.items()}
    mean_dq = {name: {"model": model[name][steady].mean(axis=0).tolist(),
                      "reference": reference[name][steady].mean(axis=0).tolist()}
               for name in ("vcdq", "ildq", "igdq")}
    steady_passes = all(value <= (LIMITS["steady_phase_max_rad"] if name == "phase_rad" else
                                  BASE_HZ * LIMITS["steady_dq_and_frequency_max_pu"] if name == "frequency_hz" else
                                  LIMITS["steady_abc_max_pu"] if name.endswith("abc") else
                                  LIMITS["steady_dq_and_frequency_max_pu"])
                        for name, value in steady_max.items())
    events = []
    for title, lo, hi in [("opening", OPEN - START, CLOSE - START), ("reclosing", CLOSE - START, STOP - START + 1)]:
        metrics = {}
        for name, error in errors.items():
            rms, peak = float(np.sqrt(np.mean(error[lo:hi]**2))), float(np.max(abs(error[lo:hi])))
            if name == "frequency_hz":
                passed = rms <= LIMITS["frequency_rms_hz"] and peak <= LIMITS["frequency_max_hz"]
            elif name == "phase_rad":
                passed = peak <= LIMITS["phase_max_rad"]
            else:
                passed = rms <= LIMITS["waveform_abc_dq_rms_pu"] and peak <= LIMITS["waveform_abc_dq_max_pu"]
            metrics[name] = {"rms_error": rms, "max_abs_error": peak, "passed": passed}
        events.append({"event": title, "samples": hi - lo, "channels": metrics,
                       "passed": all(item["passed"] for item in metrics.values())})
    return {"steady_state": {"means_dq_pu": mean_dq, "max_abs_errors": steady_max, "passed": steady_passes},
            "transients": events, "passed": steady_passes and all(row["passed"] for row in events)}


def audit_other_archives(source_dir):
    result = {}
    cases = {
        "Case2_GFL_w_pll": "b1923bf49541acd61264914251a1ef4bfb2f3d891518167e10e9e8423743473a",
        "Case2_GFL_w_i": "94335fcdd621d3cfe424059ab9db1cf53303b312c2bf2b1d313ed85689dcc3f8",
    }
    for name, digest in cases.items():
        data = read_source(name + ".mat", source_dir)
        trace = load_reference(data, name, digest)
        # Source scales current Kp by 0.6 and Ki by 0.6² only in the w_i case.
        ticks = np.rint(trace["time"] / TS).astype(int)
        scale = np.ones(len(ticks))
        if name.endswith("_w_i"):
            scale[(ticks >= OPEN) & (ticks < CLOSE)] = 0.6
        result[name] = {"source_sha256": digest, **infer_resistance(trace, scale, slice(1000, 10000))}
    return result


def read_source(name, source_dir):
    return (source_dir / name).read_bytes() if source_dir else urlopen(RAW + name, timeout=25).read()


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("replay-output"))
    parser.add_argument("--no-figure", action="store_true")
    args = parser.parse_args()
    reference = load_reference(read_source("Case2_GFL_Y.mat", args.source_dir))
    source_audit = audit_model(read_source("Case2_Sim.slx", args.source_dir))
    fit = infer_resistance(reference)
    archive_checks = audit_other_archives(args.source_dir)
    modes = ("trapezoid", "backward_euler", "two_half_backward_euler")
    candidates = [("source_mask", RF_SLX, "trapezoid")] + [
        (mode, 0.01, mode) for mode in modes
    ]
    comparison, models = {}, {}
    for name, resistance, transition_rule in candidates:
        model, switches = simulate(resistance, transition_rule)
        models[name] = model
        comparison[name] = {"resistance_ohm": resistance, "switches": switches, **compare(model, reference)}
    validated = all(comparison[mode]["passed"] for mode in modes) and all(
        abs(item["effective_resistance_ohm"] - 0.01) < 1e-8
        and item["held_out_kvl_max_error_pu"] < 1e-9
        for item in [fit, *archive_checks.values()]
    )
    result = {"paper_doi": "10.1109/TPWRS.2022.3151851", "reference_commit": COMMIT,
              "model_sha256": MODEL_HASH, "archive_sha256": ARCHIVE_HASH,
              "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
              "validation_scope": "Independent single-inverter ABC electrical model and controllers, 1.8–2.4 s, with 2.0 s opening and 2.2 s reclosing commands.",
              "limits": LIMITS, "source_audit": source_audit, "resistance_diagnostic": fit, "comparison": comparison,
              "source_parameter_validation_passed": comparison["source_mask"]["passed"],
              "archive_cross_checks": archive_checks, "conditional_validation_passed": validated}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    if not args.no_figure:
        from plot import make_figure
        make_figure(models, comparison, args.output_dir)
    print(json.dumps({name: {"steady": values["steady_state"]["passed"],
                             "transients": {event["event"]: event["passed"] for event in values["transients"]}}
                      for name, values in comparison.items()}, indent=2))
    return 0 if validated else 1


if __name__ == "__main__":
    raise SystemExit(main())
