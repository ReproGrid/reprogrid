"""Recheck the detector, observer and recorded pulse trajectories independently."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from scipy.linalg import expm
from scipy.optimize import brentq


ROOT = Path(__file__).resolve().parents[1]


def gain_coefficients(kind, damping, design_bandwidth):
    if kind == "pi":
        def half_power_error(ratio):
            numerator = ratio**2 + 2j * damping * ratio
            return abs(numerator / (numerator - 1))**2 - .5

        natural_frequency = design_bandwidth * brentq(half_power_error, 0., 1., xtol=1e-15)
        return np.array([2 * damping * natural_frequency, natural_frequency**2])

    natural_frequency = .25648 * design_bandwidth
    if kind == "eso_c1":
        multiplier = [2 * damping + 1, 2 * damping + 1, 1]
    else:
        multiplier = [3 * damping**2, 3 * damping, 1]
    return multiplier * natural_frequency ** np.arange(1, 4)


def compare_frequency_responses(paper, curve_path):
    errors = {}
    with np.load(curve_path) as curves:
        s = 1j * curves["omega_rad_s"]
        for kind in ("PI", "ESO_C1", "ESO_C2"):
            gain = gain_coefficients(kind.lower(), paper["fig8_damping"], paper["fig8_design_bandwidth_rad_s"])
            # Sum the injection through the individual integrator paths.
            loop = gain @ s[None, :] ** -np.arange(1, len(gain) + 1)[:, None]
            errors[kind] = float(np.max(abs(loop / (1 + loop) - curves[kind])))
    return errors


def observer_update(gain, period):
    order = len(gain)
    augmented = np.zeros((order + 1, order + 1))
    augmented[:order, :order] = np.eye(order, k=1)
    augmented[:order, order] = gain
    update = expm(period * augmented)
    return update[:order, :order], update[:order, order]


def axis_projectors(degrees):
    radians = np.radians(degrees)
    direction = np.column_stack((np.cos(radians), np.sin(radians)))
    # Form exact cardinal projectors for the deliberately exact lock cases.
    cardinal = degrees % 90 == 0
    direction[cardinal] = np.eye(2)[(degrees[cardinal] / 90).astype(int) % 2]
    d_axis = direction[:, :, None] * direction[:, None, :]
    return d_axis, np.eye(2) - d_axis


def apply_matrix(matrix, vector):
    return (matrix @ vector[..., None])[..., 0]


def replay(paper, settings, degrees, time, gain, seed, switched, scale):
    """Stationary-frame response using linear superposition of each PWM leg."""
    step = paper["sampling_period_s"]
    r = paper["standstill_resistance_ohm"]
    ld, lq = paper["d_inductance_h"], paper["q_inductance_h"]
    decay_rates = r / np.array([ld, lq])
    project_d, project_q = axis_projectors(degrees)
    poles = np.exp(-step * decay_rates)
    decay = poles[0] * project_d + poles[1] * project_q
    response = (np.eye(2) - decay) / r
    bus = settings["ideal_dc_bus_v"]
    leg_axes = np.array([[1., 0.], [-.5, np.sqrt(3) / 2], [-.5, -np.sqrt(3) / 2]])
    state_step, error_step = observer_update(gain, time[1] - time[0])

    def advance_current(current, voltage, rising):
        decayed = apply_matrix(decay, current)
        if not switched:
            return decayed + apply_matrix(response, voltage)
        references = voltage @ leg_axes.T
        midpoint = (references.min(axis=1) + references.max(axis=1)) / 2
        on_time = step * (.5 + (references - midpoint[:, None]) / bus)
        weights = -np.expm1(-on_time[:, :, None] * decay_rates) / r
        if rising:
            weights *= np.exp(-(step - on_time[:, :, None]) * decay_rates)
        summed_d = weights[:, :, 0] @ leg_axes
        summed_q = weights[:, :, 1] @ leg_axes
        return decayed + (2 * bus / 3) * (apply_matrix(project_d, summed_d) + apply_matrix(project_q, summed_q))

    current = np.zeros((len(degrees), 2))
    state = np.zeros((len(degrees), len(gain)))
    state[:, 1] = seed
    estimates = np.zeros((len(time), len(degrees)))
    detectors = np.empty((len(time) - 1, len(degrees)))
    for group in range(len(detectors)):
        estimate = state[:, 0]
        direct = np.column_stack((np.cos(estimate), np.sin(estimate)))
        quadrature = np.column_stack((-direct[:, 1], direct[:, 0]))
        voltage = settings["injection_vector_amplitude_v"] * direct
        before = apply_matrix(decay, current)
        positive = advance_current(before, voltage, group % 2 == 1)
        current = advance_current(positive, -voltage, group % 2 == 0)
        difference = (positive - before) - (current - positive)
        detectors[group] = (-scale * np.sqrt(2) * np.sum(difference * quadrature, axis=1)
                            / np.linalg.norm(difference, axis=1))
        correction = detectors[group] / (2 * (1 - ld / lq))
        state = state @ state_step.T - correction[:, None] * error_step
        estimates[group + 1] = state[:, 0]
    return estimates, detectors


def angular_limits(paper):
    ld, lq = paper["d_inductance_h"], paper["q_inductance_h"]
    limit = (1 - ld / lq) * np.sin(np.radians(2 * paper["convergence_nominal_electrical_deg"]))
    # Substitute y = sin²(error) and take the smaller root near the d axis.
    limits = []
    for multiplier in (2, 4):
        a = multiplier * (lq - ld)**2
        b = a + limit**2 * (lq**2 - ld**2)
        c = limit**2 * lq**2
        limits.append(float(np.degrees(np.arcsin(np.sqrt(2 * c / (b + np.sqrt(b**2 - 4 * a * c)))))))
    return float(limit), limits


def check_normalization(paper):
    ld, lq = paper["d_inductance_h"], paper["q_inductance_h"]
    error = np.linspace(-np.pi, np.pi, 1441)
    d_axis, q_axis = axis_projectors(-np.degrees(error))
    delta_i = (d_axis / ld + q_axis / lq)[:, :, 0]
    detector = -np.sqrt(2) * delta_i[:, 1] / np.linalg.norm(delta_i, axis=1)
    rhs = (lq - ld) * np.sin(2 * error) / np.hypot(lq * np.cos(error), ld * np.sin(error))
    return float(np.max(abs(np.sqrt(2) * detector - rhs)))


def first_window(valid, window, time):
    counts = np.vstack((np.zeros((1, valid.shape[1]), dtype=int), valid.cumsum(axis=0)))
    qualifies = counts[window:] - counts[:-window] == window
    return np.where(qualifies.any(axis=0), time[window + qualifies.argmax(axis=0)], np.nan)


def compare_windows(valid, window, time):
    counted = first_window(valid, window, time)
    elapsed = first_window(valid, window + 1, time)
    retained = np.isfinite(elapsed) & np.isfinite(counted)
    return counted, {
        "declarations_with_group_count": int(np.isfinite(counted).sum()),
        "declarations_with_sample_span": int(np.isfinite(elapsed).sum()),
        "max_delay_for_retained_declarations_s": float(np.max(elapsed[retained] - counted[retained], initial=0.)),
    }


def compare_traces(paper, settings, trace_path, threshold):
    runs = {}
    damping = settings["simulation_observer_damping"]
    ts = paper["sampling_period_s"]
    window = int(np.ceil(paper["convergence_hold_s"] / (3 * ts)))
    angles = {
        "averaged": np.arange(0., 360., settings["rotor_angle_increment_deg"]),
        "switched": np.arange(0., 360., settings["switched_angle_increment_deg"]),
        "zero_seed": np.arange(0., 360., 90.),
    }
    angles["scaled_sqrt2"] = angles["averaged"]
    expected_time = 3 * ts * np.arange(round(settings["duration_s"] / (3 * ts)) + 1)
    with np.load(trace_path) as traces:
        expected_names = {f"{method}_{setup}" for method in ("pi", "eso") for setup in angles}
        trace_names = {key.removesuffix("__estimate_rad") for key in traces.files if key.endswith("__estimate_rad")}
        if trace_names != expected_names:
            raise ValueError(f"Unexpected set of runs: {sorted(trace_names)}")
        for name in sorted(trace_names):
            method, setup = name.split("_", 1)
            kind = "pi" if method == "pi" else "eso_c2"
            design = paper["fig13_design_bandwidth_rad_s"][f"puvi_{kind}"]
            gain = gain_coefficients(kind, damping, design)
            degrees = traces[f"{name}__angle_deg"]
            time = traces[f"{name}__time_s"]
            if not np.array_equal(degrees, angles[setup]) or not np.allclose(time, expected_time, rtol=0, atol=1e-15):
                raise ValueError(f"Unexpected time or angle grid in {name}")
            estimate, detector = replay(paper, settings, degrees, time, gain,
                                        0. if setup == "zero_seed" else settings["initial_speed_seed_rad_s"],
                                        setup == "switched", np.sqrt(2) if setup == "scaled_sqrt2" else 1.)
            # Use a different angle wrap and rolling counts to check saved declaration times.
            axis_error = .5 * np.degrees(np.angle(np.exp(2j * (estimate - np.radians(degrees)))))
            accepted_detector = abs(detector) <= threshold
            accepted_angle = abs(axis_error) <= paper["convergence_nominal_electrical_deg"]
            flags, detector_windows = compare_windows(accepted_detector, window, time)
            joint, joint_windows = compare_windows(accepted_detector & accepted_angle[:-1] & accepted_angle[1:], window, time)
            runs[name] = {
                "angles": len(degrees),
                "max_estimate_difference_deg": float(np.degrees(np.max(abs(estimate - traces[f"{name}__estimate_rad"])))),
                "max_detector_difference": float(np.max(abs(detector - traces[f"{name}__raw_detector"]))),
                "detector_times_agree": bool(np.array_equal(flags, traces[f"{name}__flag_time_s"], equal_nan=True)),
                "joint_times_agree": bool(np.array_equal(joint, traces[f"{name}__joint_time_s"], equal_nan=True)),
                "detector_window_comparison": detector_windows,
                "joint_window_comparison": joint_windows,
            }
            if setup == "zero_seed":
                on_q_axis = degrees % 180 == 90
                runs[name]["exact_q_axis_false_declaration"] = bool(
                    np.allclose(abs(axis_error[:, on_q_axis]), 90, rtol=0, atol=1e-9)
                    and np.all(flags[on_q_axis] == time[window]) and np.isnan(joint[on_q_axis]).all()
                )
    return window, runs


def local_stability(paper, settings):
    result = {}
    damping = settings["simulation_observer_damping"]
    ratios = {"d_axis": 1 / np.sqrt(2), "q_axis": -paper["q_inductance_h"] / (np.sqrt(2) * paper["d_inductance_h"])}
    for name in ("pi", "eso_c2"):
        gain = gain_coefficients(name, damping, paper["fig13_design_bandwidth_rad_s"][f"puvi_{name}"])
        state, response = observer_update(gain, 3 * paper["sampling_period_s"])
        result[name] = {}
        for axis, ratio in ratios.items():
            jacobian = state.copy()
            jacobian[:, 0] -= ratio * response
            result[name][axis] = float(max(abs(np.linalg.eigvals(jacobian))))
    return result


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    paths = [ROOT / name for name in ("configs/protocol.json", "results/pulse_traces.npz",
                                      "results/observer_curves.npz", "results/metrics.json")]
    config_path, trace_path, curve_path, metrics_path = paths
    config = json.loads(config_path.read_text())
    metrics = json.loads(metrics_path.read_text())
    paper, settings = (config[key] for key in ("paper", "reconstruction_settings"))
    threshold, limits = angular_limits(paper)
    reference_limits = [metrics["normalization"][f"ideal_near_d_axis_limit_from_{side}_deg"] for side in ("left", "right")]
    factor_error = check_normalization(paper)
    frequency_errors = compare_frequency_responses(paper, curve_path)
    window, runs = compare_traces(paper, settings, trace_path, threshold)
    radii = local_stability(paper, settings)
    peak_error = max(r["max_estimate_difference_deg"] for r in runs.values())
    group_s = 3 * paper["sampling_period_s"]
    checks = {
        "factor_rederived_from_stationary_inductance": factor_error < 1e-11,
        "angular_limits_rederived": bool(np.allclose(limits, reference_limits, rtol=0, atol=1e-9)),
        "observer_frequency_responses_agree": max(frequency_errors.values()) < 1e-11,
        "independent_trajectories_agree": peak_error < 1e-8 and max(r["max_detector_difference"] for r in runs.values()) < 1e-10,
        "independent_flags_agree": all(r["joint_times_agree"] and r["detector_times_agree"] for r in runs.values()),
        "exact_q_axis_false_declarations_agree": all(runs[f"{kind}_zero_seed"]["exact_q_axis_false_declaration"] for kind in ("pi", "eso")),
        "q_equilibrium_is_unstable": all(r["d_axis"] < 1 < r["q_axis"] for r in radii.values()),
    }
    result = {
        "scope": "Analytical factor and threshold, observer frequency responses, stationary-frame trajectories and flag timestamps, lossless local stability",
        "code_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_sha256": {str(path.relative_to(ROOT)): sha256(path.read_bytes()).hexdigest() for path in paths},
        "checks": checks, "passed": all(checks.values()),
        "sqrt2_normalization_max_error": factor_error,
        "threshold_left_right_deg": limits,
        "frequency_response_max_errors": frequency_errors,
        "hold": {"complete_groups": window, "group_s": group_s, "counted_duration_s": window * group_s,
                 "elapsed_since_first_detector_sample_s": (window - 1) * group_s,
                 "min_samples_for_full_span": window + 1,
                 "first_possible_timestamp_for_full_span_s": (window + 1) * group_s},
        "lossless_local_spectral_radii": radii,
        "max_estimate_difference_deg": peak_error,
        "runs": runs,
    }
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Independent review {'passed' if result['passed'] else 'found differences'}. {args.out}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
