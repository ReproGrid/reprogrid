"""Generate evidence for the no-load specialization of Harnefors et al., Eq. (19)."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent


def rk4(derivative, state, step):
    def advance(slopes, dt):
        return tuple(value + dt * slope for value, slope in zip(state, slopes))

    k1 = derivative(state)
    k2 = derivative(advance(k1, step / 2))
    k3 = derivative(advance(k2, step / 2))
    k4 = derivative(advance(k3, step))
    return tuple(
        value + step * (a + 2 * b + 2 * c + d) / 6
        for value, a, b, c, d in zip(state, k1, k2, k3, k4)
    )


def response(config, inductance, use_feedforward, dt, formulation):
    """Integrate a unit-normalized response in time τ = ω_base t."""
    voltage = config["voltage_pu"]
    resistance = config["active_resistance_pu"]
    amplitude = config["reference_step_pu"]
    alpha = resistance / inductance
    feedforward = int(use_feedforward)

    if formulation == "dq":
        # Eqs. (2), (9), (10), (13) at i₀ = 0 and Vg = V.
        def derivative(state):
            current_d, current_q, angle = state
            return (
                current_q - alpha * current_d + feedforward * alpha * amplitude / voltage,
                -current_d - alpha * current_q + voltage * angle / inductance,
                resistance / voltage**2 * (amplitude - voltage * current_d),
            )

        def output(state):
            return voltage * state[0] / amplitude
    else:
        # Companion-form integration of the published closed-loop polynomial.
        def derivative(state):
            x0, x1, x2 = state
            return x1, x2, 1 - alpha * x0 - (alpha**2 + 1) * x1 - 2 * alpha * x2

        def output(state):
            x0, x1, x2 = state
            return alpha * x0 + feedforward * (alpha**2 * x1 + alpha * x2)

    count = round(config["normalized_duration"] / dt)
    state = (0.0, 0.0, 0.0)
    samples = [output(state)]
    for _ in range(count):
        state = rk4(derivative, state, dt)
        samples.append(output(state))
    return samples


def crossing(samples, level, interval_ms):
    for index, value in enumerate(samples[1:], 1):
        if value >= level:
            previous = samples[index - 1]
            return (index - 1 + (level - previous) / (value - previous)) * interval_ms
    raise ValueError(f"Response never crossed {level}")


def metrics(samples, interval_ms):
    return {
        "rise_time_ms": round(crossing(samples, 0.9, interval_ms) - crossing(samples, 0.1, interval_ms), 5),
        "overshoot_percent": round(max(0, max(samples) - 1) * 100, 5),
    }


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "results")
    options = parser.parse_args()
    config_path = HERE / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    out = options.out
    out.mkdir(parents=True, exist_ok=True)
    dt = config["normalized_time_step"]
    milliseconds_per_tau = 1000 / (2 * math.pi * config["base_frequency_hz"])
    thresholds = {"dq_vs_transfer": 1e-9, "time_step_refinement": 1e-7, "rfpsc_vs_analytic": 1e-7}
    summary = {"doi": config["doi"], "kind": "independent_small_signal_implementation", "cases": []}
    chart = {"cases": {}}

    with (out / "step-response.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["case", "time_ms", "reference_step_pu", "psc_response_pu", "rfpsc_response_pu"])
        for case in config["cases"]:
            inductance = case["inductance_pu"]
            alpha = config["active_resistance_pu"] / inductance
            traces = {}
            checks = dict.fromkeys(thresholds, 0.0)

            for label, enabled in (("psc", False), ("rfpsc", True)):
                nominal = response(config, inductance, enabled, dt, "dq")
                fine = response(config, inductance, enabled, dt / 2, "dq")
                transfer = response(config, inductance, enabled, dt / 2, "transfer")
                traces[label] = fine
                checks["dq_vs_transfer"] = max(checks["dq_vs_transfer"], max(abs(a - b) for a, b in zip(fine, transfer)))
                checks["time_step_refinement"] = max(checks["time_step_refinement"], max(abs(a - b) for a, b in zip(nominal, fine[::2])))
                if enabled:
                    checks["rfpsc_vs_analytic"] = max(
                        abs(value + math.expm1(-alpha * index * dt / 2))
                        for index, value in enumerate(fine)
                    )

            if any(value > thresholds[key] for key, value in checks.items()):
                raise RuntimeError(f"{case['id']} validation failed: {checks}")
            interval_ms = dt / 2 * milliseconds_per_tau
            summary["cases"].append({
                **case, "alpha_pu": round(alpha, 8), "check_error_upper_bounds": {key: math.ceil(value * 1e12) / 1e12 for key, value in checks.items()},
                "psc": metrics(traces["psc"], interval_ms), "rfpsc": metrics(traces["rfpsc"], interval_ms),
            })
            amplitude = config["reference_step_pu"]
            points = []
            for index in range(0, len(traces["psc"]), 10):
                time = index * interval_ms
                psc = traces["psc"][index]
                rfpsc = traces["rfpsc"][index]
                writer.writerow([case["id"], f"{time:.8f}", amplitude, f"{amplitude * psc:.10f}", f"{amplitude * rfpsc:.10f}"])
                if time <= 85:
                    points.append([round(time, 5), round(psc, 7), round(rfpsc, 7)])
            chart["cases"][case["id"]] = points

    save_json(out / "summary.json", summary)
    save_json(out / "chart.json", chart)
    save_json(out / "manifest.json", {
        "doi": config["doi"], "normalization": "tau = 2*pi*50*t; response metrics normalized by step amplitude",
        "method": "fixed-step RK4; direct dq model, published transfer function, and first-order analytic solution",
        "inputs_sha256": {"run.py": digest(Path(__file__)), "config.json": digest(config_path)},
        "outputs_sha256": {name: digest(out / name) for name in ["summary.json", "chart.json", "step-response.csv"]},
        "thresholds": thresholds, "initial_state": "no load, zero perturbation",
    })
    for case in summary["cases"]:
        print(case["id"], case["psc"], case["rfpsc"])


if __name__ == "__main__":
    main()
