"""Check the two inertial Fig. 8 examples with an independent physical state model."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from verifier import dc_matrix, physical_model, response


ROOT = Path(__file__).resolve().parents[1]


def vsm_states(a, b, c, inertia, droop_gain, damping, alpha):
    """Augment the plant with the virtual speed and its low-pass copy."""
    n = len(b)
    dimension = n + 1 + int(damping != 0)
    closed = np.zeros((dimension, dimension))
    command = np.zeros(dimension)
    power = np.zeros(dimension)
    closed[:n, :n] = a
    closed[:n, n] = b
    closed[n, :n] = -droop_gain / inertia * c
    closed[n, n] = -(droop_gain + damping) / inertia
    command[n] = droop_gain / inertia
    power[:n] = c
    if damping:
        closed[n, n+1] = damping / inertia
        closed[n+1, n:n+2] = [alpha, -alpha]
    return closed, command, power


def compare_poles(matrix, reference):
    computed = np.linalg.eigvals(matrix)
    poles = np.array([complex(*point) for point in reference])
    if len(computed) != len(poles):
        raise ValueError("Independent state model and paper transfer function orders differ")
    distances = abs(computed[:, None] - poles)
    return {"max_pole_distance_pu": float(max(distances.min(axis=0).max(), distances.min(axis=1).max())),
            "rightmost_pole_real_pu": float(max(computed.real))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, default=ROOT / "results")
    parser.add_argument("--output", type=Path, default=ROOT / "results/fig8_independent_comparison.json")
    args = parser.parse_args()
    config = json.loads((ROOT / "configs/protocol.json").read_text())
    manifest = json.loads((args.primary / "run_manifest.json").read_text())
    p = config["fig7_8"]
    omega_base = 2 * np.pi * config["base_frequency_hz"]
    inertia = 2 * p["inertia_seconds"] * omega_base
    kg, kd, alpha = 1 / p["droop"], 1 / (4 * np.sqrt(2)), p["damping_filter_rad_s"] / omega_base
    a, b, c, *_ = physical_model(1 / p["scr"], p["id"], p["iq"], Kp=p["droop"], **config["nominal"])
    cases = {}
    with np.load(args.primary / "frequency_responses.npz", allow_pickle=False) as reference:
        w = reference["w_pu"]
        s = 1j * w
        plant = response(a, b, c, w)
        for key, damping in (("vsm_H5", 0.0), ("vsm_H5_KD50", p["damping_pu"])):
            inner, command, power = vsm_states(a, b, c, inertia, kg, damping, alpha)
            inner_open = plant * kg / (inertia * s + kg + damping * s / (s + alpha))
            inner_closed = response(inner, command, power, w)
            dc_open = kd * inner_closed / s
            errors = {}
            for name, values in (("gp", inner_open), ("gc", inner_closed), ("gd", dc_open)):
                target = reference[f"fig7_8_{key}_{name}"]
                errors[name] = float(max(abs(target - values) / np.maximum(1, abs(values))))
            cases[key] = {"scaled_frequency_errors": errors,
                          "active_closed_poles": compare_poles(inner, manifest["fig7_8"][key]["gp"]["closed_poles_pu"]),
                          "dc_closed_poles": compare_poles(dc_matrix(inner, command, power, kd),
                                                           manifest["fig7_8"][key]["gd"]["closed_poles_pu"])}
    tolerance = config["numerics"]["cross_frequency_abs_tolerance"]
    passed = all(max(case["scaled_frequency_errors"].values()) < tolerance and
                 max(case["active_closed_poles"]["max_pole_distance_pu"],
                     case["dc_closed_poles"]["max_pole_distance_pu"]) < tolerance for case in cases.values())
    report = {"doi": config["doi"], "implementation_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
              "method": "Independent fixed-grid-frame PSC state model with virtual-speed and damping-filter states from Eqs. (27)-(30).",
              "samples_per_response": len(w), "tolerance": tolerance, "cases": cases, "passed": passed}
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"fig8_independent_passed": passed}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
