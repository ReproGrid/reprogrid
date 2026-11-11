"""Compare the frequency and step data from a fresh Python environment."""
from hashlib import sha256
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def main():
    result = ROOT / "results"
    replay = result / "clean_replay"
    first = json.loads((result / "run_manifest.json").read_text())
    second = json.loads((replay / "run_manifest.json").read_text())
    config = json.loads((ROOT / "configs/protocol.json").read_text())
    keys = ("source_sha256", "protocol_sha256", "implementation_sha256", "packages", "python", "platform")
    pins_match = all(first[k] == second[k] for k in keys)
    arrays = {}
    for name in ("frequency_responses.npz", "local_steps.npz", "fig5_curve_inferred_local_steps.npz"):
        with np.load(result/name, allow_pickle=False) as a, np.load(replay/name, allow_pickle=False) as b:
            if set(a.files) != set(b.files) or any(a[k].shape != b[k].shape for k in a.files):
                raise ValueError(f"Replay schema differs in {name}")
            arrays[name] = {k: float(np.max(np.abs(a[k]-b[k]))) for k in a.files}
    maximum = max(v for rows in arrays.values() for v in rows.values())
    csv_match = {name: (result/name).read_bytes() == (replay/name).read_bytes()
                 for name in ("appendix_audit.csv", "local_steps.csv", "reactive_sweep.csv", "fig5_condition_comparison.csv")}
    sign_name = "fig5_condition_comparison.json"
    sign_first, sign_second = (result/sign_name).read_bytes(), (replay/sign_name).read_bytes()
    fig8_name = "fig8_independent_comparison.json"
    fig8_first, fig8_second = (result/fig8_name).read_bytes(), (replay/fig8_name).read_bytes()
    figure_match = {name: (ROOT/"figures"/name).read_bytes() == (replay/"figures"/name).read_bytes()
                    for name in ("fig5_parameter_overlay.png", "fig5_robustness_comparison.png")}
    tolerance = config["numerics"]["clean_replay_tolerance"]
    comparison = {"same_inputs_and_pins": pins_match, "array_max_difference": maximum, "array_differences": arrays,
                  "csv_byte_equal": csv_match, "fig5_comparison_json_byte_equal": sign_first == sign_second,
                  "fig8_independent_json_byte_equal": fig8_first == fig8_second,
                  "fig5_comparison_figure_byte_equal": figure_match, "tolerance": tolerance,
                  "replay_manifest_sha256": sha256((replay/"run_manifest.json").read_bytes()).hexdigest(),
                  "primary_wall_seconds": first["wall_seconds"], "replay_wall_seconds": second["wall_seconds"],
                  "passed": pins_match and maximum <= tolerance and all(csv_match.values()) and
                  sign_first == sign_second and fig8_first == fig8_second and all(figure_match.values()) and
                  json.loads(sign_first)["passed"] and json.loads(sign_second)["passed"] and
                  json.loads(fig8_first)["passed"] and json.loads(fig8_second)["passed"] and
                  first["numerical_checks_passed"] and second["numerical_checks_passed"]}
    (result/"clean_replay_comparison.json").write_text(json.dumps(comparison, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k: comparison[k] for k in ("passed", "array_max_difference", "csv_byte_equal",
                                                "fig5_comparison_json_byte_equal", "fig5_comparison_figure_byte_equal",
                                                "fig8_independent_json_byte_equal")}))
    return 0 if comparison["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
