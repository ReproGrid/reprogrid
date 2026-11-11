"""Compare regenerated data against the primary run using the frozen tolerance."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, default=ROOT / "results/clean_replay")
    args = parser.parse_args()
    primary = ROOT / "results"
    protocol = json.loads((ROOT / "configs/protocol.json").read_text())
    tolerance = protocol["numerics"]["clean_replay_array_tolerance"]
    original = json.loads((primary / "run_manifest.json").read_text())
    replay = json.loads((args.replay / "run_manifest.json").read_text())
    keys = ("source_sha256", "protocol_sha256", "implementation_sha256", "packages", "python", "platform")
    inputs_equal = all(original[k] == replay[k] for k in keys)
    arrays = {}
    for branch in original["branches"].values():
        for case in branch["cases"].values():
            name = case["npz"]
            with np.load(primary / name, allow_pickle=False) as a, np.load(args.replay / name, allow_pickle=False) as b:
                if set(a.files) != set(b.files):
                    raise ValueError(f"Different fields in {name}")
                if any(a[k].shape != b[k].shape for k in a.files):
                    raise ValueError(f"Different array shapes in {name}")
                arrays[name] = {k: float(np.max(np.abs(a[k] - b[k]))) for k in a.files}
    max_difference = max(v for dataset in arrays.values() for v in dataset.values())
    result = {"same_inputs_and_pins": inputs_equal, "replay_manifest_sha256": sha256(
              (args.replay / "run_manifest.json").read_bytes()).hexdigest(),
              "primary_wall_seconds": original["wall_seconds"], "replay_wall_seconds": replay["wall_seconds"],
              "array_max_difference": max_difference, "array_differences": arrays,
              "tolerance": tolerance, "cases": len(arrays),
              "passed": inputs_equal and max_difference <= tolerance and
              original["numerical_checks_passed"] and replay["numerical_checks_passed"]}
    (primary / "clean_replay_comparison.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("cases", "passed", "array_max_difference")}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
