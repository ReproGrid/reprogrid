"""Compare an isolated-environment replay with the recorded primary run."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", type=Path, default=ROOT / "results/clean_replay")
    args = parser.parse_args()
    original = ROOT / "results"
    replay = args.replay.resolve()
    manifests = [json.loads((p / "run_manifest.json").read_text()) for p in (original, replay)]
    tolerance = 1e-12  # Same code, parameters, architecture and pinned dependencies.
    rows = []
    for case in manifests[0]["cases"]:
        name = case["data_file"]
        with np.load(original / name, allow_pickle=False) as a, np.load(replay / name, allow_pickle=False) as b:
            if set(a.files) != set(b.files):
                raise RuntimeError(f"Array names differ: {name}")
            errors = {key: float(np.max(np.abs(a[key] - b[key]))) for key in a.files}
            rows.append({"case_id": case["case_id"], "array_max_abs_errors": errors,
                         "passed": all(error <= tolerance for error in errors.values())})
    same_source = all(manifests[0][k] == manifests[1][k] for k in
                      ("source_sha256", "protocol_sha256", "implementation_sha256", "packages"))
    result = {
        "scope": "Fresh virtualenv; same host and pinned packages. Not a cross-platform replay.",
        "fresh_environment": str(ROOT.parents[1] / ".cache/clean-rfpsc-venv"),
        "input_and_dependency_versions_match": same_source,
        "numeric_tolerance": tolerance,
        "cases": rows,
        "both_numerical_suites_passed": all(m["all_numerical_checks_passed"] for m in manifests),
        "replay_manifest_sha256": hashlib.sha256((replay / "run_manifest.json").read_bytes()).hexdigest(),
        "primary_manifest_sha256": hashlib.sha256((original / "run_manifest.json").read_bytes()).hexdigest(),
    }
    result["passed"] = same_source and result["both_numerical_suites_passed"] and all(r["passed"] for r in rows)
    (original / "clean_replay_comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"passed": result["passed"], "max_difference": max(max(r["array_max_abs_errors"].values()) for r in rows)}))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
