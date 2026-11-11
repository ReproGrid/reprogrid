"""Verify an exact same-platform replay, including the numeric archives."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path

import numpy as np


def file_matches(path, expected):
    return path.is_file() and sha256(path.read_bytes()).hexdigest() == expected


def compare(primary, replay, primary_figures, replay_figures):
    manifests = [json.loads((path / "run_manifest.json").read_text()) for path in (primary, replay)]
    first, second = manifests
    result = {
        "protocol_sha256_equal": first["protocol_sha256"] == second["protocol_sha256"],
        "implementation_sha256_equal": first["implementation_sha256"] == second["implementation_sha256"],
        "checks_passed_in_both_runs": all(m["checks_passed"] == m["checks_total"] for m in manifests),
        "data_hashes": {},
        "figure_hashes": {},
        "array_equality": {},
    }
    for target, key, left, right in (
        ("data_hashes", "data_sha256", primary, replay),
        ("figure_hashes", "figures_sha256", primary_figures, replay_figures),
    ):
        for name in sorted(set(first[key]) | set(second[key])):
            digest = first[key].get(name)
            result[target][name] = bool(digest and digest == second[key].get(name)
                                        and file_matches(left / name, digest) and file_matches(right / name, digest))
    for name in ("impedance_curves.npz", "example_waveforms.npz"):
        with np.load(primary / name) as left, np.load(replay / name) as right:
            result["array_equality"][name] = {
                key: key in left and key in right and np.array_equal(left[key], right[key], equal_nan=True)
                for key in sorted(set(left.files) | set(right.files))
            }
    result["passed"] = (
        result["protocol_sha256_equal"] and result["implementation_sha256_equal"]
        and result["checks_passed_in_both_runs"]
        and all(result["data_hashes"].values()) and all(result["figure_hashes"].values())
        and all(all(checks.values()) for checks in result["array_equality"].values())
    )
    return result


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--primary-figures", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.primary, args.replay, args.primary_figures or args.primary / "figures",
                     args.replay / "figures")
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Exact replay {'passed' if result['passed'] else 'failed'}: {args.out}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
