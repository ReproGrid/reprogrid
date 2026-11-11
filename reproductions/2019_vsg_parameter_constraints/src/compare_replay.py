"""Verify file hashes and numeric arrays across an exact clean-environment replay."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path

import numpy as np


def hashes_match(path_a, path_b, manifest_a, manifest_b):
    def matches(path, expected):
        return path.is_file() and sha256(path.read_bytes()).hexdigest() == expected

    return {name: (manifest_a.get(name) == manifest_b.get(name)
                   and matches(path_a / name, manifest_a.get(name))
                   and matches(path_b / name, manifest_b.get(name)))
            for name in sorted(set(manifest_a) | set(manifest_b))}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--primary-figures", type=Path)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    a, b = [json.loads((root / "run_manifest.json").read_text()) for root in (args.primary, args.replay)]
    data = hashes_match(args.primary, args.replay, a["data_sha256"], b["data_sha256"])
    figures = hashes_match(args.primary_figures or args.primary / "figures", args.replay / "figures",
                           a["figures_sha256"], b["figures_sha256"])
    arrays = {}
    for name in data:
        if name.endswith(".npz"):
            with np.load(args.primary / name) as left, np.load(args.replay / name) as right:
                arrays[name] = {key: key in left and key in right and np.array_equal(left[key], right[key], equal_nan=True)
                                for key in sorted(set(left.files) | set(right.files))}
    passed = (a["protocol_sha256"] == b["protocol_sha256"]
              and a["implementation_sha256"] == b["implementation_sha256"]
              and all(x["checks_passed"] == x["checks_total"] for x in (a, b))
              and all(data.values()) and all(figures.values())
              and all(all(group.values()) for group in arrays.values()))
    args.out.write_text(json.dumps({"passed": passed, "data_sha256_match": data, "figures_sha256_match": figures,
                                    "arrays_exact": arrays}, indent=2) + "\n")
    print(f"Replay {'passed' if passed else 'failed'}. {args.out}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
