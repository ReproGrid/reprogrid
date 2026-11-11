"""Verify numerical output from a clean environment and the pinned framework replay."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parents[1]


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+"\n")


def compare_independent(original, replay):
    main = read(original/"run_manifest.json")
    clean = read(replay/"run_manifest.json")
    config = read(HERE/"configs/protocol.json")
    identity = ("protocol_sha256", "implementation_sha256", "source_sha256", "python", "platform", "packages")
    pins_equal = all(main[k]==clean[k] for k in identity)
    data = {}
    names = [f"ideal_{name}.npz" for name in main["scenarios"]]+["ideal_local_steps.npz"]
    for name in names:
        with np.load(original/name,allow_pickle=False) as a,np.load(replay/name,allow_pickle=False) as b:
            if set(a.files)!=set(b.files) or any(a[k].shape!=b[k].shape for k in a.files):
                raise ValueError(f"Array keys or shapes differ in {name}")
            data[name] = {k:float(np.max(abs(a[k]-b[k]))) for k in a.files}
    maximum = max(v for row in data.values() for v in row.values())
    same_sweep = digest(original/"ideal_inductance_sweep.csv")==digest(replay/"ideal_inductance_sweep.csv")
    tol = config["numerics"]["replay_array_tolerance"]
    return {"passed":pins_equal and same_sweep and maximum<=tol and main["numerical_checks_passed"] and clean["numerical_checks_passed"],
            "same_inputs_and_pins":pins_equal,"array_max_difference":maximum,"array_difference_by_field":data,
            "sweep_csv_byte_equal":same_sweep,"tolerance":tol,"primary_wall_seconds":main["wall_seconds"],
            "replay_wall_seconds":clean["wall_seconds"],"replay_manifest_sha256":digest(replay/"run_manifest.json")}


def compare_framework(output, redigitized):
    original = HERE/"results"
    prior = read(original/"replay_verification.json")
    comparisons = {}
    for name, prior_record in prior["compared_files"].items():
        before,after = digest(original/name),digest(output/name)
        comparisons[name] = {"original_sha256":before,"replay_sha256":after,
                             "original_matches_prior_record":before==prior_record["sha256"],"byte_equal":before==after}
    metrics = read(output/"metrics.json")
    protocol_matches = metrics["source_hashes"]["protocol"]==digest(HERE/"protocol.json")
    script_matches = metrics["source_hashes"]["script"]==digest(HERE/"reproduce.py")
    ref = {}
    if redigitized:
        for name in ("fig5b_rfpsc.csv","fig6b_do_gfm.csv","paper_metrics.json"):
            before,after = digest(HERE/"reference"/name),digest(redigitized/name)
            ref[name] = {"original_sha256":before,"replay_sha256":after,"byte_equal":before==after}
    return {"passed":protocol_matches and script_matches and metrics["motulator_commit"]==prior["motulator_commit"] and
            all(r["byte_equal"] and r["original_matches_prior_record"] for r in comparisons.values()) and
            all(r["byte_equal"] for r in ref.values()) and metrics["all_solver_refinement_checks_pass"],
            "motulator_commit":metrics["motulator_commit"],"protocol_matches":protocol_matches,"script_matches":script_matches,
            "files":comparisons,"redigitized_files":ref}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay",type=Path,default=HERE/"results/clean_replay")
    parser.add_argument("--author-output",type=Path)
    parser.add_argument("--redigitized-output",type=Path)
    args = parser.parse_args()
    original = HERE/"results"
    record = compare_independent(original,args.replay)
    write(original/"clean_replay_comparison.json",record)
    result = {"main_replay":record["passed"],"max_array_difference":record["array_max_difference"]}
    if args.author_output:
        framework = compare_framework(args.author_output,args.redigitized_output)
        write(original/"author_replay_comparison.json",framework)
        result.update(author_replay=framework["passed"],author_byte_equal_count=sum(x["byte_equal"] for x in framework["files"].values()),
                      digitized_byte_equal_count=sum(x["byte_equal"] for x in framework["redigitized_files"].values()))
    print(json.dumps(result))
    return 0 if record["passed"] and result.get("author_replay",True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
