"""Run the numerical reproduction, figures and deterministic result hashes."""

from argparse import ArgumentParser
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np

from plots import render
from study import calculate


ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT/"results")
    parser.add_argument("--reference", type=Path, default=ROOT/"results/published_curves.json")
    parser.add_argument("--figures", type=Path)
    args = parser.parse_args()
    out = args.out
    figures = args.figures or (ROOT/"figures" if out==ROOT/"results" else out/"figures")
    out.mkdir(exist_ok=True,parents=True)
    figures.mkdir(exist_ok=True,parents=True)
    started = datetime.now(timezone.utc).isoformat()
    start = perf_counter()
    config_file = ROOT/"configs/protocol.json"
    config = json.loads(config_file.read_text())
    original = json.loads(args.reference.read_text())
    source = json.loads((ROOT/"sources/source_manifest.json").read_text())
    if original["pdf_sha256"] != source["source"]["sha256"]:
        raise ValueError("Published figure coordinates must come from the declared PDF.")
    metrics,traces = calculate(config,original)
    solve_seconds = perf_counter()-start
    write_json(out/"metrics.json",metrics)
    write_json(out/"published_curves.json",original)
    np.savez_compressed(out/"numerical_traces.npz",**traces)
    image_files = render(metrics,traces,config,figures)
    files = ("metrics.json","published_curves.json","numerical_traces.npz")
    checks = metrics["checks"]
    modules = ("model.py","study.py","plots.py","reproduce.py")
    manifest = {"doi":config["doi"],"started_utc":started,"finished_utc":datetime.now(timezone.utc).isoformat(),
                "wall_seconds":perf_counter()-start,"stage_seconds":{"numerical":solve_seconds},
                "environment":{"python":platform.python_version(),"platform":platform.platform(),
                               **{p:version(p) for p in ("numpy","scipy","matplotlib")}},
                "protocol_sha256":digest(config_file),
                "implementation_sha256":{name:digest(ROOT/"src"/name) for name in modules},
                "data_sha256":{name:digest(out/name) for name in files},
                "figures_sha256":{name:digest(figures/name) for name in image_files},
                "checks_passed":sum(checks.values()),"checks_total":len(checks)}
    write_json(out/"run_manifest.json",manifest)
    failed = [name for name,ok in checks.items() if not ok]
    print(f"{manifest['checks_passed']}/{manifest['checks_total']} numerical checks in {manifest['wall_seconds']:.2f} s",flush=True)
    if failed:
        print("Failed checks",failed,flush=True)
    return int(bool(failed))


if __name__ == "__main__":
    raise SystemExit(main())
