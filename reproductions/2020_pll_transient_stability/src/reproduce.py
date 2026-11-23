"""Reproduce the PLL phase portraits, critical damping and balanced-fault replays."""

from argparse import ArgumentParser
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np

from model import SAMPLE_COLUMNS
from plots import render
from studies import build_grid, study_critical, study_phases, study_samples, study_units


ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def verify(metrics):
    c, p, s = (metrics[k] for k in ("critical_damping", "phase_portraits", "sampled_replays"))
    results = {"stable_manifold_refines": abs(c["zeta"]-c["zeta_finer_seed"]) < 5e-6,
               "stable_manifold_bracket_matches_forward_trajectories": [b["forward_outcome"] for b in c["forward_boundary_checks"][:2]] == ["crosses_saddle","converges"],
               "published_fig9_readback_matches_quoted_threshold": abs(c["article_vector_zeta"]-c["article_text_zeta"]) < .016,
               "fig13_damping_case_i_consistent_with_forward_critical_calculation": (
                   p["0.14"]["cases"]["case_i"]["converges_to_hyperbolic_equilibrium"] == (c["forward_boundary_checks"][-1]["forward_outcome"]=="converges")),
               "unique_equilibrium_at_published_deep_sag": p["0.10"]["equilibrium"]["count"] == 1,
               "fold_p_mode_agrees_with_exact_cotangent_solution": p["0.10"]["cases"]["case_iii_p_mode"]["independent_closed_form_max_angle_error_rad"] < 2e-8}
    for voltage, record in p.items():
        for case, row in record["cases"].items():
            results[f"{voltage}_{case}_independent_formulation"] = row["reformulations_agree_on_crossing_minus_pi"] and row["independent_swing_sequence_abs_error"] < 2e-6
        for name, row in s["reported_cases"][voltage].items():
            results[f"{voltage}_{name}_frequency_guard_consistent"] = row["fault_minimum_frequency_hz"] >= 45.-1e-10 and 45.-1e-10 <= row["final_postfault_frequency_hz"] <= 55.+1e-10
    for name, refined in s["adaptive_refinements"].items():
        primary = s["reported_cases"]["0.10"][name]
        results[f"{name}_refinement_preserves_saddle_crossing"] = ((primary["first_crossing_of_edge_s"] is None) == (refined["first_crossing_of_edge_s"] is None))
        results[f"{name}_refinement_clearance_phase_within_0p02_rad"] = abs(primary["phase_at_clearing_rad"]-refined["phase_at_clearing_rad"]) < .02
    return {k: bool(v) for k,v in results.items()}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    parser.add_argument("--reference", type=Path, default=ROOT / "results/published_curves.json")
    parser.add_argument("--figures", type=Path)
    args = parser.parse_args()
    out = args.out
    figures = args.figures or (ROOT/"figures" if out==ROOT/"results" else out/"figures")
    out.mkdir(exist_ok=True,parents=True)
    figures.mkdir(exist_ok=True,parents=True)
    started = datetime.now(timezone.utc).isoformat()
    start = perf_counter()
    config_file = ROOT / "configs/protocol.json"
    config = json.loads(config_file.read_text())
    original = json.loads(args.reference.read_text())
    grid = build_grid(config)
    arrays, times = {}, {}
    metrics = {"doi":config["doi"], "prefault_stable_angle_rad":grid.initial_angle,
               "inductance_pu_s":grid.inductance_pu_s, "unbounded_phase_trace_columns":["s","rad","Hz deviation"]}
    for label, stage in (("critical_damping",lambda: study_critical(config,grid,original,arrays)),
                         ("phase_portraits",lambda: study_phases(config,grid,original,arrays)),
                         ("sampled_replays",lambda: study_samples(config,grid,arrays)),
                         ("rocof_units",lambda:study_units(config,grid))):
        now = perf_counter()
        print(label, flush=True)
        metrics[label] = stage()
        times[label] = perf_counter()-now
    metrics["sampled_trace_columns"] = SAMPLE_COLUMNS
    metrics["checks"] = verify(metrics)
    write_json(out/"metrics.json",metrics)
    write_json(out/"published_curves.json",original)
    np.savez_compressed(out/"numerical_traces.npz",**arrays)
    image_files = render(metrics,arrays,figures)
    files = ("metrics.json","published_curves.json","numerical_traces.npz")
    manifest = {"doi":config["doi"],"started_utc":started,"finished_utc":datetime.now(timezone.utc).isoformat(),
                "wall_seconds":perf_counter()-start,"stage_seconds":times,
                "environment":{"python":platform.python_version(),"platform":platform.platform(),
                               **{p:version(p) for p in ("numpy","scipy","matplotlib")}},
                "protocol_sha256":digest(config_file),
                "implementation_sha256":{name:digest(ROOT/"src"/name) for name in ("model.py","stability.py","studies.py","reproduce.py","plots.py")},
                "data_sha256":{name:digest(out/name) for name in files}, "figures_sha256":{name:digest(figures/name) for name in image_files},
                "checks_passed":sum(metrics["checks"].values()), "checks_total":len(metrics["checks"])}
    write_json(out/"run_manifest.json",manifest)
    failed = [k for k,ok in metrics["checks"].items() if not ok]
    print(f"Checks {manifest['checks_passed']}/{manifest['checks_total']}, {manifest['wall_seconds']:.1f} s",flush=True)
    if failed:
        print("Failed numerical checks",failed,flush=True)
    return int(bool(failed))


if __name__ == "__main__":
    raise SystemExit(main())
