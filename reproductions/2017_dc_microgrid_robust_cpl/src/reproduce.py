"""Recreate the small-network case studies from the paper's topology and SI data."""

from argparse import ArgumentParser
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np

from network import Components
from plots import render
from studies import eight_bus_practice, eight_buses, nine_buses, time_domain


ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def write_json(path, content):
    path.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n")


def acceptance(metrics):
    nine, eight, td = (metrics[key] for key in ("nine_buses", "eight_buses", "trajectories"))
    d06, d10, d20 = (nine[key] for key in ("0.06", "0.10", "0.20"))
    checks = {"homogeneous_full_network_threshold_and_mode": bool(
                  abs(d06["uniform_power_boundary"]["watts_per_load"] - 16057.28096828506) < 1e-4
                  and d06["uniform_power_boundary"]["critical_mode_index"] == 0),
              "d006_15kw_stable_20kw_unstable": d06["at_15kw"]["rightmost_real_per_s"] < 0 < d06["at_20kw"]["rightmost_real_per_s"],
              "d010_documented_box_has_rhp_vertex": d10["paper_delta_box"]["summary"]["vertices_with_rhp"] > 0,
              "d010_equilibrium_envelope_retains_common_lyapunov_certificate": d10["physical_delta_box"]["equations"]["7"]["verdict"] == "certified",
              "d020_eq10_all_paper_vertices_verified": d20["paper_delta_box"]["equations"]["10"]["verdict"] == "certified",
              "eight_bus_independent_eq10_positive_at_03ohm_153": eight["0.3"]["eq10_direct_full_lmi_at_published_delta"]["gamma_recovered_per_s"] > 0.001,
              "eight_bus_independent_eq10_negative_at_1ohm_201": eight["1.0"]["eq10_direct_full_lmi_at_published_delta"]["gamma_recovered_per_s"] < -1e-5,
              "fig7_17kw_rhp_after_15kw_stable": td["7"]["comparison_by_power"][-2]["rightmost_real_per_s"] < 0 < td["7"]["comparison_by_power"][-1]["rightmost_real_per_s"],
              "fig8_declared_conditions_disagree_with_steady_voltage_floor": min(metrics["engineering"]["nominal_440v_operating_point"]["load_v"]) < 410,
              "fig8_declared_all_20kw_rhp": metrics["engineering"]["nominal_440v_operating_point"]["rightmost_real_per_s"] > 0}

    def assembly_checks(prefix, row):
        limits = {"branch_equilibrium_state_max_error": 1e-6, "equilibrium_branch_residual_si_per_s": 1e-6,
                  "off_equilibrium_rhs_error_si_per_s": 1e-6, "finite_difference_jacobian_error_per_s": 2e-5,
                  "equilibrium_power_balance_error_w": 1e-6}
        return {f"{prefix}_{key}": row[key] < limit for key, limit in limits.items()}

    for d, row in nine.items():
        checks.update(assembly_checks(f"droop_{d}", row["assembly_checks"]))
        checks[f"droop_{d}_full_modes_match_laplacian_partition"] = row["modal_full_matrix_error_per_s"] < 1e-7
        checks[f"droop_{d}_narrow_dc_envelope_preserves_existence_and_uniqueness"] = (
            row["dc_equilibrium_envelope"]["lipschitz_bound_in_paper_voltage_range"] < 1
            and row["dc_equilibrium_envelope"]["dc_impedance_min_ohm"] >= 0
            and row["power_corner_survey"]["min_load_v"] >= 360
            and row["power_corner_survey"]["max_load_v"] <= 440)
    for r, row in eight.items():
        checks[f"eight_eq7_{r}_one_unit_feasible_infeasible_bracket"] = row["eq7_box_bound_per_s"] == [row["paper_upper_limits_per_s"]["7"]-1,
                                                                                                row["paper_upper_limits_per_s"]["7"]]
        checks[f"eight_eq10_{r}_verified_near_boundary_bracket"] = row["eq10_boundary_bracket_per_s"][1] - row["eq10_boundary_bracket_per_s"][0] <= 0.126
    for key, row in td.items():
        checks[f"fig{key}_published_step_timing_readback"] = row["max_schedule_readback_error_s"] < .025
        if key != "8":
            checks[f"fig{key}_settled_plateaus_within_0p1v"] = row["max_settled_apparent_plateau_error_v"] < .1
        if key in ("4", "8"):
            checks[f"fig{key}_two_integrators_agree"] = row["last_step_independent_branch_bdf_max_probe_error_v"] < .04
    checks.update(assembly_checks("eight_bus_440v", metrics["engineering"]["assembly_checks"]))
    return {key: bool(value) for key, value in checks.items()}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    parser.add_argument("--figures", type=Path)
    parser.add_argument("--reference", type=Path, default=ROOT / "results/pdf_vector_readback.json")
    args = parser.parse_args()
    figures = args.figures or (ROOT / "figures" if args.out == ROOT / "results" else args.out / "figures")
    args.out.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    clock = perf_counter()
    config_path = ROOT / "configs/protocol.json"
    protocol = json.loads(config_path.read_text())
    reference = json.loads(args.reference.read_text())
    components = Components(**protocol["components"])
    arrays = {}
    metrics = {"doi": protocol["doi"]}
    stage_seconds = {}
    for key, run in (("nine_buses", lambda: nine_buses(components, protocol, arrays)),
                     ("eight_buses", lambda: eight_buses(components, protocol, arrays)),
                     ("engineering", lambda: eight_bus_practice(components, protocol, metrics["eight_buses"])),
                     ("trajectories", lambda: time_domain(components, protocol, reference, arrays))):
        since = perf_counter()
        metrics[key] = run()
        stage_seconds[key] = perf_counter() - since
    metrics["checks"] = acceptance(metrics)

    since = perf_counter()
    write_json(args.out / "metrics.json", metrics)
    write_json(args.out / "pdf_vector_readback.json", reference)
    np.savez_compressed(args.out / "numerical_evidence.npz", **arrays)
    render(metrics, arrays, figures)
    stage_seconds["write_outputs_and_figures"] = perf_counter() - since
    implementation = {p.name: digest(p) for p in sorted((ROOT / "src").glob("*.py")) if not p.name.startswith("._")}
    manifest = {"doi": protocol["doi"], "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
                "wall_seconds": perf_counter() - clock, "stage_seconds": stage_seconds,
                "environment": {"python": platform.python_version(), "platform": platform.platform(),
                                **{p: version(p) for p in ("numpy", "scipy", "cvxpy", "clarabel", "cvxopt", "matplotlib")}},
                "protocol_sha256": digest(config_path), "implementation_sha256": implementation,
                "data_sha256": {name: digest(args.out / name) for name in ("metrics.json", "pdf_vector_readback.json", "numerical_evidence.npz")},
                "figures_sha256": {name: digest(figures / name) for name in ("nine_bus_modes.png", "figure_4_5_7_8_replay.png", "eight_bus_certificate_limits.png")},
                "checks_passed": sum(metrics["checks"].values()), "checks_total": len(metrics["checks"])}
    write_json(args.out / "run_manifest.json", manifest)
    print(f"Checks {manifest['checks_passed']}/{manifest['checks_total']} in {manifest['wall_seconds']:.1f} s", flush=True)
    if manifest["checks_passed"] != manifest["checks_total"]:
        print("Failed checks", [key for key, ok in metrics["checks"].items() if not ok], flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
