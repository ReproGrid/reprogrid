"""Compare the printed Fig. 5 conditions with inputs inferred from its vector curves."""

import argparse
from hashlib import sha256
from importlib.metadata import version
import importlib.util
import json
from pathlib import Path

from matplotlib.lines import Line2D
import numpy as np
from scipy import signal
from scipy.optimize import minimize_scalar

from reproduce import ROOT, bandwidth, geometry, local_step, loops, margins, plt, response, series_csv


PRINTED = "figure_labels"
INFERRED = "flip_upper_iq_and_round_scr3_inductance"
VARIANTS = {
    PRINTED: (False, False),
    "flip_upper_iq": (True, False),
    "round_scr3_inductance": (False, True),
    INFERRED: (True, True),
}
LABELS = {PRINTED: "Printed conditions", INFERRED: "Curve-inferred conditions"}
GROUPS = {"active": "high_active", "reactive": "high_reactive"}
COLORS = {1: "#306c97", 3: "#a9531f", 10: "#287e55"}


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def canonical_digest(data):
    return sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def load_verifier(path):
    spec = importlib.util.spec_from_file_location("psc_independent_verifier", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def peak_sensitivity(function, frequency):
    """Refine each local minimum of |1+G| in log frequency."""
    distance = abs(1 + function(frequency))
    minima = 1 + np.flatnonzero((distance[1:-1] <= distance[:-2]) & (distance[1:-1] <= distance[2:]))
    candidates = [(1.0, None)]  # Limit at infinite frequency for these proper loops.
    for index in minima:
        optimum = minimize_scalar(
            lambda log_w: abs(1 + function(np.exp(log_w))), method="bounded",
            bounds=(np.log(frequency[index - 1]), np.log(frequency[index + 1])),
            options={"xatol": 1e-12},
        )
        if not optimum.success:
            raise RuntimeError("Sensitivity minimum did not converge")
        candidates.append((float(optimum.fun), float(np.exp(optimum.x))))
    closest, omega = min(candidates, key=lambda entry: entry[0])
    return {"peak_sensitivity": 1 / closest, "peak_sensitivity_omega_pu": omega}


def pole_error(primary, independent):
    if len(primary) != len(independent):
        raise ValueError("Closed-loop model orders differ")
    delta = abs(np.asarray(primary)[:, None] - independent)
    return float(max(delta.min(axis=0).max(), delta.min(axis=1).max()))


def independent_check(verifier, systems, current, inductance, nominal, kp, kd, w, common, primary):
    a, b, c, jac, eq, _, _ = verifier.physical_model(inductance, *current, Kp=kp, **nominal)
    a_closed = a - np.outer(b, c)
    a_cascade = verifier.dc_matrix(a_closed, b, c, kd)

    def gp(omega):
        return verifier.response(a, b, c, omega)

    def gc(omega):
        return verifier.response(a_closed, b, c, omega)

    def gd(omega):
        return kd * gc(omega) / (1j * np.asarray(omega))

    frequency_errors = {}
    for name, function in (("gp", gp), ("gc", gc), ("gd", gd)):
        expected = function(common)
        frequency_errors[name] = float(np.max(abs(response(systems[name], common) - expected)
                                               / np.maximum(1, abs(expected))))
    checks = {}
    for name, function, matrix in (("gp", gp, a_closed), ("gd", gd, a_cascade)):
        state_margin = verifier.margins(function)
        state_sensitivity = peak_sensitivity(function, w)
        main = primary[name]
        checks[name] = {
            "gain_margin_abs_error": abs(main["gain_margin"] - state_margin["gain_margin_factor"]),
            "phase_margin_deg_abs_error": abs(main["phase_margin_deg"] - state_margin["phase_margin_deg"]),
            "peak_sensitivity_abs_error": abs(main["peak_sensitivity"] - state_sensitivity["peak_sensitivity"]),
            "max_pole_distance_pu": pole_error([complex(*p) for p in main["closed_poles_pu"]],
                                               np.linalg.eigvals(matrix)),
        }
    return {"scaled_frequency_errors": frequency_errors, "margin_and_pole_errors": checks,
            "jacobian_finite_difference_abs_error": jac, "equilibrium_residual": eq}


def analyze_cases(config, variants_config, vectors, verifier, w, common):
    nominal = config["nominal"]
    kp, kd = nominal["Ra"] / (nominal["kappa"] * nominal["V"]**2), 1 / (4 * np.sqrt(2))
    source_parameters = variants_config["curve_inferred"]
    cases, models, csv_rows = {}, {}, []
    for name, (flip, rounding) in VARIANTS.items():
        cases[name], models[name] = {}, {}
        for group, reference_group in GROUPS.items():
            for scr in config["fig5"]["scr"]:
                key = f"fig5_{group}_scr{scr}"
                current = config["fig5"][group].copy()
                if flip and group == "active":
                    current[1] = source_parameters["upper_iq_pu"]
                inductance = source_parameters["inductance_by_scr_label_pu"].get(str(scr), 1 / scr) if rounding else 1 / scr
                _, grid_voltage, delta = geometry(current, inductance, nominal["V"])
                models[name][key], (a, b) = loops(current, inductance, nominal, kp, kd)
                system = models[name][key]
                record = {"op": current, "scr_label": scr, "effective_scr": 1 / inductance,
                          "L_pu": inductance, "Vg_pu": grid_voltage, "delta0_deg": float(np.degrees(delta)),
                          "a": a, "b_frozen": b, "converter_Q_pu": -nominal["kappa"] * nominal["V"] * current[1],
                          "active_bandwidth_pu": bandwidth(system["gc"], w)}
                for loop, source_loop in (("gp", "active_power"), ("gd", "dc_link")):
                    measurement = vectors[reference_group, source_loop, scr]
                    vector_case = measurement["candidates"][name]
                    if (vector_case["i_d0_pu"], vector_case["i_q0_pu"], vector_case["L_pu"]) != (*current, inductance):
                        raise ValueError(f"Vector comparison and simulation inputs disagree for {name}, {key}")
                    result = {**margins(system[loop], w), **peak_sensitivity(lambda omega: response(system[loop], omega), w),
                              "rms_distance_to_paper_locus_pu": vector_case["rms_nearest_locus_distance_pu"]}
                    record[loop] = result
                    csv_rows.append({"variant": name, "group": group, "scr_label": scr, "loop": loop,
                                     "i_d0_pu": current[0], "i_q0_pu": current[1], "L_pu": inductance,
                                     **{field: result[field] for field in ("gain_margin", "phase_margin_deg", "peak_sensitivity",
                                                                           "max_closed_real_pu", "rms_distance_to_paper_locus_pu")},
                                     "active_bandwidth_pu": record["active_bandwidth_pu"]})
                record["independent_state_space"] = independent_check(verifier, system, current, inductance,
                                                                       nominal, kp, kd, w, common, record)
                cases[name][key] = record
    return cases, models, csv_rows


def claim_summary(cases):
    groups = {group: [case for key, case in cases.items() if f"_{group}_" in key] for group in GROUPS}
    return {
        "all_closed_loops_stable": all(case[loop]["max_closed_real_pu"] < 0 for case in cases.values() for loop in ("gp", "gd")),
        "minimum_active_gain_margin": min(case["gp"]["gain_margin"] for case in cases.values()),
        "active_phase_margin_minimum_scr": {group: min(rows, key=lambda item: item["gp"]["phase_margin_deg"])["scr_label"]
                                            for group, rows in groups.items()},
        "dc_gain_margin_below_four_count": sum(case["gd"]["gain_margin"] < 4 for case in cases.values()),
        "dc_phase_margin_below_45_deg_cases": [key for key, case in cases.items() if case["gd"]["phase_margin_deg"] < 45],
        "high_active_dc_sensitivity_ascending_scr": [case["scr_label"] for case in
                                                        sorted(groups["active"], key=lambda item: item["gd"]["peak_sensitivity"])],
        "weak_grid_active_bandwidth_pu": cases["fig5_active_scr1"]["active_bandwidth_pu"],
        "weak_grid_high_reactive_bandwidth_pu": cases["fig5_reactive_scr1"]["active_bandwidth_pu"],
        "weak_grid_high_reactive_dc_phase_margin_deg": cases["fig5_reactive_scr1"]["gd"]["phase_margin_deg"],
    }


def step_evidence(config, inferred, models, out):
    """Validate the curve-inferred operating points with the ideal nonlinear dq model."""
    nominal, num = config["nominal"], config["numerics"]
    kp, kd = nominal["Ra"] / (nominal["kappa"] * nominal["V"]**2), 1 / (4 * np.sqrt(2))
    tau = np.linspace(0, num["local_end_tau"], num["local_samples"])
    traces = {"tau": tau, "time_s": tau / (2 * np.pi * config["base_frequency_hz"])}
    evidence = {}
    for key, case in inferred.items():
        for channel, tf_name in (("active", "gc"), ("dc", "wc")):
            n, d = models[key][tf_name]
            _, linear = signal.step((n.coef[::-1], d.coef[::-1]), T=tau)
            current, inductance = case["op"], case["L_pu"]
            step = num["local_step_pu"]
            nonlinear, residual = local_step(current, inductance, nominal, kp, kd, channel, num["solver"], tau, step)
            refined, _ = local_step(current, inductance, nominal, kp, kd, channel, num["refinement"], tau, step)
            trace_key = f"{key}_{channel}"
            traces[f"{trace_key}_linear"], traces[f"{trace_key}_nonlinear"] = linear, nonlinear
            evidence[trace_key] = {"equilibrium_residual": residual,
                                   "linear_nonlinear_max_relative_to_step": float(max(abs(linear - nonlinear))),
                                   "DOP853_Radau_max_relative_to_step": float(max(abs(nonlinear - refined)))}
    np.savez_compressed(out / "fig5_curve_inferred_local_steps.npz", **traces)
    return evidence


def make_figures(models, cases, vectors, w, out):
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 7.4), layout="constrained")
    for row, group in enumerate(GROUPS):
        for col, loop in enumerate(("gp", "gd")):
            ax = axes[row, col]
            for variant, line_style in ((PRINTED, "--"), (INFERRED, "-")):
                for scr, color in COLORS.items():
                    curve = response(models[variant][f"fig5_{group}_scr{scr}"][loop], w)
                    ax.plot(curve.real, curve.imag, color=color, ls=line_style, lw=1.55, alpha=.82)
            title = f"{'High active current' if row == 0 else 'High reactive injection'} · {'Active-power loop' if col == 0 else 'DC-link loop'}"
            ax.set(title=title, xlim=(-1.5, 0), ylim=(-.75, .75), xlabel="Real part", ylabel="Imaginary part")
            ax.plot(-1, 0, "x", color="#20242b", ms=6)
            ax.grid(alpha=.22)
    handles = [Line2D([], [], color=color, label=f"SCR label {scr}") for scr, color in COLORS.items()]
    handles += [Line2D([], [], color="#3c4248", ls=style, label=LABELS[variant])
                for variant, style in ((PRINTED, "--"), (INFERRED, "-"))]
    fig.legend(handles=handles, loc="outside lower center", ncols=5, fontsize=9)
    fig.suptitle("Fig. 5 · Printed versus curve-inferred operating points", fontsize=14)
    fig.savefig(out / "fig5_parameter_overlay.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.1, 4.3), layout="constrained")
    scrs = list(COLORS)
    markers = {PRINTED: ("#a9531f", "o"), INFERRED: ("#216d8d", "s")}
    for ax, name, figure_key, ylabel in zip(
        axes, ("gain_margin", "peak_sensitivity"), ("visible_gain_margins_factor", "visible_peak_sensitivity_estimate"),
        ("Gain margin (factor)", "Peak sensitivity $M_s$"), strict=True,
    ):
        for variant, (color, marker) in markers.items():
            values = [cases[variant][f"fig5_active_scr{scr}"]["gd"][name] for scr in scrs]
            ax.plot(range(3), values, color=color, marker=marker, ms=5, label=LABELS[variant])
        from_figure = [vectors["high_active", "dc_link", scr][figure_key] for scr in scrs]
        if name == "gain_margin":
            from_figure = [value[0] if value else np.nan for value in from_figure]
        ax.plot(range(3), from_figure, "x", color="#252831", mew=1.5, ms=8, label="PDF estimate")
        ax.set(xticks=range(3), xticklabels=scrs, xlabel="SCR label", ylabel=ylabel)
        ax.grid(axis="y", alpha=.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=3, fontsize=9)
    fig.suptitle("Fig. 5 upper row · DC-link robustness", fontsize=14)
    fig.savefig(out / "fig5_robustness_comparison.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--figures", type=Path, default=ROOT / "figures")
    parser.add_argument("--baseline", type=Path, help="run_manifest.json from the printed-condition replay")
    parser.add_argument("--reference", type=Path, default=ROOT / "verification/fig5_vector_comparison.json")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.figures.mkdir(parents=True, exist_ok=True)
    protocol_path, variants_path = ROOT / "configs/protocol.json", ROOT / "configs/fig5_variants.json"
    verifier_path = ROOT / "verification/verifier.py"
    config, variants_config = read_json(protocol_path), read_json(variants_path)
    vector_report = read_json(args.reference)
    source = read_json(ROOT / "sources/source_manifest.json")
    if vector_report["pdf_sha256"] != source["sha256"] or vector_report["doi"] != config["doi"]:
        raise ValueError("Expected numeric reference evidence from the pinned paper")
    vectors = {(row["condition"], row["loop"], row["scr_label"]): row for row in vector_report["curves"]}
    if len(vectors) != 12:
        raise ValueError("Expected all 12 Fig. 5 curves")
    baseline = read_json(args.baseline or args.output / "run_manifest.json")
    if baseline["protocol_sha256"] != digest(protocol_path) or baseline["implementation_sha256"] != digest(ROOT / "src/reproduce.py"):
        raise ValueError("Baseline run does not match the pinned protocol and implementation")

    num = config["numerics"]
    w, common = np.geomspace(*num["margin_search_pu"]), np.geomspace(*num["cross_frequency_pu"])
    cases, models, rows = analyze_cases(config, variants_config, vectors, load_verifier(verifier_path), w, common)
    time_cases = step_evidence(config, cases[INFERRED], models[INFERRED], args.output)
    gates = variants_config["validation_tolerances"]
    independent = [item["independent_state_space"] for variant in cases.values() for item in variant.values()]
    frequency_error = max(v for case in independent for v in case["scaled_frequency_errors"].values())
    margin_error = max(v for case in independent for row in case["margin_and_pole_errors"].values() for v in row.values())
    baseline_error = max(abs(cases[PRINTED][key][loop][metric] - case[loop][metric])
                         for key, case in baseline["fig5"].items() for loop in ("gp", "gd")
                         for metric in ("gain_margin", "phase_margin_deg", "max_closed_real_pu"))
    upper_vectors = [v for v in vectors.values() if v["condition"] == "high_active"]
    rms_field = "rms_nearest_locus_distance_pu"
    minimum_sign_gain = min(v["candidates"]["round_scr3_inductance"][rms_field] / v["candidates"][INFERRED][rms_field]
                            for v in upper_vectors)
    maximum_combined_error = max(v["candidates"][INFERRED][rms_field] for v in vectors.values())
    checks = {
        "printed_baseline_matches": baseline_error < gates["independent_margin_and_pole_abs_error"],
        "independent_frequency_responses_match": frequency_error < gates["independent_frequency_scaled_error"],
        "independent_margins_sensitivity_and_poles_match": margin_error < gates["independent_margin_and_pole_abs_error"],
        "independent_equilibria_and_jacobians_pass": all(item["equilibrium_residual"] < 1e-10 and
                                                            item["jacobian_finite_difference_abs_error"] < 1e-7 for item in independent),
        "sign_alone_improves_all_six_upper_curves_at_nominal_inductance": all(
            v["candidates"]["flip_upper_iq"][rms_field] < v["candidates"][PRINTED][rms_field] for v in upper_vectors),
        "sign_alone_passes_improvement_gate_at_curve_inductance":
            minimum_sign_gain > gates["minimum_upper_row_rms_improvement_factor_at_curve_inductance"],
        "combined_inputs_match_all_twelve_paper_curves": maximum_combined_error < gates["maximum_curve_inferred_rms_locus_error_pu"],
        "curve_inferred_linear_nonlinear_and_solver_checks_pass": all(
            case["equilibrium_residual"] < 1e-10 and
            case["linear_nonlinear_max_relative_to_step"] < num["local_relative_to_step_tolerance"] and
            case["DOP853_Radau_max_relative_to_step"] < num["integrator_relative_to_step_tolerance"] for case in time_cases.values()),
    }
    report = {"doi": config["doi"], "scope": variants_config["scope"],
              "inputs_sha256": {"protocol": digest(protocol_path), "variants": digest(variants_path),
                                "implementation": digest(Path(__file__)), "primary_implementation": digest(ROOT / "src/reproduce.py"),
                                "independent_implementation": digest(verifier_path), "paper_vector_evidence": digest(args.reference),
                                "printed_fig5_baseline": canonical_digest(baseline["fig5"])},
              "packages": {name: version(name) for name in ("numpy", "scipy", "matplotlib")},
              "checks": checks, "passed": all(checks.values()),
              "validation": {"max_independent_scaled_frequency_error": frequency_error,
                             "max_independent_margin_sensitivity_or_pole_abs_error": margin_error,
                             "max_baseline_abs_error": baseline_error,
                             "minimum_upper_rms_improvement_factor_from_sign_at_curve_inductance": minimum_sign_gain,
                             "max_combined_vector_rms_pu": maximum_combined_error},
              "claims": {variant: claim_summary(results) for variant, results in cases.items()},
              "cases": cases, "curve_inferred_local_steps": time_cases}
    (args.output / "fig5_condition_comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    series_csv(args.output / "fig5_condition_comparison.csv", list(rows[0]), [row.values() for row in rows])
    make_figures(models, cases, vectors, w, args.figures)
    print(json.dumps({"checks": len(checks), "failed": [name for name, passed in checks.items() if not passed]}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
