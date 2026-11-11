"""Replay the DO-GFM and RFPSC weak-grid experiments on pinned motulator."""

import argparse
from bisect import bisect_right
from dataclasses import asdict
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_trapezoid


HERE = Path(__file__).resolve().parent
PANELS = {"rfpsc": "fig5b_rfpsc", "do_gfm": "fig6b_do_gfm"}
SIGNALS = ("p_g_pu", "q_g_pu", "u_c_ref_pu", "u_g_pu", "u_c_hat_pu")


def git_value(checkout, *args):
    return subprocess.check_output(["git", "-C", str(checkout), *args], text=True).strip()


def file_hash(path):
    return sha256(path.read_bytes()).hexdigest()


def average_over_carrier(t, y, center, period):
    """Centered one-carrier mean of the piecewise-linear plant waveform."""
    integral = cumulative_trapezoid(y, t, initial=0)

    def primitive(query):
        index = np.clip(np.searchsorted(t, query, side="right") - 1, 0, len(t) - 2)
        delta = query - t[index]
        slope = (y[index + 1] - y[index]) / (t[index + 1] - t[index])
        return integral[index] + delta * (y[index] + 0.5 * slope * delta)

    low = np.maximum(t[0], center - 0.5 * period)
    high = np.minimum(t[-1], center + 0.5 * period)
    return (primitive(high) - primitive(low)) / (high - low)


def simulate(kind, protocol, solver, *, switched):
    from motulator.grid import control, model, utils

    nameplate = protocol["table_i"]
    gains = protocol["table_ii"]
    weak = protocol["weak_grid"]
    assumptions = protocol["simulation_assumptions"]
    nominal = utils.NominalValues(
        U=nameplate["voltage_line_line_rms_v"], I=nameplate["phase_current_rms_a"],
        f=nameplate["grid_frequency_hz"], P=nameplate["nameplate_va"],
    )
    base = utils.BaseValues.from_nominal(nominal)
    dt = 1.0 / nameplate["control_sampling_hz"]
    carrier_period = 1.0 / nameplate["carrier_hz"]
    if carrier_period != 2 * dt:
        raise ValueError("Double-update PWM requires a carrier period of two control samples")
    total_inductance = weak["total_inductance_pu"] * base.L
    filter_inductance = nameplate["filter_inductance_h"]
    grid_inductance = total_inductance - filter_inductance
    filter_resistance = assumptions["plant_series_resistance_pu"] * base.Z
    pwm_args = dict(t_d=assumptions["pwm_dead_time_s"], single_update=False,
                    delay=assumptions["computational_delay_samples"])
    if switched:
        modulator = model.CarrierComparison(N=assumptions["svpwm_carrier_counter_levels"], **pwm_args)
    else:
        modulator = model.ZOH(**pwm_args)
    plant = model.GridConverterSystem(
        model.VoltageSourceConverter(u_dc=nameplate["dc_voltage_v"], modulator=modulator),
        model.LFilter(L_f=filter_inductance, L_g=grid_inductance, R_f=filter_resistance),
        model.ThreePhaseSource(w_g=base.w, e_g=weak["grid_voltage_pu"] * base.u),
    )
    i_max = assumptions["current_limit_pu"] * base.i
    common = dict(T_s=dt, u_nom=base.u, w_nom=base.w, i_max=i_max,
                  R=weak["series_resistance_estimate_pu"] * base.Z)
    if kind == "rfpsc":
        cfg = control.PowerSynchronizationControllerCfg(
            R_a=gains["rfpsc_active_resistance_pu"] * base.Z,
            w_b=gains["rfpsc_current_lpf_bandwidth_pu"] * base.w, **common,
        )
        inner = control.PowerSynchronizationController(cfg)
        derived_gain = inner.k_p_psc * base.p / base.w
        if not np.isclose(derived_gain, gains["rfpsc_power_synchronization_gain_pu"]):
            raise ValueError("Derived RFPSC synchronization gain disagrees with Table II")
    else:
        observer_gain = gains["do_observer_gain_pu_re_im"]
        if observer_gain[1] != -1:
            raise ValueError("Stock motulator supports only the RFPSC-type observer design")
        cfg = control.ObserverBasedGridFormingControllerCfg(
            L=weak["observer_inductance_estimate_pu"] * base.L,
            R_a=gains["do_power_gain_pu"] * base.Z * weak["converter_voltage_setpoint_pu"],
            k_v=gains["do_voltage_gain_pu"], alpha_o=observer_gain[0] * base.w,
            alpha_c=gains["do_current_control_bandwidth_pu"] * base.w, **common,
        )
        inner = control.ObserverBasedGridFormingController(cfg)
        derived_gain = inner.R_a / base.Z / weak["converter_voltage_setpoint_pu"]

    clipped = 0
    limiter = inner.current_limiter

    def count_limit(i_ref):
        nonlocal clipped
        clipped += int(abs(i_ref) > i_max)
        return limiter(i_ref)

    inner.current_limiter = count_limit
    controller = control.GridConverterControlSystem(inner)
    ticks = [round(time / dt) for time in weak["step_times_s"]]
    levels = weak["step_levels_pu"]
    controller.set_power_ref(lambda t: levels[bisect_right(ticks, round(t / dt))] * base.p)
    controller.set_ac_voltage_ref(weak["converter_voltage_setpoint_pu"] * base.u)
    # Provide a full carrier window around the last exported sample. The
    # simulator's stop condition can otherwise omit that controller callback.
    result = model.Simulation(plant, controller, show_progress=False,
                              cfg=model.SolverCfg(**solver)).simulate(
                                  t_stop=weak["stop_s"] + carrier_period)
    t_expected = np.arange(round(weak["stop_s"] / dt) + 1) * dt
    if not result.success or result.mdl.t[-1] < weak["stop_s"] - 1e-10:
        raise RuntimeError(f"{kind} did not reach {weak['stop_s']} s")
    n = len(t_expected)
    np.testing.assert_allclose(result.ctrl.t[:n], t_expected, rtol=0, atol=1e-10)
    t_plant, unique = np.unique(result.mdl.t, return_index=True)
    filt = result.mdl.ac_filter
    grid_voltage = filt.e_g_ab[unique]
    grid_current = filt.i_g_ab[unique]
    power = 1.5 * grid_voltage * grid_current.conjugate() / base.p
    series = {
        "time_s": t_expected,
        "p_g_ref_pu": result.ctrl.ref.p_g[:n] / base.p,
        "p_g_pu": average_over_carrier(t_plant, power.real, t_expected, carrier_period),
        "q_g_pu": average_over_carrier(t_plant, power.imag, t_expected, carrier_period),
        "p_g_sample_pu": np.interp(t_expected, t_plant, power.real),
        "q_g_sample_pu": np.interp(t_expected, t_plant, power.imag),
        "u_c_ref_pu": np.abs(result.ctrl.ref.u_c[:n]) / base.u,
        "u_g_pu": np.interp(t_expected, t_plant, np.abs(grid_voltage)) / base.u,
        "i_g_sample_pu": np.interp(t_expected, t_plant, np.abs(grid_current)) / base.i,
    }
    if kind == "do_gfm":
        series["u_c_hat_pu"] = np.abs(result.ctrl.fbk.v_c[:n]) / base.u
    if not all(np.isfinite(trace).all() for trace in series.values()):
        raise RuntimeError(f"{kind} produced non-finite traces")
    expected_commands = np.asarray(levels)[np.searchsorted(ticks, np.arange(n), side="right")]
    np.testing.assert_allclose(series["p_g_ref_pu"], expected_commands, rtol=0, atol=1e-12)
    meta = {"base": asdict(base), "controller_config_si": asdict(cfg),
            "derived_power_gain_pu": float(derived_gain), "filter_inductance_h": filter_inductance,
            "grid_inductance_h": float(grid_inductance), "plant_series_resistance_ohm": float(filter_resistance),
            "plant_samples": len(t_plant), "controller_samples": n,
            "current_limiter_activations": clipped,
            "peak_plant_current_pu": float(max(abs(grid_current[t_plant <= weak["stop_s"]])) / base.i)}
    return series, meta


def rms(value):
    return float(np.sqrt(np.mean(value**2)))


def step_metrics(trace, times, levels, end, tail):
    """Use command-relative first crossings and report each plateau separately."""
    t = trace["time_s"]
    p = trace["p_g_pu"]

    def crossing(t_step, progress, level):
        hits = np.flatnonzero(progress >= level)
        if not len(hits):
            return None
        i = hits[0]
        return float(t_step[0] if i == 0 else np.interp(
            level, progress[i - 1:i + 1], t_step[i - 1:i + 1]))

    metrics = []
    for start, stop, before, after in zip(times, times[1:] + [end], levels[:-1], levels[1:], strict=True):
        window = (start <= t) & (t < stop)
        plateau = (stop - tail <= t) & (t < stop)
        progress = (p[window] - before) / (after - before)
        first = crossing(t[window], progress, 0.1)
        last = crossing(t[window], progress, 0.9)
        metrics.append({
            "step_s": start, "from_pu": before, "to_pu": after,
            "rise_or_fall_10_90_ms": (last - first) * 1000 if None not in (first, last) else None,
            "directional_overshoot_pct": max(0.0, float(100 * (max(progress) - 1))),
            "last_50_ms_mean": {key: float(np.mean(trace[key][plateau])) for key in SIGNALS if key in trace},
        })
    return metrics


def read_reference(path):
    rows = np.genfromtxt(path, delimiter=",", names=True)
    keys = ("p_g_ref_pu", *SIGNALS[:-1])
    if path.stem == PANELS["do_gfm"]:
        keys = (*keys, SIGNALS[-1])
    t = rows["time_aligned_s"]
    if not (np.isfinite(t).all() and np.all(np.diff(t) > 0)
            and all(np.isfinite(rows[key]).all() for key in keys)):
        raise ValueError(f"Invalid or missing digitized samples in {path}")
    return {"time_s": t, **{key: rows[key] for key in keys}}


def write_trace(path, trace):
    np.savetxt(path, np.column_stack(list(trace.values())), delimiter=",",
               fmt="%.10g", header=",".join(trace), comments="")


def make_figure(out, simulations, references):
    fig, axes = plt.subplots(3, 2, sharex=True, sharey="row", figsize=(12, 8), layout="constrained")
    palettes = {"p_g_pu": "#226bb4", "q_g_pu": "#b35122", "u_c_ref_pu": "#684796",
                "u_c_hat_pu": "#13835c", "u_g_pu": "#565a62"}
    for column, kind in enumerate(PANELS):
        simulated = simulations[kind]
        paper = references[kind]
        axes[0, column].set_title("RFPSC · Figure 5(b)" if column == 0 else "DO-GFM · Figure 6(b)")
        axes[0, column].step(simulated["time_s"], simulated["p_g_ref_pu"], where="post",
                             color="#484848", lw=1, ls=":", label="Command")
        for row, names in enumerate((("p_g_pu",), ("q_g_pu",), ("u_c_ref_pu", "u_c_hat_pu", "u_g_pu"))):
            axis = axes[row, column]
            for name in names:
                if name not in simulated:
                    continue
                suffix = {"u_c_ref_pu": " command", "u_c_hat_pu": " estimate", "u_g_pu": " grid"}.get(name, "")
                grid = name == "u_g_pu"
                axis.plot(simulated["time_s"], simulated[name], color=palettes[name],
                          lw=1.0 if grid else 1.5, ls=":" if grid else "-",
                          label="motulator" + suffix)
                axis.plot(paper["time_s"], paper[name], color=palettes[name], lw=1.05,
                          ls="--", alpha=0.55 if grid else 0.78, label="Paper" + suffix)
            axis.grid(alpha=0.2)
            axis.set_xlim(0, 1.5)
            axis.legend(loc="best", fontsize=8, ncols=2)
        axes[-1, column].set_xlabel("Time after panel alignment (s)")
    for axis, label in zip(axes[:, 0], ("Grid active power (p.u.)", "Grid reactive power (p.u.)",
                                       "Voltage magnitude (p.u.)"), strict=True):
        axis.set_ylabel(label)
    axes[2, 0].set_ylim(0.80, 1.12)
    fig.suptitle("Weak-grid power tracking · 0.89 p.u. total inductance", fontsize=15)
    fig.savefig(out / "fig5b_fig6b_comparison.png", dpi=180)
    fig.savefig(out / "fig5b_fig6b_comparison.svg")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True, help="pinned upstream motulator Git checkout")
    parser.add_argument("--reference-dir", type=Path, default=HERE / "reference")
    parser.add_argument("--output", type=Path, default=HERE / "results")
    args = parser.parse_args()
    protocol_file = HERE / "protocol.json"
    protocol = json.loads(protocol_file.read_text())
    checkout = args.checkout.resolve()
    commit = git_value(checkout, "rev-parse", "HEAD")
    if commit != protocol["upstream"]["commit"] or git_value(checkout, "status", "--porcelain", "-uno"):
        raise ValueError("Expected an unmodified checkout of " + protocol["upstream"]["commit"])
    sys.path.insert(0, str(checkout))
    args.output.mkdir(parents=True, exist_ok=True)
    weak = protocol["weak_grid"]
    solver = protocol["numerics"]
    main_runs, references, checks = {}, {}, {}
    summary = {
        "doi": protocol["paper"]["doi"], "motulator_commit": commit,
        "source_hashes": {"protocol": file_hash(protocol_file), "script": file_hash(Path(__file__))},
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        **{name: version(name) for name in ("numpy", "scipy", "matplotlib", "tqdm")}},
        "series_definition": "P and Q at emulator terminals, 1.5*e_g*conj(i_g)/S_base. P and Q for figures and metrics use a centered 5 kHz carrier-period mean; CSV also contains controller-instant power samples.",
        "comparison": "Paper vector traces share the nominal first command edge after a per-panel time translation. Compare by interpolation on the simulation clock with no amplitude rescaling.",
        "runs": {},
    }
    for kind, panel in PANELS.items():
        print(f"Simulating {kind}: switched, refined switched, and averaged", flush=True)
        trace, meta = simulate(kind, protocol, solver["primary"], switched=True)
        refined, _ = simulate(kind, protocol, solver["refined"], switched=True)
        averaged, _ = simulate(kind, protocol, solver["refined"], switched=False)
        filename = args.reference_dir / f"{panel}.csv"
        paper = read_reference(filename)
        references[kind] = paper
        main_runs[kind] = trace
        start, end = solver["paper_rmse_window_s"]
        t = trace["time_s"]
        mask = (start <= t) & (t <= end) & (paper["time_s"][0] <= t) & (t <= paper["time_s"][-1])
        paper_on_sim = {key: np.interp(t[mask], paper["time_s"], paper[key]) for key in SIGNALS if key in paper}
        max_error = {key: float(max(abs(refined[key] - trace[key]))) for key in SIGNALS if key in trace}
        passed = max(max_error.values()) <= solver["max_refinement_difference_pu"]
        checks[kind] = passed
        steps = (weak["step_times_s"], weak["step_levels_pu"], weak["stop_s"], solver["plateau_window_s"])
        summary["runs"][kind] = {
            **meta, "paper_reference_sha256": file_hash(filename),
            "switched_steps": step_metrics(trace, *steps), "paper_steps": step_metrics(paper, *steps),
            "switched_vs_paper_rmse_pu": {key: rms(trace[key][mask] - signal) for key, signal in paper_on_sim.items()},
            "switched_vs_averaged_rmse_pu": {key: rms((trace[key] - averaged[key])[mask]) for key in SIGNALS if key in trace},
            "solver_refinement_max_abs_difference_pu": max_error,
            "solver_refinement_pass": passed,
        }
        write_trace(args.output / f"{panel}_switched.csv", trace)
        write_trace(args.output / f"{panel}_averaged.csv", averaged)
    overlap_start = max(start, *(paper["time_s"][0] for paper in references.values()))
    overlap_end = min(end, *(paper["time_s"][-1] for paper in references.values()))
    common_time = t[(overlap_start <= t) & (t <= overlap_end)]
    summary["pairwise_rms_difference_pu"] = {
        label: {key: rms(np.interp(common_time, comparison["rfpsc"]["time_s"], comparison["rfpsc"][key])
                         - np.interp(common_time, comparison["do_gfm"]["time_s"], comparison["do_gfm"][key]))
                for key in ("p_g_pu", "q_g_pu", "u_c_ref_pu")}
        for label, comparison in (("motulator", main_runs), ("paper", references))
    }
    x = weak["total_inductance_pu"]
    summary["ideal_loss_compensated_equilibrium"] = {
        "conditions": "1 p.u. grid voltage and 1 p.u. quasi-static converter voltage, low-angle branch",
        "formula": "Q = (sqrt(1 - (X * P)^2) - 1) / X",
        "rated_q_pu": float((np.sqrt(1 - x**2) - 1) / x),
    }
    summary["all_solver_refinement_checks_pass"] = all(checks.values())
    (args.output / "metrics.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    make_figure(args.output, main_runs, references)
    if not all(checks.values()):
        raise RuntimeError("Solver refinement threshold exceeded. See the saved metrics and traces.")
    print(f"Saved traces, metrics and plots in {args.output}")


if __name__ == "__main__":
    main()
