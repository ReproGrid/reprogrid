"""Separate timing conventions and a series-resistance sensitivity in RFPSC Fig. 5(b)."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

CASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CASE))
from reproduce import read_reference, step_metrics
from src.reproduce import PROTOCOL, dynamics_rfpsc, make_scenarios, plt, read_json


def summarize(trace, sequence, window):
    times, levels, end = (sequence[key] for key in ("times_s", "p_pu", "stop_s"))
    command = step_metrics(trace, times, levels, end, window)
    t = trace["time_s"]
    before_first = (t >= times[0] - window) & (t < times[0])
    plateaus = [float(np.mean(trace["p_g_pu"][before_first]))]
    plateaus.extend(row["last_50_ms_mean"]["p_g_pu"] for row in command)
    realized = step_metrics(trace, times, plateaus, end, window)
    return [
        {
            "step_s": nominal["step_s"],
            "command_pu": [nominal["from_pu"], nominal["to_pu"]],
            "active_power_plateaus_pu": [actual["from_pu"], actual["to_pu"]],
            "q_after_pu": nominal["last_50_ms_mean"]["q_g_pu"],
            "command_relative_10_90_ms": nominal["rise_or_fall_10_90_ms"],
            "plateau_relative_10_90_ms": actual["rise_or_fall_10_90_ms"],
        }
        for nominal, actual in zip(command, realized, strict=True)
    ]


def sustained_rise(trace, start, stop, levels, hold_s):
    t = trace["time_s"]
    in_step = (t >= start) & (t < stop)
    time = t[in_step]
    low, high = levels
    progress = (trace["p_g_pu"][in_step] - low) / (high - low)

    def entry(index, level):
        if index == 0:
            return time[0]
        return np.interp(level, progress[index - 1:index + 1], time[index - 1:index + 1])

    ten = np.flatnonzero(progress >= .1)
    if not ten.size:
        return None
    t10 = entry(ten[0], .1)
    runs = np.flatnonzero(np.diff(np.r_[False, progress >= .9, False])).reshape(-1, 2)
    for first, after in runs:
        t90 = entry(first, .9)
        exit_time = time[-1] if after == time.size else np.interp(
            .9, progress[after - 1:after + 1][::-1], time[after - 1:after + 1][::-1]
        )
        if exit_time - t90 >= hold_s:
            return float((t90 - t10) * 1000)
    return None


def simulate_resistance_mismatch(residual, cfg, scenario, time, solver):
    """Apply R - Rhat to the plant and retain the Appendix C net-voltage feedback."""
    base = cfg["base"]
    ug, wbase = base["grid_voltage_pu"], 2 * np.pi * base["frequency_hz"]
    inductance, gains = scenario["L"], cfg["table_ii"]
    state = np.zeros(4)
    samples = np.empty((time.size, state.size))
    requested = np.empty(time.size, dtype=complex)
    for start, stop, _, pref, _, vref in scenario["segments"]:
        def rhs(_, x):
            dx, current, *_ = dynamics_rfpsc(x, ug, pref, vref, inductance, gains)
            dx[:2] -= residual / inductance * np.array([current.real, current.imag])
            return wbase * dx

        result = solve_ivp(rhs, (start, stop), state, dense_output=True, **solver)
        if not result.success:
            raise RuntimeError(result.message)
        mask = (time >= start) & ((time <= stop) if stop == time[-1] else (time < stop))
        samples[mask] = result.sol(time[mask]).T
        requested[mask] = pref / vref + 1j * samples[mask, 3]
        state = result.y[:, -1]
    return {
        "time_s": time,
        "p_g_pu": ug * samples[:, 0],
        "q_g_pu": -ug * samples[:, 1],
    }, float(np.max(abs(requested)))


def plot_rated_step(traces, metrics, rated_index, destination):
    fig, (absolute, normalized) = plt.subplots(1, 2, figsize=(10, 4.1), layout="constrained")
    fig.suptitle("RFPSC weak-grid rated step · Fig. 5(b)")
    for (label, trace), color, style in zip(
        traces.items(), ("#252525", "#185a88", "#ad5325"), ("-", "--", "-."), strict=True
    ):
        row = metrics[label][rated_index]
        elapsed = (trace["time_s"] - row["step_s"]) * 1000
        mask = (elapsed >= 0) & (elapsed <= 180)
        power = trace["p_g_pu"][mask]
        before, after = row["active_power_plateaus_pu"]
        absolute.plot(elapsed[mask], power, color=color, ls=style, lw=1.5, label=label)
        normalized.plot(elapsed[mask], (power - before) / (after - before),
                        color=color, ls=style, lw=1.5)
    p0, p1 = metrics["Fig. 5(b)"][rated_index]["command_pu"]
    for fraction in (.1, .9):
        absolute.axhline(p0 + fraction * (p1 - p0), color=".55", lw=.9, ls=":")
        normalized.axhline(fraction, color=".55", lw=.9, ls=":")
    absolute.set(title="Thresholds from power command", ylabel="Grid active power (pu)", ylim=(.62, 1.02))
    normalized.set(title="Thresholds from each curve's plateaus", ylabel="Fraction of realized step")
    for ax in (absolute, normalized):
        ax.set(xlabel="Time since rated command (ms)", xlim=(0, 180))
        ax.grid(alpha=.22)
    absolute.legend(loc="lower right", fontsize=8)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=CASE / "diagnostics/weak_grid_gap.json")
    parser.add_argument("--figure", type=Path, default=CASE / "figures/rfpsc_weak_grid_gap.png")
    args = parser.parse_args()
    cfg = read_json(PROTOCOL)
    seq, numerics = cfg["power_sequence"], cfg["numerics"]
    scenario = next(row for row in make_scenarios(cfg) if row["name"] == "rfpsc_weak")
    rated_index = seq["p_pu"].index(max(seq["p_pu"])) - 1
    window = numerics["plateau_window_s"]
    reference = CASE / "reference/fig5b_rfpsc.csv"
    archive = CASE / "results/ideal_rfpsc_weak.npz"
    paper = read_reference(reference)
    with np.load(archive, allow_pickle=False) as data:
        nominal = {"time_s": data["time_s"], "p_g_pu": data["p"], "q_g_pu": data["q"]}
    t = nominal["time_s"]
    zero_residual, _ = simulate_resistance_mismatch(0., cfg, scenario, t, numerics["main_solver"])
    for signal in ("p_g_pu", "q_g_pu"):
        np.testing.assert_allclose(zero_residual[signal], nominal[signal], rtol=0,
                                   atol=numerics["replay_array_tolerance"])

    paper_metrics = summarize(paper, seq, window)
    starts = seq["times_s"]
    stops = [*starts[1:], seq["stop_s"]]
    fit_bracket = (0., .15)
    fitted = []
    for index in range(rated_index + 1):
        target = paper_metrics[index]["active_power_plateaus_pu"][1]
        tail = (t >= stops[index] - window) & (t < stops[index])

        def plateau_error(residual):
            trace, _ = simulate_resistance_mismatch(residual, cfg, scenario, t, numerics["main_solver"])
            return float(np.mean(trace["p_g_pu"][tail])) - target

        residual = brentq(plateau_error, *fit_bracket, xtol=1e-10)
        fitted.append({"command_pu": seq["p_pu"][index + 1], "fitted_R_minus_Rhat_pu": residual})

    rated_fit = fitted[rated_index]["fitted_R_minus_Rhat_pu"]
    adjusted, peak = simulate_resistance_mismatch(rated_fit, cfg, scenario, t, numerics["main_solver"])
    refined, _ = simulate_resistance_mismatch(rated_fit, cfg, scenario, t, numerics["refined_solver"])
    refinement_error = max(float(np.max(abs(adjusted[key] - refined[key]))) for key in ("p_g_pu", "q_g_pu"))
    if refinement_error > numerics["trajectory_tolerance_pu"]:
        raise RuntimeError(f"Refined P/Q error exceeds the trajectory tolerance: {refinement_error}")

    traces = {"Fig. 5(b)": paper, "Nominal average model": nominal, "Resistance fit to rated P": adjusted}
    metrics = {name: summarize(trace, seq, window) for name, trace in traces.items()}
    # One cycle at the Table I grid frequency.
    hold_s = 1 / cfg["base"]["frequency_hz"]
    sustained = {}
    for name, trace in traces.items():
        rated = metrics[name][rated_index]
        sustained[name] = {
            kind: sustained_rise(trace, starts[rated_index], stops[rated_index], rated[key], hold_s)
            for kind, key in (("command_relative", "command_pu"),
                              ("plateau_relative", "active_power_plateaus_pu"))
        }
    report = {
        "doi": cfg["doi"],
        "scope": "Aligned digitized grid-terminal P/Q and the 0.89 pu continuous averaged RFPSC model.",
        "input_sha256": {str(path.relative_to(CASE)): sha256(path.read_bytes()).hexdigest()
                         for path in (PROTOCOL, reference, archive, CASE / "src/reproduce.py", CASE / "reproduce.py")},
        "timing": (
            "First directed, linearly interpolated crossings after the same aligned command edge. "
            "Command-relative thresholds use the commanded levels. Plateau-relative thresholds use "
            "the respective trace's mean P in windows before the step and before the following step."
        ),
        "plateau_window_s": window,
        "metrics": metrics,
        "rated_step_plateau_window_sensitivity": [
            {"window_s": tail, **{name: summarize(trace, seq, tail)[rated_index]
                                 for name, trace in traces.items() if name != "Resistance fit to rated P"}}
            for tail in (0.025, window, 0.1)
        ],
        "rated_step_sustained_90": {
            "method": (
                "Time from first 10% crossing to the start of the first interval at or above 90% "
                "that lasts one grid cycle. Null means there is no such interval in the sampled step."
            ),
            "hold_s": hold_s,
            "rise_10_to_sustained_90_ms": sustained,
        },
        "series_resistance_sensitivity": {
            "model": "Physical command adds Rhat*i and the plant subtracts R*i; synchronization uses the net command as in Appendix C.",
            "inductance_pu": scenario["L"],
            "fit_bracket_R_minus_Rhat_pu": fit_bracket,
            "separate_fits_to_each_positive_p_plateau": fitted,
            "rated_calibration_command_pu": seq["p_pu"][rated_index + 1],
            "rated_fit_peak_current_reference_pu": peak,
            "rated_fit_refinement_max_abs_pq_error_pu": refinement_error,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    plot_rated_step(traces, metrics, rated_index, args.figure)


if __name__ == "__main__":
    main()
