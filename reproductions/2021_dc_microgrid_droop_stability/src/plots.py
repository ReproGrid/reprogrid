"""Generate figures from the numerical outputs and paper inputs."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from model import voltage_limits


COLORS = ("#2b5298", "#187465", "#b95837", "#94672a", "#78579e", "#9c3766", "#525d6d")


def save(figure, folder, name):
    path = folder / name
    figure.savefig(path, dpi=165)
    plt.close(figure)
    return path


def load_limits(folder, cfg, data):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True, layout="constrained")
    u = cfg["paper"]["equivalent_simulation"]["source_voltage_v"]
    l = cfg["paper"]["equivalent_simulation"]["inductance_h"]
    r = np.linspace(.001, .12, 1400)
    bounds = cfg["paper"]["voltage_bounds_pu"]
    cap = cfg["paper"]["equivalent_simulation"]["cases"]
    for ax, key in zip(axes, cap, strict=True):
        c = cap[key]["capacitance_f"]
        ratio = r**2 * c / l
        fold = u**2 / (4 * r)
        stability = fold * np.where(ratio < 1, 4 * ratio / (1 + ratio)**2, 1)
        lower, upper = voltage_limits(u, r[:, None], bounds).T
        ax.plot(r, stability / 1e3, lw=1.7, color=COLORS[0], label="Local stability boundary")
        ax.plot(r, lower / 1e3, color=COLORS[2], label="Steady bus-voltage limits")
        ax.plot(r, upper / 1e3, color=COLORS[2])
        ax.fill_between(r, upper / 1e3, np.minimum(stability, lower) / 1e3, color=COLORS[1], alpha=.12)
        rec = data["equivalent"][key]
        r0 = rec["limits"]["r_eq_ohm"]
        ax.axvline(r0, color="#737986", ls="--", lw=1, label="Table II droop resistance")
        step = rec["independent_step_sensitivity"]["voltage_band_limits"]
        ax.scatter([r0, r0], [np.mean(step["positive_load_w"]) / 1e3, -np.mean(step["negative_load_magnitude_w"]) / 1e3],
                   color=COLORS[1], s=33, zorder=5, label="Step limit for ±10% voltage")
        ax.set(xlim=(0., .12), ylim=(-180, 320), xlabel="Droop resistance (Ω)", title=f"Bus capacitance {key}")
        ax.grid(alpha=.22)
    axes[0].set_ylabel("Constant load power (kW)")
    axes[0].legend(loc="lower right", frameon=False, fontsize=8.3)
    return save(fig, folder, "load_limits.png")


def phase_portraits(folder, cfg, traces):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.85), layout="constrained")
    caps = cfg["paper"]["equivalent_simulation"]["cases"]
    for ax, (tag, case) in zip(axes, caps.items(), strict=True):
        for p, color in zip(case["phase_portrait_loads_w"], COLORS, strict=True):
            _, i, v = traces[f"{tag}_{p}"].T
            ax.plot(i / 1e3, v, label=f"{p / 1e3:g} kW", color=color, lw=1.2)
        for v in (180., 220.):
            ax.axhline(v, color="#737986", ls=":", lw=.9)
        ax.set(title=f"Bus capacitance {tag}", xlabel="Equivalent source current (kA)", ylabel="Bus voltage (V)")
        ax.grid(alpha=.17)
        ax.legend(loc="lower left", ncol=2, frameon=False, fontsize=8.2)
    return save(fig, folder, "phase_portraits.png")


def finite_steps(folder, traces, counter, data):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.45), layout="constrained")
    ax, right = axes
    for tag, color in zip(("25mF", "6mF"), COLORS):
        steady = data["equivalent"][tag]["within_static_limits"]["equilibrium_v"]
        t, _, v = traces[f"{tag}_within_static_limits"].T
        ax.plot(t * 1e3, v, color=color, label=tag)
    ax.axhline(180, ls="--", color=COLORS[2], label="0.9 p.u. floor")
    ax.axhline(steady, ls=":", color="#606875", label="Steady state")
    ax.set(xlim=(0, 16), title="70 kW step from no load", xlabel="Time (ms)", ylabel="Bus voltage (V)")
    ax.legend(frameon=False, loc="lower right", fontsize=8.5)
    t, _, v = counter["0.025"].T
    points = data["counterexample"]
    right.plot(t * 1e3, v, color=COLORS[0], label="Nonlinear trajectory")
    right.axhline(points["high_equilibrium_v"], color=COLORS[1], ls="--", label="Stable equilibrium")
    right.axhline(points["saddle_voltage_v"], color=COLORS[2], ls=":", label="Saddle voltage")
    right.axhline(points["voltage_below_which_zero_is_unavoidable_v"], color="#737986", ls="-.",
                  label="Below this, recovery is impossible")
    right.set(title="200 kW step from no load · 25mF", xlabel="Time (ms)", ylabel="Bus voltage (V)")
    right.legend(frameon=False, loc="upper right", fontsize=8.5)
    for axis in axes:
        axis.grid(alpha=.22)
    return save(fig, folder, "finite_load_steps.png")


def four_branch_steps(folder, cfg, traces):
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7), sharex="col", layout="constrained", height_ratios=(1, 2.3))
    for index, (key, spec) in enumerate(cfg["paper"]["four_converter_system"]["cases"].items()):
        ax, bx = axes[:, index]
        t = [0, *spec["change_times_s"], spec["end_s"]]
        p = np.r_[0, spec["load_levels_w"], spec["load_levels_w"][-1]] / 1e3
        ax.step(np.asarray(t) * 1e3, p, where="post", color=COLORS[0])
        ax.set(title=f"Four converters · {key}", ylabel="Load (kW)")
        for branch, color, ls, label in (
            ("full", COLORS[0], "-", "Four branches"),
            ("parallel_equivalent", COLORS[1], "--", "Single equivalent"),
            ("hypothetical_current_limited_load", COLORS[2], ":", "Current-limited load scenario"),
        ):
            data = traces[f"{key}_{branch}"]
            bx.plot(data[:, 0] * 1e3, data[:, -1], color=color, ls=ls, lw=1.5, label=label)
        bx.axhline(90, color="#737986", lw=.8, ls="-.", label="0.9 p.u. floor")
        bx.set(xlabel="Time (ms)", ylabel="Bus voltage (V)", xlim=(0., spec["end_s"] * 1e3), ylim=(0., 105.))
        for a in (ax, bx):
            a.grid(alpha=.2)
    axes[1, 0].legend(frameon=False, loc="lower left", fontsize=8.5)
    return save(fig, folder, "four_converter_steps.png")


def draw(folder, cfg, data, phase, four, counter):
    return (load_limits(folder, cfg, data), phase_portraits(folder, cfg, phase),
            finite_steps(folder, phase, counter, data), four_branch_steps(folder, cfg, four))
