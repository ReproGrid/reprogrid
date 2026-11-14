"""Draw frequency curves, pole sweeps, and the calculated load-step response."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


COLORS = ("#365fa4", "#bb6143", "#a47a15", "#147769")
LABELS = {
    "traditional": "Traditional VSG",
    "power_only": "Power feedback",
    "acceleration_only": "Acceleration feedback",
    "combined": "Combined feedback",
}


def save(fig, directory, name):
    path = directory / name
    fig.savefig(path, dpi=165)
    plt.close(fig)
    return path


def bode_comparison(directory, bode, source):
    fig, axes = plt.subplots(2, 2, figsize=(11., 6.7), sharex="col", layout="constrained")
    for column, number, cases in zip(axes.T, ("5", "6"), (LABELS, ("0.5", "1.0", "1.5")), strict=True):
        for key, color in zip(cases, COLORS):
            label = LABELS[key] if number == "5" else f"{key} mH"
            result = bode[f"fig{number}_{key}"][:, 0]
            for ax, quantity, data in zip(column, ("magnitude_db", "phase_deg"),
                                          (20 * np.log10(abs(result)), np.angle(result, deg=True)), strict=True):
                ax.semilogx(bode["frequency_hz"], data, color=color, label=label, lw=1.5)
                ax.semilogx(*source[number, key, quantity].T, color=color, ls="--", lw=.9, alpha=.75)
        for ax in column:
            ax.grid(alpha=.2, which="both")
            ax.set_xlim(.01, 100.)
        column[0].legend(frameon=False, loc="lower left", fontsize=8.4)
        column[1].set_xlabel("Frequency (Hz)")
    axes[0, 0].set(title="Figure 5 · Controller comparison", ylabel="Magnitude (dB)")
    axes[0, 1].set_title("Figure 6 · Line inductance Lₛ₁")
    axes[1, 0].set_ylabel("Phase (deg)")
    fig.legend(handles=[Line2D([], [], color="#434955", label="Recalculation"),
                        Line2D([], [], color="#434955", ls="--", label="Author manuscript readback")],
               loc="outside upper center", frameon=False, ncol=2)
    return save(fig, directory, "frequency_responses.png")


def source_poles(source, panel, key):
    return np.concatenate([array for group, array in source.items() if group[:2] == (panel, key)])


def pole_comparison(directory, loci, source):
    fig, axes = plt.subplot_mosaic([["8a", "9"], ["8b", "9"]], figsize=(11.6, 6.4), layout="constrained")
    for panel, case, limits, y_limit in (("8a", "traditional", (-12, -1.8), (-39, 39)),
                                        ("8b", "combined", (-25, .1), (-1, 1))):
        ax = axes[panel]
        vals = loci[f"fig8_{case}"][:, :3].ravel()
        points = source_poles(source, panel, case)
        ax.scatter(*points.T, marker="+", color="#6a7383", s=18, label="Author manuscript")
        ax.scatter(vals.real, vals.imag, s=13, marker="o", facecolors="none", edgecolors=COLORS[0], label="Recalculation")
        ax.set(xlim=limits, ylim=y_limit, title=f"Figure {panel} · {LABELS[case]}")
    axes["8a"].legend(loc="upper center", frameon=False, fontsize=8, ncol=2)
    ax = axes["9"]
    for key, color in zip(("30.0", "50.0", "70.0"), (COLORS[0], COLORS[3], COLORS[1]), strict=True):
        vals = loci[f"fig9_{key}"].ravel()
        visible = (vals.real >= -45) & (vals.real <= 0) & (abs(vals.imag) <= 25)
        points = source_poles(source, "9", key)
        ax.scatter(*points.T, marker="+", color=color, s=17, alpha=.5)
        ax.scatter(vals.real[visible], vals.imag[visible], s=13, facecolors="none", edgecolors=color,
                   label=f"k₁₂ = {float(key):g}")
    ax.set(xlim=(-45, 0.3), ylim=(-26, 26), title="Figure 9 · VSG1 acceleration-feedback sweep")
    ax.legend(loc="lower right", frameon=False)
    for axis in axes.values():
        axis.grid(alpha=.2)
        axis.set(xlabel="Real part (s⁻¹)", ylabel="Imaginary part (s⁻¹)")
    return save(fig, directory, "pole_loci.png")


def load_step(directory, traces):
    fig, ax = plt.subplots(figsize=(7.9, 3.85), layout="constrained")
    for key, color in zip(LABELS, COLORS, strict=True):
        ax.plot(traces["time_s"], traces[key][:, 0], color=color, label=LABELS[key])
    ax.set(xlim=(0., 5.), xlabel="Time (s)", ylabel="VSG1 power increment (p.u.)",
           title="Calculated small-signal response to a 0.01 p.u. load step")
    ax.grid(alpha=.2)
    ax.legend(frameon=False, ncol=2)
    return save(fig, directory, "load_step.png")


def draw(directory, bode, loci, bode_source, poles_source, step):
    return (bode_comparison(directory, bode, bode_source), pole_comparison(directory, loci, poles_source),
            load_step(directory, step))
