"""Publication-readable plots from the replayed numerical data."""

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


BLUE = "#176482"
ORANGE = "#a84025"
GREEN = "#238169"
INK = "#293b44"
STYLE = {"axes.spines.top": False, "axes.spines.right": False, "axes.labelcolor": INK, "text.color": INK,
         "axes.titleweight": "bold", "axes.grid": True, "grid.alpha": .22, "font.size": 10, "savefig.dpi": 160}


def uniform_modes(metrics, arrays, out):
    fig, ax = plt.subplots(figsize=(9.3, 4.2), layout="constrained")
    for d, color in (("0.06", ORANGE), ("0.10", BLUE), ("0.20", GREEN)):
        key = f"nine_{round(100*float(d)):02}"
        p = arrays[f"{key}_uniform_powers_w"] / 1000
        alpha = arrays[f"{key}_modal_rates_per_s"]
        ax.plot(p, alpha[:, 0], c=color, lw=2, label=f"{d} Ω, uniform mode")
        ax.plot(p, np.max(alpha[:, 1:], axis=1), c=color, ls=":", lw=1.7)
    boundary = metrics["nine_buses"]["0.06"]["uniform_power_boundary"]["watts_per_load"] / 1000
    ax.axhline(0, c=INK, lw=.8)
    ax.scatter([boundary], [0], c=ORANGE, zorder=5)
    ax.annotate(f"{boundary:.3f} kW", xy=(boundary, 0), xytext=(-8, 22), textcoords="offset points", ha="center")
    ax.set(title="Nine-bus stability with equal CPL powers", xlabel="Power per CPL (kW)", ylabel="Rightmost real part (s⁻¹)")
    ax.legend(handles=[*ax.get_legend_handles_labels()[0], Line2D([0], [0], ls=":", color=INK,
                                                                 label="Largest nonuniform-mode real part")], loc="lower right")
    fig.savefig(out / "nine_bus_modes.png")
    plt.close(fig)


def event_replays(metrics, arrays, out):
    fig, axs = plt.subplots(2, 2, figsize=(11.5, 6.4), layout="constrained")
    for ax, key in zip(axs.flat, ("4", "5", "7", "8"), strict=True):
        row = metrics["trajectories"][key]
        t = arrays[f"fig{key}_time_s"]
        ax.plot(t, arrays[f"fig{key}_probe_v"], c=BLUE, lw=1.2, label="Branch-equation replay")
        timing = [(np.mean(s["window_s"]), s["pdf_median_v"]) for s in row["comparison_by_power"]
                  if s["pdf_median_v"] is not None and s["pdf_span_v"] < .1]
        ax.scatter(*np.array(timing).T, s=22, marker="o", facecolors="white", edgecolors=ORANGE,
                   zorder=5, label="PDF settled-voltage readback")
        if key == "8":
            ax.plot(t, arrays["fig8_min_cpl_v"], c=GREEN, lw=1.0, label="Minimum across all CPLs")
            ax.axhline(410, c=INK, ls="--", lw=1, label="410 V in article")
            ax.legend(loc="lower left", fontsize=8)
        elif key == "5":
            ax.legend(loc="lower left", fontsize=8)
        detail = {"4": "d = 0.06 Ω", "5": "d = 0.20 Ω", "7": "Vref = 400 V", "8": "Vref = 440 V"}[key]
        probe = "bus" if row["probe"] == "bus_voltage" else "CPL"
        ax.set(title=f"Fig. {key}   {detail}", xlabel="Time (s)", ylabel=f"Bus {row['probe_bus']} {probe} voltage (V)",
               xlim=(t[0], t[-1]))
    fig.savefig(out / "figure_4_5_7_8_replay.png")
    plt.close(fig)


def certificate_limits(metrics, out):
    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    labels = []
    for r, row in metrics["eight_buses"].items():
        bounds = {"7": row["eq7_box_bound_per_s"], "9": [row["eq9_analytic_boundary_per_s"]] * 2,
                  "10": row["eq10_boundary_bracket_per_s"]}
        for equation, (lo, hi) in bounds.items():
            y = len(labels)
            labels.append(f"{r} Ω  Eq. ({equation})")
            ax.plot([lo, hi], [y, y], color=BLUE, lw=3)
            ax.scatter([(lo+hi)/2], [y], color=BLUE, s=26, zorder=3)
            published = row["paper_upper_limits_per_s"].get(equation)
            if published is not None:
                ax.scatter([published], [y], c=ORANGE, marker="x", s=72, linewidth=1.5, zorder=4)
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.invert_yaxis()
    ax.set(title="Eight-bus robust certificate bounds", xlabel="Uniform upper incremental CPL gain δ (s⁻¹)")
    ax.legend(handles=[Line2D([0], [0], color=BLUE, marker="o", label="Reconstruction, verified range"),
                       Line2D([0], [0], color=ORANGE, marker="x", ls="None", label="Published value")], loc="lower left")
    fig.savefig(out / "eight_bus_certificate_limits.png")
    plt.close(fig)


def render(metrics, arrays, out):
    with plt.rc_context(STYLE):
        uniform_modes(metrics, arrays, out)
        event_replays(metrics, arrays, out)
        certificate_limits(metrics, out)
