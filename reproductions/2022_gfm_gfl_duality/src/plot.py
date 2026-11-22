"""Draw independent outputs and aggregate verification errors."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def make_figure(models, comparison, directory: Path):
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "text.color": "#142941", "axes.labelcolor": "#274662"})
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8), layout="constrained")
    fig.patch.set_facecolor("#f3f7f8")
    inferred = models["trapezoid"]
    t = inferred["time"] - 1.8
    for ax in axes.flat:
        ax.grid(color="#dfe8ec", alpha=0.85)
        ax.set_axisbelow(True)
    for ax in axes[0]:
        ax.axvspan(0.2, 0.4, color="#e7f3ef")
        ax.set(xlim=(0, 0.6), xlabel="Time on paper axis (s)")
    ax = axes[0, 0]
    ax.plot(t, models["source_mask"]["frequency_hz"] - 50, color="#bd5b22", label="SLX mask · 10⁻⁹ Ω", lw=1.5)
    ax.plot(t, inferred["frequency_hz"] - 50, color="#076d68", label="Trace-inferred · 0.01 Ω", lw=2)
    ax.set(title="PLL response through both changes", ylabel="Frequency deviation (Hz)")
    ax.legend(loc="upper left", frameon=False)
    ax = axes[0, 1]
    for j, (axis, color) in enumerate(zip(("d", "q"), ("#076d68", "#594c9b"))):
        ax.plot(t, inferred["ildq"][:, j], color=color, lw=1.8, label=f"{axis}-axis current")
    ax.set(title="Current loop with inferred resistance", ylabel="Filter current (pu)")
    ax.legend(loc="lower left", frameon=False)
    ax = axes[1, 0]
    for j, (phase, color) in enumerate(zip("ABC", ("#076d68", "#bd5b22", "#594c9b"))):
        ax.plot(t, inferred["igabc"][:, j], color=color, lw=1.6, label=f"Phase {phase}")
        event = next(item for item in comparison["trapezoid"]["switches"] if item["phase"] == phase and not item["closed"])
        when = event["time_s"] - 1.8
        ax.axvline(when, color=color, lw=0.8, ls=":")
        ax.text(when, 0.92, phase, ha="center", transform=ax.get_xaxis_transform(), color=color)
    ax.set(title="Opening occurs at each phase's current zero", xlim=(0.198, 0.216),
           ylim=(-0.58, 0.58), ylabel="Grid current (pu)", xlabel="Time on paper axis (s)")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax = axes[1, 1]
    options = [("source_mask", "SLX mask"), ("trapezoid", "Trapezoid"),
               ("backward_euler", "Backward Euler"), ("two_half_backward_euler", "Two half steps")]
    y = np.arange(len(options))
    for index, (color, label) in enumerate((("#2b758d", "Opening"), ("#7555a6", "Reclosing"))):
        values = [comparison[key]["transients"][index]["channels"]["frequency_hz"]["max_abs_error"] for key, _ in options]
        ax.barh(y + (index - 0.5) * 0.30, values, height=0.28, color=color, label=label)
    ax.axvline(0.15, color="#bd5b22", lw=1.15, ls="--", label="0.15 Hz limit")
    ax.set(yticks=y, yticklabels=[label for _, label in options], xlim=(0, 0.46),
           title="Only the inferred-resistance cases pass", xlabel="Peak frequency error against archive (Hz)")
    ax.invert_yaxis()
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    fig.suptitle("Grid-following inverter · independent single-grid replay", fontsize=17, fontweight="bold")
    fig.supxlabel("Waveform panels show independent model outputs. The error panel reports aggregate comparisons. "
                  "Scope: computational validation with an inferred resistance.", fontsize=9, color="#38536d")
    fig.savefig(directory / "gfl_closed_loop.png", dpi=180)
    plt.close(fig)
