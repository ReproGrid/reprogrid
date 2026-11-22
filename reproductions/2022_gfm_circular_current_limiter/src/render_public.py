"""Render the independent model's outputs. No digitized author traces are plotted."""
from argparse import ArgumentParser
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, default=Path("results/extended_model_traces.npz"))
    parser.add_argument("--output", type=Path, default=Path("figures/sampled_circuit_response.png"))
    args = parser.parse_args()
    with np.load(args.traces, allow_pickle=False) as signals:
        a = signals["digital_si_nominal_fig5"]
        b = signals["digital_si_nominal_fig6"]
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": "#294158", "text.color": "#142c41"})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    fig.patch.set_facecolor("#f2f7fa")
    for ax in axes.flat:
        ax.grid(color="#dce6ec", linewidth=0.8)
        ax.set_axisbelow(True)
    blue, orange, gray = "#176783", "#b65335", "#53667b"
    phase, near, power, current = axes.flat
    for values, color, length in ((a, blue, 450), (b, orange, 650)):
        time, angle = values[:, 0], values[:, 1] - values[0, 1]
        phase.plot(time, angle, lw=1.8, color=color, label=f"{length} ms sag")
        phase.axvline(length / 1000, color=color, lw=0.9, ls=":")
        near.plot(time, angle, lw=1.6, color=color, label=f"{length} ms sag")
    phase.axhline(2 * np.pi, color=gray, linestyle="--", linewidth=0.85, label="one full turn")
    phase.set(xlim=(0, 3.2), title="Unwrapped angle during recovery", xlabel="Seconds from sag onset", ylabel="Angle change (rad)")
    phase.legend(frameon=False, fontsize=9, loc="lower right")
    near.axvspan(0, 0.45, color="#e9f1f5")
    near.axvline(0.45, color=blue, ls=":")
    near.axvline(0.65, color=orange, ls=":")
    near.set(xlim=(0, 0.8), ylim=(-0.01, 1.02 * max(a[a[:, 0] <= 0.8, 1] - a[0, 1])), title="Clearing windows at 450 and 650 ms",
             xlabel="Seconds from sag onset", ylabel="Angle change (rad)")
    rows = []
    for start, stop in ((0, 4500), (4500, 6501)):
        for left in range(start, stop, 200):
            section = a[left:min(left + 200, stop)]
            p = section[:, 2]
            rows.append((1000 * section[:, 0].mean(), p.min(), p.mean(), p.max()))
    t_power, low, mean, high = np.asarray(rows).T
    power.axvspan(0, 450, color="#e9f1f5")
    power.fill_between(t_power, low, high, alpha=0.19, color=blue, label="Range per 20 ms")
    power.plot(t_power, mean, color=blue, linewidth=1.8, label="Mean per 20 ms")
    power.axvline(450, color=gray, ls=":", linewidth=0.9)
    power.set(xlim=(0, 650), title="450 ms case · terminal active power",
              ylabel="Power (pu)", xlabel="Milliseconds from sag onset")
    power.legend(frameon=False, fontsize=9, loc="lower left")
    stop = np.searchsorted(a[:, 0], 0.02, side="right")
    current.plot(a[:stop, 0] * 1000, a[:stop, 4], color=blue, lw=1.4, label="Simulated current")
    current.set(xlim=(0, 20), title="450 ms case · current at sag onset",
                ylabel="Current (pu)", xlabel="Milliseconds from sag onset")
    current.step(a[:stop, 0] * 1000, a[:stop, 7], where="post", color=orange, lw=1.15, label="Limited reference")
    current.axhline(1.2, lw=0.9, color=gray, ls="--", label="1.2 pu reference ceiling")
    current.legend(frameon=False, fontsize=9)
    fig.suptitle("Circular current limit · independent 10 kHz circuit", fontsize=17, fontweight="bold")
    fig.supxlabel("Both simulated trajectories slip by one turn. The published 450 ms experiment returned without a slip.",
                  fontsize=9.5, color=gray)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=190)
    plt.close(fig)


if __name__ == "__main__":
    main()
