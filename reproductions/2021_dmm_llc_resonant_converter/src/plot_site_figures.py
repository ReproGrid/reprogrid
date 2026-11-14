from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np

colors = ['#00796c', '#af502b', '#3954a7', '#648013']
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.labelcolor": "#23463e",
                     "figure.facecolor": "#f6f9f5", "axes.facecolor": "white",
                     "axes.titlecolor": "#18382f", "text.color": "#18382f"})

def finish(fig, axes, path):
    for axis in axes:
        axis.grid(color="#d7e3dd", lw=.6, alpha=.9)
        axis.set_axisbelow(True)
    fig.savefig(path, dpi=175)
    plt.close(fig)

def llc(root):
    plots = root / "figures"
    with np.load(root / "results/deep/sequence_trajectories.npz", allow_pickle=False) as data:
        continuous = data["ideal_continuous"]
        reset = data["ideal_reset"]
        time = continuous[:, 0] * 1e-12
        fig, chart = plt.subplots(2, 1, figsize=(10, 6.2), sharex=True, layout="constrained")
        chart[0].plot(time, continuous[:, 4], color=colors[0], lw=1.7)
        chart[0].set_title("Output short and recovery", loc="left", fontweight="bold")
        chart[0].set_ylabel("Output voltage (V)")
        chart[1].step(time, continuous[:, 9] / 1000, where="post", color=colors[2], lw=1.7)
        chart[1].set(xlabel="Time (s)", ylabel="Switching frequency (kHz)", xlim=(0, .60))
        for axis in chart:
            axis.axvspan(.1, .13, color="#e9b278", alpha=.26, lw=0)
        chart[1].xaxis.set_major_locator(MultipleLocator(.1))
        finish(fig, chart, plots / "computed_full_sequence.png")

        fig, chart = plt.subplots(2, 2, figsize=(10.5, 6.7), layout="constrained")
        for axis, center, half in zip(chart.flat, (.05, .1, .26, .55), (5e-6, 5e-6, 10e-6, 4e-6)):
            for wave, name, color, style in ((continuous, "Continuous phase", colors[0], "-"),
                                              (reset, "Phase reset at frequency change", colors[1], "--")):
                t = wave[:, 0] * 1e-12
                window = abs(t - center) <= half * 1.001
                axis.plot((t[window] - center) * 1e6, wave[window, 1], label=name, lw=1.65,
                          color=color, ls=style)
            axis.set_title(f"Near {center*1000:g} ms", loc="left", fontweight="bold")
            axis.set(xlabel="Time relative to window center (µs)", ylabel="Resonant current (A)")
        chart[0, 0].legend(loc="best", frameon=False, fontsize=9)
        finish(fig, chart.flat, plots / "computed_current_windows.png")

if __name__ == "__main__":
    llc(Path(__file__).resolve().parents[1])
