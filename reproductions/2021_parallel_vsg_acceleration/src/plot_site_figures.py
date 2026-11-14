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

def vsg(root):
    plots = root / "figures"
    with np.load(root / "results/frequency_responses.npz", allow_pickle=False) as data:
        fig, chart = plt.subplots(2, 2, figsize=(11, 6.8), sharex="col", layout="constrained")
        modes = (("traditional", "Traditional"), ("power_only", "Power only"),
                 ("acceleration_only", "Acceleration only"), ("combined", "Combined"))
        inductance = ((key, f"{key} mH") for key in ("0.5", "1.0", "1.5"))
        for i, (number, items, title) in enumerate(((5, modes, "Controller comparison"),
                                                    (6, inductance, "Line-inductance sweep"))):
            for (key, label), color in zip(items, colors):
                value = data[f"fig{number}_{key}"][:, 0]
                chart[0, i].semilogx(data["frequency_hz"], 20 * np.log10(abs(value)), color=color, lw=2.0, label=label)
                chart[1, i].semilogx(data["frequency_hz"], np.angle(value, deg=True), color=color, lw=2.0)
            chart[0, i].set_title(title, loc="left", fontweight="bold")
            chart[0, i].legend(loc="lower left", fontsize=9, frameon=False)
            chart[1, i].set(xlabel="Frequency (Hz)", xlim=(.01, 100))
        chart[0, 0].set_ylabel("Magnitude (dB)")
        chart[1, 0].set_ylabel("Phase (°)")
        finish(fig, chart.flat, plots / "computed_frequency_response.png")
    with np.load(root / "results/pole_loci.npz", allow_pickle=False) as data:
        fig, chart = plt.subplots(1, 2, figsize=(11, 4.6), layout="constrained")
        for tag, label, color in zip(("traditional", "combined"), ("Traditional", "Combined"), colors):
            p = data[f"fig8_{tag}"][:, :3].ravel()
            chart[0].scatter(p.real, p.imag, label=label, color=color, s=22, alpha=.8)
        for cutoff, color in zip(("30.0", "50.0", "70.0"), colors):
            p = data[f"fig9_dense_{cutoff}"].ravel()
            chart[1].scatter(p.real, p.imag, label=f"k₁₂ = {float(cutoff):g}", color=color, s=7, alpha=.6)
        chart[0].set(xlim=(-25, 1), ylim=(-40, 40))
        chart[1].set(xlim=(-46, 1), ylim=(-27, 27))
        for axis, title in zip(chart, ("Inertia sweep", "Acceleration-feedback sweep")):
            axis.set_title(title, loc="left", fontweight="bold")
            axis.set(xlabel="Real part (s⁻¹)", ylabel="Imaginary part (s⁻¹)")
            axis.legend(loc="lower left", frameon=False)
            axis.axvline(0, color="#607067", lw=1, ls="--")
        finish(fig, chart, plots / "computed_pole_loci.png")

if __name__ == "__main__":
    vsg(Path(__file__).resolve().parents[1])
