"""Generate the scientific figures from the frozen numerical run."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np


COLORS = ("#252a34", "#97612b", "#107869", "#355ba2", "#ac4169")
BRANCH_LABELS = {
    "table_si": "Table I · 48 mΩ · 314 rad/s",
    "table_50hz": "Table I · 48 mΩ · exact 50 Hz",
    "fig4_si": "Figure 4 · 0 Ω · 12 mH",
    "fig4_scr2": "Figure 4 · 0 Ω · exact SCR = 2",
}


def save(figure, folder, name):
    path = folder / name
    figure.savefig(path, dpi=170)
    plt.close(figure)
    return path


def power_angle(folder, metrics, angle, power):
    fig, ax = plt.subplots(figsize=(7.5, 4.4), layout="constrained")
    for row, curve, color in zip(metrics["fig2"], power, COLORS, strict=True):
        ax.plot(angle, curve / 1000, color=color, label=f"R = {row['r_pu']:g}, X = {row['x_pu']:g} p.u.")
    ax.axhline(2, color="#74777f", linestyle=":", label="2 kW reference")
    ax.set(xlabel="Power angle δ (rad)", ylabel="Active power (kW)", title="Power-angle curves with Q–V droop")
    ax.grid(alpha=.2)
    ax.legend(frameon=False, ncol=2)
    return save(fig, folder, "power_angle.png")


def phase_portraits(folder, metrics, figures, traces):
    name = "table_50hz"
    parameters = metrics["parameters"][name]
    gain_unit = parameters["gain_unit"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), layout="constrained")
    ax, bx = axes
    for index, (gain, color) in enumerate(zip(traces["gain_pu"], COLORS, strict=True)):
        angle, speed, lowpass = traces[name][:, :, index].T
        beyond = np.flatnonzero(angle > 2.5)
        stop = beyond[0] + 1 if len(beyond) else len(angle)
        ax.plot(angle[:stop], speed[:stop], color=color, label=f"{gain:g}")
        ref_a = figures["3a", str(int(gain))]
        ax.plot(*ref_a.T, color=color, ls="--", lw=1, alpha=.65)
        if ("3b", str(int(gain))) in figures:
            x2 = -gain * gain_unit * lowpass[:stop]
            bx.plot(speed[:stop], x2, color=color)
            bx.plot(*figures["3b", str(int(gain))].T, color=color, ls="--", lw=1, alpha=.65)
    ax.axvline(parameters["saddle_rad"], color="#69717e", ls=":", lw=1)
    ax.set(xlim=(.5, 2.5), ylim=(-.5, 2.), xlabel="Power angle δ (rad)", ylabel="Speed deviation Δω (rad/s)",
           title="Power angle and speed")
    bx.set(xlim=(-.5, 2), ylim=(-400, 200), xlabel="Speed deviation Δω (rad/s)", ylabel="Filter term x₂ (W)",
           title="Low-pass contribution to damping")
    for axis in axes:
        axis.grid(alpha=.2)
        axis.axhline(0, color="#9a9fa6", lw=.75)
    ax.legend(title="Kₕ (p.u.)", ncol=2, frameon=False)
    bx.legend(handles=[Line2D([], [], color="#434955", label="Recalculation"),
                       Line2D([], [], color="#434955", ls="--", label="Figure 3 readback")], frameon=False)
    fig.suptitle("1.0 → 0.6 p.u. voltage sag · α = 3 rad/s · 48 mΩ, exact 50 Hz")
    return save(fig, folder, "phase_portraits.png")


def stability_maps(folder, figures, maps):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8.5), sharex=True, sharey=True, layout="constrained")
    colors = ("#e8ecf2", "#e5af62", "#18796c")
    palette = ListedColormap(colors)
    norm = BoundaryNorm([-1.5, -.5, .5, 1.5], palette.N)
    outline = figures["4", "stable_fill_2"]
    for ax, (name, label) in zip(axes.flat, BRANCH_LABELS.items(), strict=True):
        ax.pcolormesh(maps["alpha_rad_s"], maps["gain_pu"], maps[name].T, cmap=palette, norm=norm, shading="nearest")
        ax.plot(*outline.T, color="#252a34", ls="--", lw=1.15)
        ax.set(xlim=(0, 3.5), ylim=(0, 250), title=label)
        ax.label_outer()
    fig.supxlabel("High-pass cutoff α (rad/s)")
    fig.supylabel("Damping gain Kₕ (p.u.)")
    fig.legend(handles=[Patch(color=colors[2], label="Energy-certified capture"),
                        Patch(color=colors[0], label="Saddle crossed"),
                        Patch(color=colors[1], label="Unresolved by horizon"),
                        Line2D([], [], color="#252a34", ls="--", label="Figure 4 readback")],
               loc="outside upper center", ncol=2, frameon=False)
    return save(fig, folder, "stability_maps.png")


def draw(folder, metrics, figures, traces, maps, angles, power):
    return (power_angle(folder, metrics, angles, power), phase_portraits(folder, metrics, figures, traces),
            stability_maps(folder, figures, maps))
