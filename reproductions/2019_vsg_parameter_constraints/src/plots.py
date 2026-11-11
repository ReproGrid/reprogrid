"""Scientific plots of the independent model and numeric paper readbacks."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot(directory, omega, time, cases, responses, waveforms, source, step_start):
    plt.rcParams.update({"axes.grid": True, "grid.alpha": .22, "axes.spines.top": False,
                         "axes.spines.right": False, "font.size": 10, "savefig.dpi": 160})
    paths = []

    def save(fig, name):
        path = directory / name
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

    fig, axs = plt.subplots(2, 3, figsize=(12.2, 5.7), sharex="col", layout="constrained")
    color = {"full": "#a43b1b", "reduced": "#176480"}
    tags = ("case1", "case2_inferred_high_x", "case3")
    titles = ("Case 1 · printed parameters", "Case 2 · inferred reactance", "Case 3 · printed parameters")
    for col, (tag, title) in enumerate(zip(tags, titles, strict=True)):
        figure = cases[tag]["source_figure"]
        left_limit = 10 if figure == 3 else 1
        selected = (omega >= left_limit) & (omega <= 1e4)
        for name in color:
            curve = responses[tag][name]
            axs[0, col].semilogx(omega[selected], 20 * np.log10(abs(curve[selected])),
                                 c=color[name], label=f"{name.capitalize()} model")
            phase = np.degrees(np.unwrap(np.angle(curve)))
            axs[1, col].semilogx(omega[selected], phase[selected], c=color[name])
            for row, axis in enumerate(("magnitude_db", "phase_deg")):
                w, val = source[figure, axis, name]
                show = w <= 1e4
                axs[row, col].semilogx(w[show], val[show], ".", c=color[name], ms=2.4, alpha=.7)
        axs[0, col].axhline(0., c=".4", lw=.7, ls=":")
        axs[1, col].axhline(-180., c=".4", lw=.7, ls=":")
        axs[0, col].set_title(title)
        axs[1, col].set_xlabel("Frequency (rad/s)")
    axs[0, 0].set_ylabel("Open-loop magnitude (dB)")
    axs[1, 0].set_ylabel("Unwrapped phase (°)")
    axs[0, 0].plot([], [], ".", color=".35", ms=3, label="UCD manuscript vertices")
    axs[0, 0].legend(loc="lower left", fontsize=8)
    save(fig, "bode_validation.png")

    fig, axs = plt.subplots(1, 2, figsize=(10.2, 3.9), sharex=True, layout="constrained")
    for ax, tag in zip(axs, ("case1", "case3"), strict=True):
        t, p = source[5, "power_step_mw", tag]
        ax.plot(t, p, color="#aa4229", lw=1.2, label="Fig. 5 EMT curve")
        ax.plot(time, waveforms[tag]["average_w"] / 1e6, color="#15616d", lw=1.6,
                label="Independent averaged circuit")
        ax.plot(time, waveforms[tag]["linear_w"] / 1e6, color="#34343c", lw=.9, ls="--",
                label="Linearized circuit")
        ax.axhline(.5, color=".5", lw=.7, ls=":")
        ax.axvline(step_start, color=".5", lw=.7, ls=":", label="Inferred command time")
        ax.set(xlabel="Time (s)", title=f"Case {tag[-1]}")
    axs[0].set_ylabel("Active power (MW)")
    axs[1].legend(loc="upper left", fontsize=8)
    save(fig, "step_comparison.png")
    return paths
