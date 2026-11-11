"""Scientific figures for the independent impedance-identification benchmark."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from identify import loop_eigenvalues


def draw_figures(directory, cases, curves, frequency, excitation, ts):
    paths = []
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": .22,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 160})

    def save(figure, name):
        path = directory / name
        figure.savefig(path, bbox_inches="tight")
        plt.close(figure)
        paths.append(path)

    fig, (ax_time, ax_freq) = plt.subplots(1, 2, figsize=(10, 3.2), layout="constrained")
    colors = ("#176b87", "#c45522")
    labels = ("d channel · MLBS", "q channel · IRS")
    sample = np.arange(100)
    spectrum = abs(np.fft.rfft(excitation, axis=0))
    f = np.fft.rfftfreq(len(excitation), ts)
    for axis, parity in enumerate((0, 1)):
        ax_time.step(1e3 * ts * sample, excitation[sample, axis] + (1 - axis) * 2.5, where="post",
                     color=colors[axis], label=labels[axis])
        selected = (f > 0.) & (f <= 30.) & (np.arange(len(f)) % 2 == parity)
        ax_freq.vlines(f[selected], 0., spectrum[selected, axis] / spectrum[1:, axis].max(),
                       color=colors[axis], linewidth=1.3, label=labels[axis])
    ax_time.set(xlabel="Time in first frame (ms)", ylabel="Binary chip level", yticks=[-1, 1, 1.5, 3.5],
                yticklabels=["−1", "+1", "−1", "+1"])
    ax_time.legend(loc="center right", fontsize=8)
    ax_freq.set(xlabel="dq frequency (Hz)", ylabel="Normalized DFT magnitude", ylim=(0., 1.09))
    ax_freq.legend(fontsize=8)
    save(fig, "orthogonal_excitation.png")

    fig, (ax_g, ax_t) = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained", sharey=True)
    colors = plt.colormaps["tab10"].colors
    for row, color in zip(cases, colors):
        tag = row["id"]
        for ax, key in ((ax_g, "grid_relative_error"), (ax_t, "source_relative_error")):
            ax.loglog(frequency, 100 * curves[tag][key], color=color, lw=1.1, label=tag)
    ax_g.set(title="Passive network at the PCC", ylabel="Relative matrix error vs. Eq. (9) (%)")
    ax_t.set_title("GFM twin output impedance")
    for ax in (ax_g, ax_t):
        ax.set_xlabel("dq frequency (Hz)")
    ax_g.legend(fontsize=8, ncols=2, loc="best")
    save(fig, "identification_error.png")

    tag = "RLC_1600uF"
    data = curves[tag]
    fig, axs = plt.subplots(2, 2, figsize=(10, 6.0), layout="constrained", sharex=True)
    for column, (port, name) in enumerate((("grid", "Passive RLC network"), ("source", "GFM twin"))):
        for (row, index), color, component in zip(((0, 0), (0, 1)), colors, ("dd", "dq")):
            expected = data[f"{port}_eq9"][:, row, index]
            identified = data[f"{port}_eq10"][:, row, index]
            axs[0, column].loglog(frequency, abs(expected), c=color, label=f"{component}, Eq. (9)")
            axs[0, column].loglog(frequency[::16], abs(identified[::16]), ".", c=color, ms=3,
                                 label=f"{component}, Eqs. (10)–(11)")
            axs[1, column].semilogx(frequency, np.unwrap(np.angle(expected)) * 180 / np.pi, c=color)
            phase = np.unwrap(np.angle(identified)) * 180 / np.pi
            axs[1, column].semilogx(frequency[::16], phase[::16], ".", c=color, ms=3)
        axs[0, column].set_title(name)
        axs[0, column].legend(fontsize=8)
        axs[1, column].set_xlabel("dq frequency (Hz)")
    axs[0, 0].set_ylabel("Magnitude (Ω)")
    axs[1, 0].set_ylabel("Unwrapped phase (°)")
    fig.suptitle("SCR 3, 1600 µF shunt · averaged benchmark")
    save(fig, "impedance_bode.png")

    fig, axs = plt.subplots(1, 2, figsize=(10, 4.0), layout="constrained")
    for ax, tag in zip(axs, ("RL_SCR2", "RLC_1600uF")):
        data = curves[tag]
        continuous = loop_eigenvalues(data["loop_continuous"])
        estimated = loop_eigenvalues(data["loop_eq10"])
        ax.plot(continuous.real, continuous.imag, c="#26607e", lw=1.25)
        ax.plot(estimated[::9].real, estimated[::9].imag, ".", c="#c45522", ms=2.5)
        ax.plot(-1, 0, "kx", ms=7, mew=1.6)
        ax.plot([], [], "-", c="#26607e", label="Continuous circuit")
        ax.plot([], [], ".", c="#c45522", label="Identified sampled record")
        ax.set(title=tag, xlabel="Real part", ylabel="Imaginary part", aspect="equal")
        ax.legend(fontsize=8)
    fig.suptitle("Return-ratio eigenvalues · positive-frequency band 10–1000 Hz")
    save(fig, "nyquist_band.png")
    return paths
