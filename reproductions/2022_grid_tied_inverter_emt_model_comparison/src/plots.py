"""Publication-style plots for the simulated small system."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


PALETTE = {"event": "#222c3d", "sw2": "#1a6e98", "vi10": "#c34820",
           "sw10": "#687688", "av100": "#377f58"}
LABELS = {"event": "Exact switching", "sw2": "SW 2 µs", "vi10": "VI 10 µs",
          "sw10": "SW 10 µs", "av100": "AV 100 µs"}


def plot_setup():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11,
        "axes.titleweight": "bold", "axes.grid": True, "grid.alpha": 0.24,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "figure.facecolor": "white", "savefig.dpi": 170,
    })


def bandwave(value, time, carrier):
    harmonics = np.fft.rfft(value)
    freq = np.fft.rfftfreq(len(value), time[1] - time[0])
    harmonics[abs(freq - carrier) > 250 + 1e-9] = 0
    return np.fft.irfft(harmonics, n=len(value))


def make_figures(metrics, waves, directory: Path):
    plot_setup()
    output = []

    def save(fig, name):
        path = directory / name
        fig.savefig(path, bbox_inches="tight", pad_inches=0.16)
        plt.close(fig)
        output.append(path)

    fig, ax = plt.subplots(2, 2, figsize=(10.7, 6.0), layout="constrained")
    fig.suptitle("Carrier-sideband response in the CIRED three-phase circuit", fontsize=14, weight="bold")
    for col, (name, carrier) in enumerate((("cable_4500", 4500), ("cable_6000", 6000))):
        ax[0, col].set_title(f"{carrier / 1000:g} kHz carrier")
        for model in ("event", "sw2", "vi10", "sw10", "av100"):
            t = waves[name + "_" + model + "_time_s"]
            i = waves[name + "_" + model + "_i_cable"][:, 0]
            if 0.5 / (t[1] - t[0]) <= carrier + 250:
                continue
            band = bandwave(i, t, carrier)
            w = (t >= 0.003) & (t <= 0.0039)
            styles = {"event": "-", "sw2": "--", "vi10": "-.", "sw10": ":", "av100": "--"}
            ax[0, col].plot(1e3 * t[w], band[w], styles[model], lw=1.5, color=PALETTE[model],
                            label=LABELS[model])
        ax[0, col].set(xlabel="Time (ms)", ylabel="Band-limited cable current $i_{C,a}$ (A)")
        ref_freq = np.arange(carrier - 250, carrier + 251, 50)
        for offset, model in enumerate(("event", "sw2", "vi10", "sw10")):
            t = waves[name + "_" + model + "_time_s"]
            v = waves[name + "_" + model + "_v_line"]
            val = np.sqrt(2) * np.abs(np.fft.rfft(v[:, 0] - v[:, 1])) / len(t)
            ax[1, col].bar(ref_freq + 8 * (offset - 1.5), val[ref_freq // 50], width=7,
                           color=PALETTE[model], label=LABELS[model])
        ax[1, col].set(xlabel="Frequency (Hz)", ylabel="Line voltage $V_{ab}$ (V RMS)")
    handles, labels = ax[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=5, fontsize=9)
    save(fig, "carrier_comparison.png")

    fig, ax = plt.subplots(2, 2, figsize=(10.7, 6.8), layout="constrained")
    fig.suptitle("Step-size and sampling-phase sensitivity", fontsize=14, weight="bold")
    sweep = metrics["timestep_phase_sweep"]
    steps = sorted({r["step_us"] for r in sweep})
    step_ticks = [step for step in steps if step != 5]
    tol = metrics["engineering_harmonic_amplitude_allowance_pct"]
    for row, carrier in enumerate((4500, 6000)):
        cases = [case for case in sweep if case["carrier_hz"] == carrier]
        for axis, key, observable in ((ax[row, 0], "v_line_ab", "line-to-line voltage"),
                                      (ax[row, 1], "i_cable_a", "phase-A cable current")):
            for model, color in (("SW", PALETTE["sw2"]), ("VI", PALETTE["vi10"])):
                samples = np.array([[r[model][key]["amplitude_error_pct"] for r in cases if r["step_us"] == step]
                                    for step in steps])
                axis.fill_between(steps, np.min(samples, axis=1), np.max(samples, axis=1), color=color, alpha=0.20)
                axis.plot(steps, np.median(samples, axis=1), ".-", color=color, lw=1.6, label=model)
            axis.axhspan(-tol, tol, color="#667078", alpha=0.09, label=f"±{tol:g}% band")
            axis.set(title=f"{carrier / 1000:g} kHz {observable}", xlabel="Fixed step (µs)",
                     ylabel="Signed band amplitude error (%)", xticks=step_ticks)
            axis.axhline(0, lw=0.7, color="#777777")
    ax[0, 0].legend(title="Median and phase range", fontsize=8)
    save(fig, "step_phase_sensitivity.png")

    fig, ax = plt.subplots(1, 2, figsize=(10.7, 4.0), layout="constrained")
    fig.suptitle("Network resonance and response to grid-supplied distortion", fontsize=14, weight="bold")
    freq = waves["transfer_hz"]
    for key, color, label in (("cable", PALETTE["vi10"], "1.2 µF cable branch"),
                              ("open", PALETTE["sw2"], "Cable branch open")):
        ax[0].plot(freq / 1e3, waves[key + "_transfer_v_bridge_to_v_line"],
                   color=color, lw=1.7, label=label)
    ax[0].plot(waves["trapezoidal_transfer_hz"] / 1e3, waves["cable_transfer_v_bridge_to_v_line"],
               color="#4f5667", ls="--", lw=1.1, label="Cable branch, TR 10 µs")
    for marker in (4.5, 6):
        ax[0].axvline(marker, color="#777777", ls=":", lw=1)
    ax[0].set(title="Bridge-to-line voltage transfer", xlabel="Frequency (kHz)", ylabel="$|V_{line}/V_{bridge}|$")
    ax[0].legend(fontsize=9)
    low = metrics["grid_fifth_delta"]
    model_names = ("sw2", "vi10", "av100")
    pos = np.arange(len(model_names))
    actual = [low[name]["i_grid_a"]["rms"] for name in model_names]
    ref = low["sw2"]["i_grid_a"]["reference_rms"]
    bars = ax[1].bar(pos, actual, color=[PALETTE[m] for m in model_names], width=0.58, zorder=3)
    ax[1].axhline(ref, color=PALETTE["event"], lw=1, ls="--", zorder=4, label="Nodal solution")
    ax[1].bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
    ax[1].set(title="250 Hz grid-current increment", xticks=pos, xticklabels=[LABELS[m] for m in model_names],
              ylabel="Phase-A grid current (A RMS)", ylim=(0, 1.2 * ref))
    ax[1].legend(loc="upper right", fontsize=9)
    save(fig, "network_and_grid_fifth.png")
    return output
