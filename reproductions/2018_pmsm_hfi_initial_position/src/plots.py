"""Generate scientific plots for the reproduced observer and detector results."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {"PI": "#27637a", "ESO_C1": "#367a4b", "ESO_C2": "#ae4225"}


def draw(directory, frequency, responses, detector, normalization, traces, source):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": .22, "savefig.dpi": 160})
    paths = []

    def save(fig, name):
        path = directory / name
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

    fig, ax = plt.subplots(2, 1, figsize=(8.4, 5.8), sharex=True, layout="constrained")
    for kind, h in responses.items():
        color = COLORS[kind]
        ax[0].semilogx(frequency, 20 * np.log10(abs(h)), color=color, lw=1.6, label=kind.replace("_", " "))
        ax[1].semilogx(frequency, np.degrees(np.angle(h)), color=color, lw=1.6)
        for row, axis in enumerate(("gain_db", "phase_deg")):
            x, y = source[8, axis, kind]
            ax[row].semilogx(x[::3], y[::3], ".", color=color, alpha=.6, ms=3)
    ax[0].axhline(-3.0103, color=".45", ls=":", lw=.7)
    ax[0].plot([], [], ".", c=".45", label="Fig. 8 vector points")
    ax[0].set(ylabel="Gain (dB)", title="Position observers at ζ = 5")
    ax[0].legend(ncols=2, fontsize=8)
    ax[1].set(xlabel="Angular frequency (rad/s)", ylabel="Phase (°)")
    save(fig, "observer_bode.png")

    fig, (ax, near) = plt.subplots(1, 2, figsize=(10.3, 3.65), layout="constrained")
    angles = detector["angle_deg"]
    threshold = normalization["paper_normalized_detector_threshold"]
    for area in (ax, near):
        area.plot(angles, detector["left"], color="#27637a", lw=1.5, label="Current norm in Eq. (15)")
        area.plot(angles, detector["right"], color="#ae4225", lw=1.4, label="Printed expression")
        area.axhline(threshold, c=".45", lw=.8, ls=":")
        area.axhline(-threshold, c=".45", lw=.8, ls=":")
        area.set_xlabel("Electrical angle error (°)")
    ax.set(xlim=(-90, 90), ylabel="Detector output", title="d-axis and q-axis zeros")
    ax.legend(fontsize=8)
    near.set(xlim=(0, 7), ylim=(0, .19), title="Near the d-axis")
    for tag, color in (("left", "#27637a"), ("right", "#ae4225")):
        x = normalization[f"ideal_near_d_axis_limit_from_{tag}_deg"]
        near.plot(x, threshold, "o", c=color, ms=5)
        near.annotate(f"{x:.2f}°", (x, threshold), xytext=(5, -13 if tag == "left" else 5),
                      textcoords="offset points", color=color)
    save(fig, "detector_normalization.png")

    fig, (lock, sweep) = plt.subplots(1, 2, figsize=(11.0, 3.9), layout="constrained")
    for tag, label, color, style in (
        ("pi_zero_seed", "PI with zero initial states", "#606060", "--"),
        ("pi_averaged", "PI with speed seed", COLORS["PI"], "-"),
        ("eso_averaged", "ESO C2 with speed seed", COLORS["ESO_C2"], "-"),
    ):
        run = traces[tag]
        q = np.flatnonzero(run["angle_deg"] == 90.)[0]
        axis_error = abs(run["axis_error_deg"][:, q])
        lock.plot(1e3 * run["time_s"], axis_error, label=label, color=color, ls=style)
    t, angles = source[9, "angle_deg", "estimated_rotor"]
    error = abs((angles - 90 + 90) % 180 - 90)
    lock.plot(1e3 * t[::10], error[::10], ".", c="#191919", ms=2.5, label="Fig. 9 vector points")
    lock.set(xlim=(0, 85), ylim=(-1, 95), xlabel="Time (ms)", ylabel="Error modulo 180° (electrical degrees)",
             title="Rotor fixed at 90°")
    lock.legend(fontsize=8)
    for tag, label, color in (("pi", "PI", COLORS["PI"]), ("eso", "ESO C2", COLORS["ESO_C2"])):
        averaged, switched = traces[f"{tag}_averaged"], traces[f"{tag}_switched"]
        sweep.plot(averaged["angle_deg"], 1e3 * averaged["joint_time_s"], c=color, label=f"{label}, averaged")
        sweep.plot(switched["angle_deg"], 1e3 * switched["joint_time_s"], ".", c=color, ms=4,
                   label=f"{label}, switched")
    sweep.set(xlabel="Rotor initial position (electrical degrees)", ylabel="Detector and angle convergence (ms)",
              title="Standstill sweep with speed seed")
    sweep.legend(fontsize=8)
    save(fig, "pulse_lock_and_convergence.png")
    return paths
