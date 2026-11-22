"""Plot quantitative comparisons and the sensitivity to disclosed assumptions."""

from matplotlib import pyplot as plt
import numpy as np


BLUE = "#206781"
ORANGE = "#b3482e"
GREEN = "#247f6e"
DARK = "#2e404b"
STYLE = {"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": .22,
         "axes.labelcolor": DARK, "text.color": DARK, "axes.titleweight": "bold", "font.size": 10, "savefig.dpi": 175}


def phasor_plot(metrics, arrays, readback, path):
    fig, (ax, bound) = plt.subplots(1, 2, figsize=(10.8, 4), layout="constrained")
    display = (0., .41)
    paper = readback["fig3_fine"]
    plain = arrays["fig3_fine_si_without_cap"]
    cap = arrays["fig3_fine_si_with_cap"]
    for column, colour, name in ((1, BLUE, "1.0 pu grid"), (2, ORANGE, "0.5 pu grid")):
        ax.plot(plain[:, 0], plain[:, column], color=colour, lw=1.75, label=name)
        ax.plot(cap[:, 0], cap[:, column], color=colour, lw=1.25, ls="--")
        ax.scatter(paper[8::48, 0], paper[8::48, column], edgecolors=colour, facecolors="white", s=12, zorder=5)
    ax.axhline(.8, color=DARK, ls=":", lw=1.2)
    ax.set(xlabel="Converter-grid angle (rad)", ylabel="Active power (pu)", title="Fig. 3 around the two equilibria", xlim=display)
    ax.plot([], [], color=DARK, ls="--", label="Capacitor retained")
    ax.scatter([], [], edgecolors=DARK, facecolors="white", s=16, label="Paper vector readback")
    ax.legend(fontsize=8, loc="lower right")
    si = metrics["phasor"]["models"]["si"]
    r = si["voltage_sag_sensitivity"]
    bound.plot([x["sag_pu"] for x in r], [1000*x["same_basin_boundary_s"] for x in r], "o-", color=BLUE)
    bound.axhline(450, linewidth=1, color=ORANGE, ls=":", label="Paper Fig. 5 duration")
    bound.axvline(.5, linewidth=1, color=DARK, ls="--", label="Published 0.50 pu sag")
    bound.set(xlabel="Residual grid voltage in the sag (pu)", ylabel="Original-basin clearing boundary (ms)",
              title="Sensitivity to sag amplitude")
    bound.legend(fontsize=8, loc="upper left")
    fig.savefig(path)
    plt.close(fig)


def time_plot(metrics, arrays, path):
    fig, axs = plt.subplots(2, 2, figsize=(11.6, 6.8), layout="constrained")
    p_fig, a_fig, recovery, current = axs.ravel()
    scope_t, scope_p, scope_a, scope_replay, _ = arrays["scope_fig5_phase_replay"].T
    a = arrays["digital_si_nominal_fig5"]
    b = arrays["digital_si_nominal_fig6"]
    for ax in (p_fig, a_fig):
        ax.axvspan(0, .45, color="#e0e8ec", alpha=.45, zorder=-2)
        ax.set(xlabel="Time from sag onset (s)", xlim=(-.05, .56))
    # Summarize the dense sampled waveform by grid-cycle bins and retain its range.
    sample_stop = np.searchsorted(a[:, 0], .56)
    trip = np.searchsorted(a[:, 0], .45)
    bin_rows = []
    for first, end in ((0, trip), (trip, sample_stop)):
        for left in range(first, end, 200):
            chunk = a[left:min(left+200, end)]
            bin_rows.append((chunk[:, 0].mean(), chunk[:, 2].mean(), chunk[:, 2].min(), chunk[:, 2].max()))
    mean_t, mean_p, minimum, maximum = np.array(bin_rows).T
    p_fig.fill_between(mean_t, minimum, maximum, color=BLUE, alpha=.12, lw=0, label="Model range per 20 ms")
    p_fig.plot(mean_t, mean_p, lw=1.45, color=BLUE, label="20 ms model mean")
    p_fig.plot(scope_t, scope_p, lw=1.25, color=ORANGE, label="Fig. 5 digitized")
    pmax = metrics["phasor"]["models"]["si"]["without_capacitor"]["peak_fault_power_pu"]
    p_fig.hlines(pmax, 0, .45, colors=DARK, linestyles=":", linewidth=1.4, label="Reduced-model sag maximum")
    p_fig.set(ylabel="Terminal active power (pu)", title="450 ms sag and clearing power", ylim=(.4, 1.38))
    p_fig.legend(fontsize=8, loc="upper left")

    a_fig.plot(scope_t, scope_a, lw=1.1, c=ORANGE, label="Fig. 5 measured angle")
    a_fig.plot(scope_t, scope_replay, lw=1.3, c=GREEN, ls="--", label="Droop integral of Fig. 5 power")
    a_fig.plot(a[:, 0], a[:, 1]-a[0, 1], lw=1.1, c=BLUE, label="10 kHz sampled model")
    a_fig.set(ylabel="Angle change (rad)", title="Fixed droop gain with pictured power", ylim=(-.025, .36))
    a_fig.legend(fontsize=8, loc="upper left")

    for trace, color, label in ((a, BLUE, "450 ms sag"), (b, ORANGE, "650 ms sag")):
        recovery.plot(trace[:, 0], trace[:, 1]-trace[0, 1], c=color, lw=1.3, label=label)
    recovery.axhline(2*np.pi, c=DARK, ls=":", lw=1.1, label="Next 2π equilibrium")
    recovery.set(xlabel="Time from sag onset (s)", ylabel="Angle change (rad)", title="Unwrapped sampled-model recovery", xlim=(0, 3.2))
    recovery.legend(fontsize=8, loc="lower right")

    current.plot(1000*a[:, 0], a[:, 4], color=BLUE, lw=1.35, label="Converter current")
    current.step(1000*a[:, 0], a[:, 7], c=ORANGE, lw=1.0, where="post", label="Projected reference")
    current.axhline(1.2, c=DARK, ls=":", lw=1.1)
    current.set(xlabel="Time from sag onset (ms)", ylabel="Current magnitude (pu)", title="Current in the first 10 ms", xlim=(0, 10), ylim=(.65, 1.47))
    current.legend(fontsize=8, loc="lower right")
    fig.savefig(path)
    plt.close(fig)


def render(metrics, arrays, readback, output):
    filenames = ["fig3_curve_and_voltage_sensitivity.png", "fig5_diagnostic_and_sampled_replay.png"]
    with plt.rc_context(STYLE):
        phasor_plot(metrics, arrays, readback, output / filenames[0])
        time_plot(metrics, arrays, output / filenames[1])
    return filenames
