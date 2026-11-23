"""Build comparisons for the section III and IV reproduction."""

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np


BLUE = "#176783"
PURPLE = "#725197"
RED = "#ae4938"
GREEN = "#22836a"
DARK = "#2d414d"
STYLE = {"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": .22,
         "axes.labelcolor": DARK, "text.color": DARK, "axes.titleweight": "bold", "font.size": 10, "savefig.dpi": 160}


def phase_plot(metrics, arrays, path):
    fig, axes = plt.subplots(1, 2, figsize=(10.9, 4.3), layout="constrained", sharey=True)
    key_colours = (("case_i", PURPLE, "Case I, ζ = 0.5"),
                   ("case_ii", BLUE, "Case II, ζ = 1.5"),
                   ("case_iii_p_mode", RED, "Case III, P mode"))
    for ax, voltage in zip(axes, metrics["phase_portraits"], strict=True):
        for key, colour, label in key_colours:
            d = arrays[f"fig13_{voltage}_{key}"]
            ax.plot(d[:,1],d[:,2],color=colour,lw=1.8,label=label)
        boundary = metrics["phase_portraits"][voltage]["equilibrium"]["lower_separatrix_rad"]
        ax.axvline(boundary,c=DARK,lw=.8,ls=":")
        ax.axhline(0,c=DARK,lw=.7,alpha=.4)
        ax.set(xlim=(-np.pi,np.pi/2),ylim=(-20,10),title=f"Grid residual voltage {voltage} pu", xlabel="PLL-grid phase difference (rad)")
    axes[0].set_ylabel("Frequency deviation (Hz)")
    axes[1].legend(fontsize=8,loc="upper left")
    fig.savefig(path)
    plt.close(fig)


def critical_plot(metrics, arrays, path):
    fig, ax = plt.subplots(figsize=(7.8, 4.25), layout="constrained")
    baseline = arrays["fig9"]
    ax.plot(baseline[:,0],baseline[:,1],"o-",lw=1.7,ms=3.5,color=BLUE,label="Tables I–II, saddle manifold")
    d = metrics["critical_damping"]
    ax.scatter([d["ratio"]],[d["article_text_zeta"]],marker="D",color=RED,s=25,label="Threshold reported in text",zorder=4)
    ax.axvline(d["ratio"],c=DARK,lw=.8,ls=":")
    ax.scatter([d["ratio"]],[d["zeta"]],c=BLUE,s=33,zorder=4)
    ax.annotate(f"ζ = {d['zeta']:.3f} at 0.14 pu",(d["ratio"],d["zeta"]),xytext=(-9,47),textcoords="offset points",
                arrowprops={"arrowstyle":"-","color":BLUE},ha="left")
    ax.set(xlabel="Voltage drop ratio |v_zq| / V_fault",ylabel="Critical nominal damping ratio ζ",xlim=(.596,1.001),ylim=(-.09,8.4),
           title="Critical damping for the chosen initialization")
    ax.legend(loc="upper left",fontsize=9)
    fig.savefig(path)
    plt.close(fig)


def hybrid_plot(metrics, arrays, path):
    fig, axes = plt.subplots(2,2,figsize=(10.9,7.2),layout="constrained")
    phase, frequency, rocof, duration = axes.ravel()
    cases = (("case_iii_fig12_order",BLUE,"Fig. 12 block order"),
             ("adaptive_abs_rocof_first",RED,"Filter on |df/dt|"), ("p_mode_all_the_time",GREEN,"P held"))
    for name, colour, label in cases:
        tr = arrays[f"sag_0.10_{name}"]
        phase.plot(tr[:,0],tr[:,1],c=colour,lw=1.65,label=label)
        frequency.plot(tr[:,0],tr[:,2],c=colour,lw=1.65,label=label)
        if name != "p_mode_all_the_time":
            rocof.plot(tr[:,0],tr[:,6],c=colour,lw=1.45,label=label)
    for ax in (phase,frequency,rocof):
        ax.axvspan(0.,.6,color="#dfe7eb",alpha=.45,zorder=-2)
        ax.set_xlabel("Time from sag onset (s)")
    phase.axhline(-np.pi/2,color=DARK,lw=1,ls=":",label="Fault-equilibrium fold")
    phase.set(ylabel="Phase difference (rad)",title="0.10 pu sag, published 600 ms duration",xlim=(0,.9))
    phase.legend(loc="lower right",fontsize=8)
    frequency.set(ylabel="PLL output frequency (Hz)",title="Frequency through clearing",xlim=(0,.9),ylim=(44.5,55.5))
    rocof.axhline(.5,c=DARK,lw=1,ls=":",label="Switch-off threshold")
    rocof.axhline(5,c=DARK,lw=1,ls="--",label="Switch-on threshold")
    rocof.set(ylabel="Filtered rate (Hz/s)",title="Order of abs, derivative and LPF",xlim=(0,.6),ylim=(-.12,10.))
    rocof.legend(loc="upper right",fontsize=8)
    first = metrics["sampled_replays"]["duration_sensitivity_at_fold"][0]["duration_s"]
    for key,colour,label in (("fig12",BLUE,"Fig. 12 block order"),("abs_first",RED,"Filter on |df/dt|"),("p_only",GREEN,"P held")):
        tr = arrays[f"long_{first}_{key}"]
        duration.plot(tr[:,0],tr[:,1],c=colour,lw=1.55,label=label)
    duration.axhline(-np.pi/2,c=DARK,ls=":",lw=1.)
    duration.axvline(first,c=DARK,ls="--",lw=1.)
    duration.set(xlabel="Time from sag onset (s)",ylabel="Unwrapped phase (rad)",title=f"Separate {first:g} s fault sensitivity",xlim=(0,first+.18))
    duration.legend(loc="lower left",fontsize=8)
    fig.savefig(path)
    plt.close(fig)


def render(metrics, arrays, figures):
    files = ["critical_damping.png","phase_portrait_replay.png","adaptive_switching.png"]
    with plt.rc_context(STYLE):
        critical_plot(metrics,arrays,figures/files[0])
        phase_plot(metrics,arrays,figures/files[1])
        hybrid_plot(metrics,arrays,figures/files[2])
    return files
