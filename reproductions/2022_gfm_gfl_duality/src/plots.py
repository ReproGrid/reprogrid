"""Scientific figures for the single-inverter reproductions."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


COLORS=("#0072BD","#D95319","#EDB120","#7E2F8E","#77AC30","#4DBEEE","#A2142F")


def make_figures(metrics,arrays,directory):
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.grid":True,"grid.alpha":0.22,
                         "axes.spines.top":False,"axes.spines.right":False,"savefig.dpi":180})
    paths=[]
    def save(fig,name):
        path=directory/name
        fig.savefig(path,bbox_inches="tight",pad_inches=0.16)
        plt.close(fig)
        paths.append(path)
    fig,axes=plt.subplots(3,2,figsize=(10,10),layout="constrained")
    fig.suptitle("Single-inverter stability under the published parameter sweeps",fontsize=14,weight="bold")
    for ax,locus in zip(axes.flat,metrics["loci"],strict=True):
        for i,sample in enumerate(locus["samples"]):
            modes=sample["all_modes"]
            for mode in modes:
                if abs(mode["imag_hz"])<45 and mode["real_per_s"]/(2*np.pi)>-25:
                    ax.plot(mode["real_per_s"]/(2*np.pi),mode["imag_hz"],"x",c=COLORS[i],ms=7,mew=1.5)
            marker=sample["author_marker"]
            if marker:
                ax.plot(marker["real_hz"],marker["imag_hz"],"o",mfc="none",mec=COLORS[i],ms=10,mew=0.8)
        ax.axvline(0,c="#555555",lw=0.8)
        ax.set(title=f"Fig. {locus['figure'].split(',')[0]}) · {locus['key']}",
               xlabel="Real part / 2π (Hz)",ylabel="Imaginary part / 2π (Hz)",ylim=(-40,40),
               xlim=(-20,10) if locus["kind"]=="GFM" else (-10,5))
    axes[0,0].plot([],[],"kx",label="Computed roots")
    axes[0,0].plot([],[],"ko",mfc="none",label="Author dominant marker")
    axes[0,0].legend(fontsize=8,loc="lower right")
    save(fig,"six_root_loci.png")

    fig,axes=plt.subplots(3,2,figsize=(11,8),layout="constrained")
    fig.suptitle("Published trajectories and fixed-parameter sampled reconstruction",fontsize=14,weight="bold")
    for ax,locus in zip(axes.flat,metrics["loci"],strict=True):
        key=locus["key"];t=arrays[key+"_time_s"]-1.8
        ax.axvspan(.2,.4,color="#c88738",alpha=.1)
        ax.plot(t,50*(arrays[key+"_author_w_pu"]-1),c="#222e44",lw=1.3,label="Author MAT data")
        ax.plot(t,50*(arrays[key+"_model_w_display_pu"]-1),c="#c24b29",lw=1.1,label="TR reconstruction")
        ax.plot(t,50*(arrays[key+"_be_event_w_display_pu"]-1),c="#2c7e81",lw=1.0,ls="--",label="BE at switching events")
        number=locus['figure'].split('(')[0]
        panel=locus['figure'].split(',')[-1].strip(')')
        ax.set(title=f"Fig. {number}({panel}) · {key}",
               xlabel="Paper time (s)",ylabel="Frequency deviation (Hz)",xlim=(0,.8))
    axes[0,0].legend(fontsize=8,loc="upper left")
    save(fig,"six_trajectory_comparisons.png")

    fig,axes=plt.subplots(1,2,figsize=(11,4),layout="constrained")
    fig.suptitle("Low-frequency modes and sampled electrical dynamics",fontsize=14,weight="bold")
    keys=[locus["key"] for locus in metrics["loci"]]
    for method,color in (("trapezoidal","#1a6d96"),("exact","#c24b29")):
        modes=[next(case for case in metrics["time_cases"][key]["sampled_dynamics"]
                    if case["phase"]=="baseline" and case["integrator"]==method and case["step_s"]==1e-4)
               for key in keys]
        values=[m["dominant"]["real_per_s"] for m in modes]
        offset=-0.17 if method=="trapezoidal" else .17
        axes[0].bar(np.arange(6)+offset,values,width=.32,color=color,label=method)
    axes[0].axhline(0,c="#555555",lw=.8)
    axes[0].set(xticks=np.arange(6),xticklabels=[k.replace("_","\n") for k in keys],
                ylabel="Largest oscillatory-mode real part (s⁻¹)",title="100 µs update at the initial operating point")
    axes[0].legend(fontsize=9)
    for i,key in enumerate(keys):
        original=metrics["time_cases"][key]["author_fitted_unstable_mode"]
        analytic=metrics["loci"][i]["samples"][-1]["dominant"]
        axes[1].plot(i,analytic["imag_hz"],"o",c="#1a6d96",label="Analytic changed-mode frequency" if i==0 else None)
        if original:
            axes[1].plot(i,original["imag_hz"],"x",ms=8,c="#c24b29",label="Frequency estimated from author data" if i==0 else None)
    axes[1].set(xticks=np.arange(6),xticklabels=[k.replace("_","\n") for k in keys],ylabel="Oscillation frequency (Hz)",title="Original source data after the parameter change")
    axes[1].legend(fontsize=8)
    save(fig,"sampling_and_mode_comparison.png")

    fig,axes=plt.subplots(1,2,figsize=(11,4),layout="constrained")
    fig.suptitle("Synchronization-loop replay from recorded measurement inputs",fontsize=14,weight="bold")
    key="GFM_m"
    t=arrays[key+"_time_s"]-1.8
    axes[0].plot(t,50*(arrays[key+"_author_w_pu"]-1),c="#222e44",lw=1.6,label="Author frequency")
    axes[0].plot(t,50*(arrays[key+"_recorded_input_loop_w_pu"]-1),c="#c24b29",lw=1.1,ls="--",label="Recorded-input droop replay")
    axes[0].set(xlabel="Paper time (s)",ylabel="Frequency deviation (Hz)",title="Fig. 9(c), measured grid current drives droop")
    axes[0].legend(fontsize=8)
    for i,key in enumerate(keys):
        t=arrays[key+"_time_s"]-1.8
        err=50*(arrays[key+"_recorded_input_loop_w_pu"]-arrays[key+"_author_w_pu"])
        axes[1].plot(t,1e9*err,label=key,lw=1,color=COLORS[i])
    axes[1].set(xlabel="Paper time (s)",ylabel="Replayed minus recorded frequency (nHz)",title="All six source trajectories")
    axes[1].legend(fontsize=7,ncols=2)
    save(fig,"recorded_input_control_replay.png")
    return paths
