"""Draw the single-converter voltage-sag comparisons."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def render(metrics, traces, cfg, folder):
    cond = cfg["literature_conditions"]
    fig, panel = plt.subplots(2,2,figsize=(12,8),layout="constrained")
    for grid, color in (("0.80","C0"),("0.60","C1")):
        for mode, line, label in (("finite_avr","-","finite AVR"),("algebraic_avr","--","static AVR"),
                                  ("fixed_voltage",":","held voltage")):
            t, angle, omega, voltage, _ = traces[f"fig6_{grid}_{mode}"].T
            panel[0,0].plot(angle,omega,line,color=color,label=f"{grid} pu {label}",lw=1.6)
            show = t <= 5.1
            if mode != "fixed_voltage":
                panel[0,1].plot(t[show],voltage[show],line,color=color,lw=1.6,label=f"{grid} pu {label}")
    panel[0,0].set(xlim=(.48,2.51),ylim=(.9945,1.0145),xlabel="Power angle (rad)",ylabel="Frequency (pu)",
                   title="Fig. 6(a) | Conventional VSG")
    panel[0,0].legend(fontsize=8,ncols=2)
    panel[0,1].set(xlim=(-.25,5.05),xlabel="Seconds from sag",ylabel="Internal voltage (pu)",
                   title="Fig. 6(b) | Calculated internal voltage")
    panel[0,1].legend(fontsize=8)
    a,p0,p1,p_nominal = traces["fig8_power_angle"].T
    panel[1,0].plot(a,p_nominal,color=".4",ls="--",label="Before sag, k = 0")
    panel[1,0].plot(a,p0,label="During sag, k = 0")
    panel[1,0].plot(a,p1,label="During sag, k = 1")
    stable = metrics["fig9"]["0.00"]["sep_rad"]
    saddle = metrics["fig9"]["0.00"]["uep_rad"]
    panel[1,0].fill_between(a,p0,p1,where=(a>=metrics["prefault_state"][0])&(a<=stable),alpha=.18,color="C1")
    panel[1,0].fill_between(a,p0,p1,where=(a>=stable)&(a<=saddle),alpha=.16,color="C2")
    panel[1,0].axhline(cfg["table_i"]["active_reference_pu"],color=".35",ls=":",label="Active-power reference")
    panel[1,0].legend(fontsize=8)
    panel[1,0].set(xlabel="Power angle (rad)",ylabel="Electrical power (pu)",title="Fig. 8 | Algebraic AVR, governor omitted")
    for j,(key,row) in enumerate(metrics["fig9"].items()):
        _,a,w,*_ = traces[f"fig9_{key}"].T
        panel[1,1].plot(a,w,color=f"C{j}",label=f"k = {key}")
    saddle = metrics["fig9"]["0.00"]["uep_rad"]
    panel[1,1].axvline(saddle,color=".35",ls=":",label="Calculated saddle")
    panel[1,1].set(xlim=(.49,2.51),ylim=(.9968,1.0092),xlabel="Power angle (rad)",ylabel="Frequency (pu)",
                   title="Fig. 9 | Independent model replays")
    panel[1,1].legend(fontsize=8)
    for ax in panel.flat:
        ax.grid(alpha=.18)
    fig.savefig(folder/"vsg_voltage_and_phase.png",dpi=160)
    plt.close(fig)

    fig,(a,b) = plt.subplots(2,1,figsize=(9,6.3),sharex=True,layout="constrained",
                             gridspec_kw={"height_ratios":[1,2]})
    for key,label in (("table_i","Table I, finite AVR"),("text_scr","Text SCR 1.9, finite AVR"),
                      ("instantaneous_avr","Table I, instantaneous AVR")):
        k,_,peak = traces[f"gains_{key}"].T
        row = metrics["gain_search"][key]
        lo,hi = row["selected_gain_grid_pu"]
        index = len(a.lines)
        a.plot([lo,hi],[index,index],"o-",lw=2,label=label)
        b.plot(k,peak,label=label)
    a.set_yticks(range(3),["Table I","SCR 1.9","Static AVR"])
    a.axvspan(*cond["report_k_limits_pu"],color="C3",alpha=.12,label="Paper interval")
    a.legend(loc="upper right",fontsize=8)
    a.set_title("Admissible 0.01-pu gains, with angle and voltage constraints")
    b.axhline(cond["voltage_headroom_limit_pu"],color=".35",ls=":",label="1.20 pu voltage budget")
    b.set(xlabel="Acceleration gain k (pu)",ylabel="Peak internal voltage (pu)")
    b.legend(fontsize=8)
    a.grid(alpha=.18)
    b.grid(alpha=.18)
    fig.savefig(folder/"gain_headroom.png",dpi=160)
    plt.close(fig)
    return ("vsg_voltage_and_phase.png","gain_headroom.png")
