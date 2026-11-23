"""Draw the power-angle, phase-portrait and derating comparisons."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def render(metrics,traces,cfg,folder):
    fig,panel = plt.subplots(2,2,figsize=(12,8),layout="constrained")
    for kind,ax,title in (("resistance",panel[0,0],"Fig. 4(a) | Real and virtual resistance"),
                           ("grid_voltage",panel[0,1],"Fig. 4(b) | Fixed power through deeper sags")):
        for j,row in metrics["fig4"][kind].items():
            angle,power = traces[f"fig4_{kind}_{j}"].T
            label = f"Rg {row['rg_pu']:g}, Rv {row['rv_pu']:g}" if kind=="resistance" else f"Vg = {row['grid_pu']:g} pu"
            ax.plot(angle,power,label=label)
        ax.axhline(cfg["table_i"]["p_ref_pu"],ls=":",color=".4",label="Power reference")
        ax.set(xlabel="Virtual angle (rad)",ylabel="Physical PCC active power (pu)",title=title)

    for j,row in metrics["fig5"]["table_si"].items():
        _,a,w,*_ = traces[f"fig5_table_si_{j}"].T
        panel[1,0].plot(a,w,label=f"Rg {row['rg_pu']:g}, Rv {row['rv_pu']:g}",lw=1.7)
    panel[1,0].set(title="Fig. 5 | No sag power derating")
    for j,row in metrics["fig10"]["table_si"].items():
        _,a,w,*_ = traces[f"fig10_table_si_{j}"].T
        color = f"C{int(j)-1}"
        panel[1,1].plot(a,w,color=color,label=f"Case {j}, K = {row['gain_number_interpreted_as_w_per_v']:g}",lw=1.7)
    panel[1,1].set(title="Fig. 10 | Independent model replays")
    for ax in panel[1]:
        ax.set(xlim=(.45,2.4),ylim=(-.75,2.85),xlabel="Virtual angle (rad)",ylabel="Speed deviation (rad/s)")
    for ax in panel.flat:
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.savefig(folder/"resistance_power_and_phase.png",dpi=160)
    plt.close(fig)

    fig,panels = plt.subplots(1,2,figsize=(11,4.5),sharey=True,layout="constrained")
    for ax,rg in zip(panels,("0.000","0.003")):
        for variant,line,label in (("table_si","o-","314 rad/s and 12 mH"),("printed_pu","s--","Printed X = 0.5 pu"),
                                    ("exact_50hz","^-.","50 Hz and 12 mH")):
            rv,lo,hi = traces[f"fig11_{variant}_rg{rg}"].T
            ax.plot(rv,(lo+hi)/2,line,label=label,lw=1.5,ms=4)
        ax.set(title=f"Real grid resistance = {rg} pu",xlabel="Virtual resistance (pu)")
        ax.grid(alpha=.2)
    panels[0].set_ylabel("Critical derating gain (W/V)")
    panels[1].legend(fontsize=8)
    fig.suptitle("Fig. 11 | Minimum gain against first-swing loss of synchronism")
    fig.savefig(folder/"vsg_derating_boundary.png",dpi=160)
    plt.close(fig)
    return ("resistance_power_and_phase.png","vsg_derating_boundary.png")
