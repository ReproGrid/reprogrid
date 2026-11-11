"""Independent RFPSC reproduction from paper equations; no author code imported.

State convention: converter dq current, converter angle relative to grid,
and the two current-filter states. Time in model equations is tau=omega_base*t.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT.parents[1] / ".cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class Parameters:
    L: float
    Ra: float = 0.2
    V: float = 1.0
    Vg: float = 1.0
    wb: float = 0.1
    R: float = 0.0


def parameters_from_protocol(cfg, L, Ra=None):
    """Bind every physical parameter supported by the implemented model."""
    if cfg["power_scaling_kappa_pu"] != 1.0:
        raise ValueError("This per-unit implementation requires power_scaling_kappa_pu=1")
    return Parameters(L=L,
                      Ra=cfg["active_resistance_pu_primary"] if Ra is None else Ra,
                      V=cfg["voltage_pu"], Vg=cfg["grid_voltage_pu"],
                      wb=cfg["filter_bandwidth_pu"], R=cfg["nonlinear_cases"]["R_grid_pu"])


def coefficients(p: Parameters, i0: complex, xi: int):
    """Eq.19 with R=0 and frozen current filters, in omega1=kappa=1 units."""
    if p.R != 0.0:
        raise ValueError("Eq.19 assumes a purely inductive grid (R=0)")
    alpha = p.Ra / p.L
    a = p.L * i0.imag / p.V
    b = -(p.Ra**2 / p.V) * (i0.imag / p.L + abs(i0)**2 / p.V)
    c = p.Ra * i0.real / p.V
    d = p.Ra**2 * i0.imag / (p.L * p.V)
    q = (1 + a + b) * alpha
    den = np.array([1., (2 + a) * alpha, alpha**2 + 1, q])
    num = np.array([xi*c, (a + xi*(1+c))*alpha,
                    xi*(c+d+alpha**2), q])
    return num, den, {"alpha": alpha, "a": a, "b": b, "c": c, "d": d}


def output(x, pref, p, xi, frozen=None):
    current = x[0] + 1j*x[1]
    h = (x[3] + 1j*x[4]) if frozen is None else frozen
    iref = (pref / p.V if xi else h.real) + 1j*h.imag
    voltage = p.V + p.Ra*(iref-current)
    power = float((voltage*np.conj(current)).real)
    return current, voltage, power


def derivative_tau(x, pref, p, xi, frozen=None):
    current, voltage, power = output(x, pref, p, xi, frozen)
    delta_rate = p.Ra/(p.V**2)*(pref-power)
    # Eq.8 includes the perturbation of the converter frame's angular speed.
    di = (voltage-p.Vg*np.exp(-1j*x[2])-p.R*current)/p.L - 1j*(1+delta_rate)*current
    base = [di.real, di.imag, delta_rate]
    if frozen is not None:
        return np.asarray(base)
    return np.asarray(base + [p.wb*(x[0]-x[3]), p.wb*(x[1]-x[4])])


def operating_point(p, i0):
    # Arbitrary Fig.3 currents imply a corresponding grid voltage magnitude.
    z = p.V - (p.R + 1j*p.L)*i0
    p = Parameters(p.L, p.Ra, p.V, abs(z), p.wb, p.R)
    return p, np.array([i0.real, i0.imag, -np.angle(z)]), p.V*i0.real


def numerical_linearization(p, i0, xi, eps=1e-6):
    p, x0, u0 = operating_point(p, i0)
    f = lambda x,u: derivative_tau(x,u,p,xi,frozen=i0)
    g = lambda x,u: output(x,u,p,xi,frozen=i0)[2]
    eye = np.eye(3)*eps
    A = np.column_stack([(f(x0+e,u0)-f(x0-e,u0))/(2*eps) for e in eye])
    B = ((f(x0,u0+eps)-f(x0,u0-eps))/(2*eps))[:,None]
    C = np.array([[(g(x0+e,u0)-g(x0-e,u0))/(2*eps) for e in eye]])
    D = np.array([[(g(x0,u0+eps)-g(x0,u0-eps))/(2*eps)]])
    return signal.StateSpace(A,B,C,D), float(np.max(np.abs(f(x0,u0))))


def simulate(p, xi, cfg, max_step=None, rtol=None, atol=None):
    schedule = cfg["reference_sequence"]
    solver = cfg["solver"]
    omega = 2*np.pi*cfg["base_frequency_hz"]
    step = solver["sample_interval_s"]
    ts = np.linspace(0, schedule["stop_s"], round(schedule["stop_s"]/step)+1)
    ys = np.empty((len(ts),5))
    refs = np.empty(len(ts))
    state = np.zeros(5)
    starts = schedule["times_s"]
    ends = starts[1:] + [schedule["stop_s"]]
    for k,(start,end,pref) in enumerate(zip(starts,ends,schedule["levels_pu"])):
        sol = solve_ivp(lambda t,y: omega*derivative_tau(y,pref,p,xi), (start,end), state,
                        method=solver["method"], rtol=rtol or solver["rtol"],
                        atol=atol or solver["atol"], max_step=max_step or solver["max_step_s"],
                        dense_output=True)
        if not sol.success:
            raise RuntimeError(sol.message)
        mask = (ts>=start) & ((ts<end) if k<len(starts)-1 else (ts<=end))
        ys[mask] = sol.sol(ts[mask]).T
        refs[mask] = pref
        state = sol.sol(end)
    currents = ys[:,0]+1j*ys[:,1]
    irefs = (refs/p.V if xi else ys[:,3]) + 1j*ys[:,4]
    voltage = p.V + p.Ra*(irefs-currents)
    power = np.real(voltage*np.conj(currents))
    if not np.all(np.isfinite(ys)):
        raise RuntimeError("Nonfinite state; no clipping is used")
    return {"time_s":ts, "reference_pu":refs, "power_pu":power,
            "id_pu":ys[:,0], "iq_pu":ys[:,1], "delta_rad":ys[:,2],
            "voltage_abs_pu":np.abs(voltage), "h_d":ys[:,3], "h_q":ys[:,4]}


def metrics(data,cfg):
    seq=cfg["reference_sequence"]
    items=[]
    for k in range(1,len(seq["times_s"])):
        t0=seq["times_s"][k]
        t1=seq["times_s"][k+1] if k+1<len(seq["times_s"]) else seq["stop_s"]
        previous,target=seq["levels_pu"][k-1:k+1]
        mask=(data["time_s"]>=t0)&(data["time_s"]<t1)
        t=data["time_s"][mask]-t0
        y=data["power_pu"][mask]
        normalized=(y-previous)/(target-previous)
        def crossing(level):
            ids=np.flatnonzero(normalized>=level)
            return float(t[ids[0]]) if len(ids) else None
        t10,t90=crossing(.1),crossing(.9)
        error=np.abs(y-target)
        outside=np.flatnonzero(error>0.02*abs(target-previous))
        settling=float(t[outside[-1]+1]) if len(outside) and outside[-1]+1<len(t) else (0. if not len(outside) else None)
        items.append({"step_time_s":t0,"from_pu":previous,"to_pu":target,
                      "rise_10_90_ms":None if t10 is None or t90 is None else 1000*(t90-t10),
                      "overshoot_percent_of_step":float(max(0.,np.max(normalized)-1)*100),
                      "settling_2pct_ms":None if settling is None else 1000*settling,
                      "end_window_error_pu":float(error[-1]),
                      "integral_abs_error_pu_s":float(np.trapezoid(error,t)),
                      "max_voltage_deviation_pu":float(np.max(np.abs(data["voltage_abs_pu"][mask]-cfg["voltage_pu"])))})
    return items


def analytical_suite(cfg,figures):
    checks=[]
    summary=[]
    p=parameters_from_protocol(cfg,cfg["paper_fig3"]["L_pu"])
    # These fixed axes and the paper-Fig.3 label describe the published case.
    # Each nonzero operating point derives its own Vg in operating_point();
    # cfg.grid_voltage_pu instead belongs to the full time-domain cases.
    if (p.L,p.Ra,p.V,p.R) != (0.5,0.2,1.0,0.0):
        raise ValueError("Paper Fig.3 requires L=0.5, Ra=0.2, V=1 and R=0 p.u.; "
                         "a different case must be labeled as a parameter extension")
    fig,axes=plt.subplots(2,2,figsize=(8.0,6.3),layout="constrained")
    for ax,point in zip(axes.flat,cfg["paper_fig3"]["operating_points_id_iq"]):
        i0=complex(*point)
        num,den,terms=coefficients(p,i0,1)
        zeros=np.roots(np.trim_zeros(num,"f")); poles=np.roots(den)
        ax.scatter(poles.real,poles.imag,marker="x",color="#235caa",label="poles")
        ax.scatter(zeros.real,zeros.imag,marker="o",facecolors="none",edgecolors="#d77716",label="zeros")
        ax.set(xlim=(-.3,-.1),ylim=(.9,1.1),xlabel="Real(s), p.u.",ylabel="Imag(s), p.u.",title=f"id={point[0]:g}, iq={point[1]:g}")
        ax.grid(alpha=.2)
        upper=[z for z in poles if z.imag>.1]
        distances=[float(min(abs(z-q) for q in zeros)) for z in upper]
        row={"id0":point[0],"iq0":point[1],"poles":[[float(z.real),float(z.imag)] for z in poles],
             "zeros":[[float(z.real),float(z.imag)] for z in zeros],"upper_complex_pole_zero_distance":distances,"coefficients":terms}
        for xi in [0,1]:
            tfnum,tfden,_=coefficients(p,i0,xi)
            syslin,residual=numerical_linearization(p,i0,xi)
            frequencies=np.logspace(-3,2,200)
            _,response=signal.freqresp(syslin,w=frequencies)
            analytic=np.polyval(tfnum,1j*frequencies)/np.polyval(tfden,1j*frequencies)
            err=float(np.max(np.abs(response-analytic)))
            residual_tol=cfg["predeclared_checks"]["steady_state_residual"]
            checks.append({"check":"independent_equation_jacobian_vs_eq19","id0":point[0],"iq0":point[1],"xi":xi,
                           "max_transfer_abs_error":err,"tolerance":1e-7,
                           "passed":err<1e-7 and residual<residual_tol,
                           "equilibrium_residual":residual,"equilibrium_residual_tolerance":residual_tol})
        summary.append(row)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("RFPSC: independent reconstruction of paper Fig. 3")
    fig.savefig(figures/"fig3_pole_zero_reconstruction.png",dpi=180);plt.close(fig)
    for L in [.05,.1,.2,.5,1.,2.]:
        p=parameters_from_protocol(cfg,L)
        num,den,_=coefficients(p,0j,1)
        s=1j*np.logspace(-4,3,400)
        response=np.polyval(num,s)/np.polyval(den,s)
        exact=(p.Ra/L)/(s+p.Ra/L)
        error=float(np.max(np.abs(response-exact)))
        tol=cfg["predeclared_checks"]["zero_current_transfer_abs_error"]
        checks.append({"check":"zero_current_exact_cancellation","L_pu":L,"max_error":error,"tolerance":tol,"passed":error<tol})
    return checks,summary


def local_small_signal_check(cfg):
    checks=[]
    omega=2*np.pi*cfg["base_frequency_hz"]
    times=np.linspace(0,.2,2001)
    amplitude=cfg["predeclared_checks"]["small_signal_step_pu"]
    for L in sorted(cfg["nonlinear_cases"]["L_pu"]):
        p=parameters_from_protocol(cfg,L)
        if p.Vg != p.V:
            raise ValueError("The zero-current small-step check requires grid_voltage_pu=voltage_pu")
        for xi in [0,1]:
            num,den,_=coefficients(p,0j,xi)
            sol=solve_ivp(lambda t,x:omega*derivative_tau(x,amplitude,p,xi,frozen=0j),
                          (0,.2),np.zeros(3),t_eval=times,method="DOP853",rtol=1e-10,atol=1e-13,max_step=.0001)
            y=np.array([output(x,amplitude,p,xi,frozen=0j)[2] for x in sol.y.T])/amplitude
            _,linear=signal.step((np.trim_zeros(num,"f"),den),T=times*omega)
            error=float(np.max(np.abs(y-linear)))
            tol=cfg["predeclared_checks"]["local_small_signal_relative_error"]
            checks.append({"check":"small_step_nonlinear_vs_linear_frozen_filters","L_pu":L,"xi":xi,
                           "relative_to_step_max_error":error,"tolerance":tol,"passed":error<tol})
    return checks


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=ROOT/"results")
    args=parser.parse_args()
    outputdir=args.output.resolve();outputdir.mkdir(parents=True,exist_ok=True)
    figures=ROOT/"figures" if outputdir==ROOT/"results" else outputdir/"figures"
    figures.mkdir(exist_ok=True)
    cfg=json.loads((ROOT/"configs/protocol.json").read_text())
    start=time.perf_counter()
    checks,polezero=analytical_suite(cfg,figures)
    checks+=local_small_signal_check(cfg)
    cases=[]
    primary=cfg["active_resistance_pu_primary"]
    alt=cfg["active_resistance_alternative"]
    for branch,Ra in [("paper_pu",primary),("table_ohm",alt["actual_ohm"]/alt["base_ohm"])]:
        fig,axes=plt.subplots(3,2,figsize=(12,8),sharex=True,layout="constrained")
        for col,L in enumerate(cfg["nonlinear_cases"]["L_pu"]):
            for xi,label,color in [(0,"PSC","#ca6736"),(1,"RFPSC","#245ca6")]:
                p=parameters_from_protocol(cfg,L,Ra)
                data=simulate(p,xi,cfg)
                refined=simulate(p,xi,cfg,max_step=cfg["solver"]["refinement_max_step_s"],rtol=1e-10,atol=1e-12)
                diff=float(np.max(np.abs(data["power_pu"]-refined["power_pu"])))
                tol=cfg["predeclared_checks"]["time_domain_refinement_max_power_difference_pu"]
                checks.append({"check":"time_step_and_tolerance_refinement","branch":branch,"L_pu":L,"controller":label,
                               "max_power_difference_pu":diff,"tolerance":tol,"passed":diff<tol})
                name=f"{branch}_L{L:g}_{label.lower()}"
                np.savez_compressed(outputdir/(name+".npz"),**data)
                header=",".join(data)
                np.savetxt(outputdir/(name+".csv"),np.column_stack(list(data.values())),delimiter=",",header=header,comments="",fmt="%.11g")
                cases.append({"case_id":name,"branch":branch,"L_pu":L,"Ra_pu":Ra,"controller":label,
                              "metrics":metrics(data,cfg),"data_file":name+".npz"})
                for row,key in enumerate(["power_pu","id_pu","voltage_abs_pu"]):
                    axes[row,col].plot(data["time_s"],data[key],color=color,label=label,lw=1.2)
            axes[0,col].step(data["time_s"],data["reference_pu"],where="post",color="black",ls="--",lw=.9,label="reference")
            axes[0,col].set_title(f"{'Weak' if L==1 else 'Strong'} grid: L={L:g} p.u.")
            axes[2,col].set_xlabel("Time (s)")
        for row,label in enumerate(["Active power (p.u.)","d-axis current (p.u.)","Voltage magnitude (p.u.)"]):
            axes[row,0].set_ylabel(label)
        for ax in axes.flat:ax.grid(alpha=.2)
        axes[0,0].legend(ncol=3,fontsize=8)
        fig.suptitle(f"Continuous average-value reconstruction, Ra={Ra:.5g} p.u. ({branch})\nExperimental waveforms are not reproduced quantitatively")
        fig.savefig(figures/f"step_responses_{branch}.png",dpi=170);plt.close(fig)
    source_manifest=json.loads((ROOT/"sources/source_manifest.json").read_text())
    manifest={"paper_doi":cfg["paper_doi"],"evidence_scope":cfg["evidence_scope"],
              "source_sha256":source_manifest["sha256"],
              "protocol_sha256":hashlib.sha256((ROOT/"configs/protocol.json").read_bytes()).hexdigest(),
              "implementation_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "python":sys.version,"platform":platform.platform(),
              "packages":{p:importlib.metadata.version(p) for p in ["numpy","scipy","matplotlib"]},
              "elapsed_seconds":time.perf_counter()-start,
              "all_numerical_checks_passed":all(c["passed"] for c in checks),
              "case_count":len(cases),"checks":checks,"pole_zero_cases":polezero,"cases":cases,
              "parameter_ambiguity":{"Ra_reported_pu":primary,"Ra_actual_over_base":alt["actual_ohm"]/alt["base_ohm"],"resolved":False}}
    (outputdir/"run_manifest.json").write_text(json.dumps(manifest,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"output":str(outputdir),"cases":len(cases),"checks":len(checks),"passed":sum(c["passed"] for c in checks),"elapsed_s":manifest["elapsed_seconds"]}))
    if not manifest["all_numerical_checks_passed"]:raise SystemExit(2)


if __name__=="__main__":main()
