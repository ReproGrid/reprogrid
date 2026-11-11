"""Continuous averaged reproduction of the DO-GFM equations in nominal coordinates."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import platform
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import linear_sum_assignment, root


HERE = Path(__file__).resolve().parents[1]
PROTOCOL = HERE / "configs/protocol.json"
I_MAX = 1.3  # Section VI-C; reused as specified in the frozen protocol.
COMPARE_R = .05  # V-C estimate, used only for the displayed physical command.


def read_json(path):
    return json.loads(path.read_text())


def pack(i, z):
    return np.array([i.real, i.imag, z.real, z.imag])


def unpack(x):
    return complex(x[0], x[1]), complex(x[2], x[3])


def point(L, p, v, ug):
    delta = np.arcsin(p*L/(ug*v))
    v0 = complex(v*np.exp(1j*delta))
    return (v0-ug)/(1j*L), v0


def ko_for(mode, g):
    return g["alpha_o"] if mode == "do12" else complex(g["alpha_o"], -g["omega_hat"])


def unit(z):
    size = abs(z)
    if size < 1e-10:
        raise ValueError("Undefined voltage direction at a zero estimate")
    return z/size


def do_feedback(i, vhat, mode, p, q, v, g, Lhat, limit=I_MAX):
    """Eqs. (10b–d), (11–15), (20). Return applied correction and reference magnitudes."""
    phat = (vhat*i.conjugate()).real
    if mode == "gfl":
        uhat = vhat-1j*g["omega_hat"]*Lhat*i
        qhat = (uhat*i.conjugate()).imag
        e = unit(uhat)*(g["gfl_kp"]*(p-phat)-1j*g["gfl_kq"]*(q-qhat))
    else:
        direction = unit(vhat)
        if mode == "do12":
            uhat = vhat-1j*g["omega_hat"]*Lhat*i
            kv = -1j*g["kv"]*unit(uhat)
        elif mode == "do13":
            kv = (1-1j*g["kv"])*direction
        else:
            raise ValueError(f"Unknown control mode {mode}")
        e = g["kp"]*direction*(p-phat) + kv*(v-abs(vhat))
    kc = g["alpha_c"]*Lhat
    target = i+e/kc
    raw = abs(target)
    if raw > limit:
        e = kc*(target*limit/raw-i)
    return e, raw, min(raw, limit)


def dynamics_do(x, ug, mode, p, q, v, L, Lhat, g, limit=I_MAX):
    i, z = unpack(x)
    ko = ko_for(mode, g)
    vhat = z-ko*Lhat*i
    e, raw, applied = do_feedback(i, vhat, mode, p, q, v, g, Lhat, limit)
    u = vhat+e
    di = (u-ug)/L-1j*g["omega_hat"]*i
    dz = (ko+1j*g["omega_hat"])*e
    return pack(di, dz), i, z, vhat, e, u, raw, applied


def dynamics_rfpsc(x, ug, p, v, L, g):
    """Appendix C, Eq. (30). Store angle relative to the nominal coordinates."""
    i = complex(x[0], x[1])
    phase = np.exp(1j*x[2])
    local_i = i*phase.conjugate()
    requested = p/v+1j*x[3]
    raw = abs(requested)
    iref = requested*I_MAX/raw if raw > I_MAX else requested
    local_u = v+g["rfpsc_ra"]*(iref-local_i)
    u = local_u*phase
    di = (u-ug)/L-1j*g["omega_hat"]*i
    return (np.array([di.real, di.imag, g["rfpsc_kps"]*(p-(local_u*local_i.conjugate()).real),
                      g["rfpsc_filter"]*(local_i.imag-x[3])]), i, u, raw, min(raw, I_MAX))


def grid_at(t, scenario, wbase, ug):
    ramp = scenario.get("ramp")
    if ramp is None:
        return complex(ug), 1., 1.+0j
    start, stop, end = ramp
    elapsed = max(0., t-start)
    span = stop-start
    during = min(elapsed, span)
    omega = 1+(end-1)*during/span
    phase = np.exp(1j*wbase*(end-1)*(during**2/(2*span)+max(0., elapsed-span)))
    return ug*phase, omega, phase


def make_scenarios(cfg):
    seq = cfg["power_sequence"]
    for item in seq["cases"]:
        bounds = [0., *seq["times_s"], seq["stop_s"]]
        yield {**item, "segments": [(start, stop, item["control"], p, 0., seq["v_ref_pu"])
                                     for start, stop, p in zip(bounds[:-1], bounds[1:], seq["p_pu"], strict=True)]}
    seq = cfg["frequency_sequence"]
    bounds = [0., *seq["ramp_s"], seq["stop_s"]]
    yield {**seq, "ramp": [*seq["ramp_s"], seq["omega_pu"][-1]],
           "segments": [(start, stop, seq["control"], seq["p_pu"], 0., seq["v_ref_pu"])
                        for start, stop in zip(bounds[:-1], bounds[1:], strict=True)]}
    seq = cfg["mode_sequence"]
    bounds = [0., *seq["times_s"], seq["stop_s"]]
    yield {**seq, "segments": [(start, stop, mode, seq["p_pu"], q, seq["v_ref_pu"])
                                for start, stop, mode, q in zip(bounds[:-1], bounds[1:], seq["control"],
                                                               seq["q_pu"], strict=True)]}


def integrate_scenario(scenario, cfg, solver):
    g, base = cfg["table_ii"], cfg["base"]
    wbase = 2*np.pi*base["frequency_hz"]
    L, Lhat = scenario["L"], g["Lhat"]
    ugmag = base["grid_voltage_pu"]
    segments = scenario["segments"]
    _, _, initial_mode, p0, _, v0mag = segments[0]
    i0, v0 = point(L, p0, v0mag, ugmag)
    rfpsc = initial_mode == "rfpsc"
    if rfpsc:
        state = np.array([i0.real, i0.imag, np.angle(v0), (i0*unit(v0).conjugate()).imag])
    else:
        state = pack(i0, v0+ko_for(initial_mode, g)*Lhat*i0)
    h = cfg["numerics"]["sample_s"]
    t_all = np.round(np.arange(round(segments[-1][1]/h)+1)*h, 12)
    trace = {}
    switches = []

    def instant(t, x, mode, p, q, v):
        ug, omega, phase = grid_at(t, scenario, wbase, ugmag)
        if rfpsc:
            dx, i, u, raw, applied = dynamics_rfpsc(x, ug, p, v, L, g)
            z = vhat = e = None
        else:
            dx, i, z, vhat, e, u, raw, applied = dynamics_do(x, ug, mode, p, q, v, L, Lhat, g)
        grid_power = ug*i.conjugate()
        row = {"p": grid_power.real, "q": grid_power.imag, "p_ref": p, "q_ref": q,
               "v_true": abs(ug+1j*omega*L*i), "u_net": abs(u), "u_command_comparison": abs(u+COMPARE_R*i),
               "current": abs(i), "current_ref_raw": raw, "current_ref_applied": applied, "omega": omega,
               "i_grid": i*phase.conjugate(), "u_grid": u*phase.conjugate()}
        if z is not None:
            row.update(vhat=abs(vhat), vhat_grid=vhat*phase.conjugate(),
                       z_grid=z*phase.conjugate(), e_grid=e*phase.conjugate())
        return dx, row

    for index, (a, b, mode, p, q, v) in enumerate(segments):
        def rhs(t, x):
            ug = grid_at(t, scenario, wbase, ugmag)[0]
            dx = (dynamics_rfpsc(x, ug, p, v, L, g)[0] if rfpsc else
                  dynamics_do(x, ug, mode, p, q, v, L, Lhat, g)[0])
            return wbase*dx
        sol = solve_ivp(rhs, (a,b), state, dense_output=True, **solver)
        if not sol.success:
            raise RuntimeError(f"{scenario['name']}: {sol.message}")
        mask = (t_all >= a) & ((t_all <= b) if index == len(segments)-1 else (t_all < b))
        for t, x in zip(t_all[mask], sol.sol(t_all[mask]).T, strict=True):
            _, row = instant(t, x, mode, p, q, v)
            for key, value in row.items():
                trace.setdefault(key, []).append(value)
        state = sol.y[:,-1]
        if index < len(segments)-1 and mode != segments[index+1][2]:
            _, left = instant(b, state, mode, p, q, v)
            _, right = instant(b, state, *segments[index+1][2:])
            switches.append({"time_s": b, "from": mode, "to": segments[index+1][2],
                             "grid_current_jump": abs(right["i_grid"]-left["i_grid"]),
                             "observer_state_jump": abs(right["z_grid"]-left["z_grid"]),
                             "voltage_command_vector_jump": abs(right["u_grid"]-left["u_grid"])})
    return {"time_s": t_all, **{k: np.array(v) for k,v in trace.items()}}, switches


def analytical_model(mode, L, Lhat, p, v, ug, g):
    """Appendices A–B using [delta p, delta |v|, delta error real, delta error imag]."""
    i0, v0 = point(L, p, v, ug)
    ko = ko_for(mode,g)
    kp = g["kp"]*unit(v0)
    # Eq. (12) uses the estimated grid voltage, which equals Ug only at matched L.
    uhat0 = v0-1j*g["omega_hat"]*Lhat*i0
    kv = -1j*g["kv"]*unit(uhat0) if mode == "do12" else (1-1j*g["kv"])*unit(v0)
    epsilon = (L-Lhat)/L
    wo = g["omega_hat"]

    def equation(x, ref):
        dp, dv = x[:2]
        error = complex(x[2], x[3])
        dp_err = (i0.conjugate()*error).real
        dv_err = (v0.conjugate()*error).real/v
        de = kp*(ref[0]-dp+dp_err)+kv*(ref[1]-dv+dv_err)
        net = de-error
        der = -(ko*(1-epsilon)+1j*wo)*error-epsilon*ko*de
        return np.array([(ug*net.conjugate()).real/L, wo*(v0*net.conjugate()).imag/v,
                         der.real, der.imag])
    A = np.column_stack([equation(col, np.zeros(2)) for col in np.eye(4)])
    B = np.column_stack([equation(np.zeros(4), col) for col in np.eye(2)])
    params = {"alpha_p": float((ug*kp.conjugate()).real/L),
              "alpha_v": float(wo*(v0*kv.conjugate()).imag/v),
              "beta_pv": float((ug*kv.conjugate()).real/L),
              "beta_vp": float(wo*(v0*kp.conjugate()).imag/v),
              "observer_decay_real": float((-(ko*(1-epsilon)+1j*wo)).real),
              "kv_error_coupling_real": float((epsilon*kv*ko*unit(v0).conjugate()).real)}
    return A, B, params


def finite_jacobian(mode, L, Lhat, p, v, ug, g, h):
    i0, v0 = point(L,p,v,ug)
    ko = ko_for(mode,g)
    equilibrium = pack(i0,v0+ko*Lhat*i0)

    def fun(x, r):
        return dynamics_do(x, complex(ug), mode, p+r[0], 0., v+r[1], L,Lhat,g)[0]
    A = np.column_stack([(fun(equilibrium+col,np.zeros(2))-fun(equilibrium-col,np.zeros(2)))/(2*h)
                         for col in h*np.eye(4)])
    B = np.column_stack([(fun(equilibrium,col)-fun(equilibrium,-col))/(2*h) for col in h*np.eye(2)])
    def transform(col):
        i,z = unpack(col)
        dv = 1j*g["omega_hat"]*L*i
        error = dv-z+ko*Lhat*i
        return np.array([(ug*i.conjugate()).real, (v0.conjugate()*dv).real/v,error.real,error.imag])
    T = np.column_stack([transform(col) for col in np.eye(4)])
    return T@A@np.linalg.inv(T), T@B, float(np.max(abs(fun(equilibrium,np.zeros(2)))))


def pole_difference(a,b):
    distance = abs(np.asarray(a)[:,None]-np.asarray(b)[None,:])
    rows, cols = linear_sum_assignment(distance)
    return float(np.max(distance[rows,cols]))


def linear_studies(cfg):
    small, num, g = cfg["small_signal"], cfg["numerics"], cfg["table_ii"]
    ug = cfg["base"]["grid_voltage_pu"]
    rows, checks, sweep = [], {}, []
    for mode in small["gain_options"]:
        for L in small["grid_L"]:
            for p in small["power_operating_points"]:
                for label in small["Lhat_modes"]:
                    Lhat = L if label == "matched" else g["Lhat"]
                    A,B,coef = analytical_model(mode,L,Lhat,p,1.,ug,g)
                    numeric_A,numeric_B,residual = finite_jacobian(mode,L,Lhat,p,1.,ug,g,num["jacobian_step_pu"])
                    key = f"{mode}_{L:.2f}_{p:.1f}_{label}"
                    err = max(float(np.max(abs(A-numeric_A))),float(np.max(abs(B-numeric_B))))
                    checks[key+"_jacobian"] = err < num["jacobian_abs_tolerance"]
                    checks[key+"_equilibrium"] = residual < num["equilibrium_residual_tolerance"]
                    poles = np.linalg.eigvals(A)
                    row = {"mode": mode,"L": L,"Lhat": Lhat,"p": p,**coef,"jacobian_max_abs_error":err,
                           "equilibrium_max_abs_residual":residual,
                           "poles":[[float(x.real),float(x.imag)] for x in poles]}
                    if label == "matched":
                        obs = -ko_for(mode,g)-1j*g["omega_hat"]
                        target = [-coef["alpha_p"],-coef["alpha_v"],obs,obs.conjugate()]
                        perr = pole_difference(poles,target)
                        ap,av,beta = (coef[k] for k in ("alpha_p","alpha_v","beta_pv"))
                        terr = 0.
                        for s in 1j*np.geomspace(.001,100.,51):
                            exact = np.linalg.solve(s*np.eye(4)-A,B)[:2,:]
                            expected = np.array([[ap/(s+ap),beta*s/((s+ap)*(s+av))],
                                                 [0.,av/(s+av)]])
                            terr = max(terr,float(np.max(abs(exact-expected))))
                        row.update(matched_pole_max_abs_error=perr, reference_transfer_max_abs_error=terr)
                        checks[key+"_matched_poles"] = perr < num["matched_pole_tolerance"]
                        checks[key+"_matched_reference_transfer"] = terr < num["jacobian_abs_tolerance"]
                    rows.append(row)
        for p in small["sweep_p"]:
            for L in np.linspace(*small["sweep_L"]):
                A,_,coef = analytical_model(mode,L,g["Lhat"],p,1.,ug,g)
                poles = np.linalg.eigvals(A)
                sweep.append([mode,float(L),p,float(max(poles.real)),coef["observer_decay_real"],coef["kv_error_coupling_real"]])
    checks["eq29_voltage_error_decoupling"] = max(abs(r[5]) for r in sweep if r[0]=="do13") < 1e-12
    checks["appendix_B_observer_coefficient_negative"] = max(r[4] for r in sweep) < 0.
    return rows, sweep, checks


def local_studies(cfg):
    g, small, num = cfg["table_ii"], cfg["small_signal"], cfg["numerics"]
    ug = cfg["base"]["grid_voltage_pu"]
    wbase = 2*np.pi*cfg["base"]["frequency_hz"]
    t = np.linspace(0., small["local_stop_s"], 401)
    amp = small["local_step_pu"]
    arrays, rows, checks = {"time_s":t}, {}, {}
    for case in small["local_cases"]:
        mode,L,Lhat,p = (case[k] for k in ("control","L","Lhat","p"))
        i0,v0 = point(L,p,1.,ug)
        y0 = pack(i0,v0+ko_for(mode,g)*Lhat*i0)
        A,B,_ = analytical_model(mode,L,Lhat,p,1.,ug,g)
        for k, channel in enumerate(("p","v")):
            key = f"{mode}_L{L}_{channel}"
            def physical(_,x):
                return wbase*dynamics_do(x,complex(ug),mode,p+(amp if k==0 else 0.),0.,
                                         1.+(amp if k==1 else 0.),L,Lhat,g)[0]
            def linear(_,x):
                return wbase*(A@x+B[:,k])
            nonlin = solve_ivp(physical,(t[0],t[-1]),y0,t_eval=t,**num["main_solver"])
            lin = solve_ivp(linear,(t[0],t[-1]),np.zeros(4),t_eval=t,**num["main_solver"])
            if not nonlin.success or not lin.success:
                raise RuntimeError(f"Local step failed: {key}")
            i = nonlin.y[0]+1j*nonlin.y[1]
            response = np.vstack([(ug*i.real-p)/amp,(abs(ug+1j*L*i)-1.)/amp])
            error = float(np.max(abs(response-lin.y[:2])))
            arrays[key+"_linear"] = lin.y[:2]
            arrays[key+"_nonlinear"] = response
            rows[key] = {"max_abs_error_relative_to_step":error}
            checks[key+"_small_step"] = error < num["local_relative_to_step_tolerance"]
    return arrays, rows, checks


def static_frequency_studies(cfg):
    """Compare Eq. (16)/(26) at s=0 to Eqs. (21)–(23) and nonlinear steady states."""
    g, small, num = cfg["table_ii"], cfg["small_signal"], cfg["numerics"]
    spec = small["static_frequency_derivative"]
    L,h = spec["L"],spec["omega_difference_pu"]
    ug = cfg["base"]["grid_voltage_pu"]
    rows, checks = [], {}
    for mode in small["gain_options"]:
        for p in spec["power_operating_points"]:
            i0,v0 = point(L,p,1.,ug)
            ko = ko_for(mode,g)
            z0 = v0+ko*L*i0
            _,_,coef = analytical_model(mode,L,L,p,1.,ug,g)
            ap,av,beta = (coef[k] for k in ("alpha_p","alpha_v","beta_pv"))
            ratio = z0/(ko+1j*g["omega_hat"])
            e = 1j*ratio  # Unit frequency deviation and Eq. (21a) at s=0.
            e_v = (v0.conjugate()*e).real
            d = ((ug-ap*L*i0.conjugate())*ratio).imag/(ap*L)
            q = ((-av+1j*g["omega_hat"])*ratio*v0.conjugate()).imag/av
            printed = -d-beta*q/ap
            derived = d+beta*(e_v-q)/ap

            def steady(omega):
                def residual(x):
                    i,z = unpack(x)
                    vhat = z-ko*L*i
                    correction = do_feedback(i,vhat,mode,p,0.,1.,g,L)[0]
                    di = (vhat+correction-ug)/L-1j*omega*i
                    dz = (ko+1j*g["omega_hat"])*correction+1j*(g["omega_hat"]-omega)*z
                    return pack(di,dz)
                result = root(residual,pack(i0,z0),tol=1e-11)
                if np.max(abs(residual(result.x))) >= num["equilibrium_residual_tolerance"]:
                    raise RuntimeError(f"Steady frequency solution failed: {mode}, {omega}")
                i,_ = unpack(result.x)
                return np.array([ug*i.real,abs(ug+1j*omega*L*i)])

            delta = (steady(1.+h)-steady(1.-h))/(2*h)
            exact_49 = steady(.98)
            checks[f"static_frequency_{mode}_{p}"] = max(abs(delta[0]-derived),abs(delta[1]-q)) < num["static_droop_derivative_tolerance"]
            rows.append({"mode":mode,"p0":p,"p_derivative_numeric":float(delta[0]),
                         "p_derivative_eq16_26_printed":float(printed),"p_derivative_from_eq21_23":float(derived),
                         "v_derivative_numeric":float(delta[1]),"v_derivative_eq26b":float(q),
                         "p_at_49_hz":float(exact_49[0]),"v_at_49_hz":float(exact_49[1])})
    return rows, checks


def step_metrics(trace, cfg):
    t = trace["time_s"]
    seq = cfg["power_sequence"]
    window = cfg["numerics"]["plateau_window_s"]
    rows = []
    for start,stop,p0,p1 in zip(seq["times_s"],[*seq["times_s"][1:],seq["stop_s"]],
                                seq["p_pu"][:-1],seq["p_pu"][1:],strict=True):
        selection = (t>=start)&(t<stop)
        x,y = t[selection],(trace["p"][selection]-p0)/(p1-p0)
        def cross(level):
            idx = np.flatnonzero(y>=level)
            if idx.size==0:
                return None
            n = idx[0]
            return x[n] if n==0 else float(np.interp(level,y[n-1:n+1],x[n-1:n+1]))
        a,b = cross(.1),cross(.9)
        tail = (t>=stop-window)&(t<stop)
        rows.append({"at_s":start,"p_ref":p1,"transition_10_90_ms":1000*(b-a) if a is not None and b is not None else None,
                     "plateau_p":float(np.mean(trace["p"][tail])),"plateau_q":float(np.mean(trace["q"][tail]))})
    return rows


def make_figures(traces, sweep, cfg, target):
    fig, ax = plt.subplots(3,2,sharex=True,figsize=(10,7),layout="constrained")
    for col,strength in enumerate(("strong","weak")):
        for mode,color,label in (("do13","#286b9d","DO-GFM Eq. (13)"),("rfpsc","#bf6931","RFPSC Eq. (30)")):
            values = traces[f"{mode}_{strength}"]
            for row,(key,unit_label) in enumerate((("p","Grid P (pu)"),("q","Grid Q (pu)"),
                                                 ("u_command_comparison","Voltage command (pu)"))):
                ax[row,col].plot(values["time_s"],values[key],color=color,label=label)
                ax[row,col].set_ylabel(unit_label)
                ax[row,col].grid(alpha=.25)
        values = traces[f"do13_{strength}"]
        ax[0,col].step(values["time_s"],values["p_ref"],where="post",color=".5",ls=":",label="P reference")
        ax[0,col].set_title(f"{strength.capitalize()} grid")
        ax[0,col].legend(fontsize=8)
        ax[2,col].set_xlabel("Time (s)")
    fig.suptitle("Independent continuous averaged comparison")
    fig.savefig(target/"independent_tracking.png",dpi=160)
    plt.close(fig)

    fig,ax = plt.subplots(2,2,sharex="col",figsize=(10,6),layout="constrained")
    f = traces["frequency_fig9"]
    ax[0,0].plot(f["time_s"],50*f["omega"])
    ax[0,0].set_ylabel("Grid frequency (Hz)")
    ax[1,0].plot(f["time_s"],f["p"],label="P")
    ax[1,0].plot(f["time_s"],f["vhat"],label="Voltage estimate")
    ax[1,0].legend()
    ax[0,0].set_title("Fig. 9 frequency ramp, ideal plant")
    m = traces["mode_fig10"]
    ax[0,1].plot(m["time_s"],m["p"],label="P")
    ax[0,1].plot(m["time_s"],m["q"],label="Q")
    ax[0,1].legend()
    ax[1,1].plot(m["time_s"],m["vhat"],label="Voltage estimate")
    ax[1,1].plot(m["time_s"],m["u_command_comparison"],label="Voltage command")
    ax[1,1].legend()
    ax[0,1].set_title("Fig. 10 mode switch, ideal plant")
    for col in range(2):
        ax[1,col].set_xlabel("Time (s)")
        for row in range(2):
            ax[row,col].grid(alpha=.25)
            if not (row==0 and col==0):
                ax[row,col].set_ylabel("pu")
    for t in cfg["mode_sequence"]["times_s"]:
        for axis in ax[:,1]:
            axis.axvline(t,color=".6",ls=":",lw=1)
    fig.savefig(target/"frequency_and_mode.png",dpi=160)
    plt.close(fig)

    fig,ax = plt.subplots(1,2,sharey=True,figsize=(9,3.8),layout="constrained")
    for axis,mode,label in zip(ax,("do12","do13"),("Decoupling gains (12)","RFPSC-type gains (13)"),strict=True):
        for p in cfg["small_signal"]["sweep_p"]:
            rows = [r for r in sweep if r[0]==mode and r[2]==p]
            axis.plot([r[1] for r in rows],[r[3] for r in rows],label=f"P₀={p:g}")
        axis.set(title=label,xlabel="Actual inductance L (pu)")
        axis.axhline(0,color=".4",ls="--",lw=.8)
        axis.grid(alpha=.25)
        axis.legend(fontsize=8)
    ax[0].set_ylabel("Maximum real closed-loop pole (pu)")
    fig.suptitle("Inductance mismatch with L̂ = 0.15 pu")
    fig.savefig(target/"inductance_sweep.png",dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=HERE/"results")
    parser.add_argument("--figures",type=Path,default=HERE/"figures")
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    args.figures.mkdir(parents=True,exist_ok=True)
    start = datetime.now(timezone.utc)
    timer = perf_counter()
    cfg = read_json(PROTOCOL)
    num = cfg["numerics"]
    traces, results, checks = {}, {}, {}
    for scenario in make_scenarios(cfg):
        name = scenario["name"]
        data, switches = integrate_scenario(scenario,cfg,num["main_solver"])
        refined,_ = integrate_scenario(scenario,cfg,num["refined_solver"])
        errors = {k:float(np.max(abs(data[k]-refined[k]))) for k in data}
        checks[name+"_solver_refinement"] = max(errors.values()) < num["trajectory_tolerance_pu"]
        checks[name+"_finite_trace"] = all(np.isfinite(x).all() for x in data.values())
        checks[name+"_limited_reference"] = float(np.max(data["current_ref_applied"])) <= I_MAX+1e-12
        results[name] = {"L_pu":scenario["L"],"refinement_max_abs_by_field":errors,"switches":switches,
                         "limiter_active_sample_count":int(np.count_nonzero(data["current_ref_raw"]>I_MAX)),
                         "peak_current_pu":float(max(data["current"])),
                         "max_voltage_command_pu":float(max(data["u_command_comparison"]))}
        if name in {r["name"] for r in cfg["power_sequence"]["cases"]}:
            results[name]["steps"] = step_metrics(data,cfg)
        traces[name] = data
        np.savez_compressed(args.output/f"ideal_{name}.npz",**data)
    lin,sweep,lin_checks = linear_studies(cfg)
    checks.update(lin_checks)
    local,local_results,local_checks = local_studies(cfg)
    checks.update(local_checks)
    static,static_checks = static_frequency_studies(cfg)
    checks.update(static_checks)
    np.savez_compressed(args.output/"ideal_local_steps.npz",**local)
    sweep_path = args.output/"ideal_inductance_sweep.csv"
    with sweep_path.open("w") as handle:
        handle.write("mode,L_pu,p_pu,max_closed_loop_real_pu,observer_coefficient_real_pu,voltage_error_coupling_real_pu\n")
        for row in sweep:
            handle.write(",".join(map(str,row))+"\n")
    f = traces["frequency_fig9"]
    p_49 = float(np.mean(f["p"][f["time_s"]>=1.2]))
    checks["fig9_approximate_steady_state_power"] = abs(p_49-cfg["frequency_sequence"]["steady_state_power_target_approx_pu"]) <= cfg["frequency_sequence"]["paper_approximate_tolerance_pu"]
    m = traces["mode_fig10"]
    gfl = (m["time_s"]>=.2)&(m["time_s"]<.8)
    gfl_p_dev = float(np.max(abs(m["p"][gfl]-cfg["mode_sequence"]["p_pu"])))
    checks["gfl_matched_P_independent_of_Q_reference"] = gfl_p_dev < num["gfl_decoupling_tolerance_pu"]
    def difference(a,b):
        return float(np.sqrt(np.mean((traces[a]["p"]-traces[b]["p"])**2)))
    comparison = {strength:difference(f"do13_{strength}",f"rfpsc_{strength}") for strength in ("strong","weak")}
    make_figures(traces,sweep,cfg,args.figures)
    sweep_summary = {}
    for mode in cfg["small_signal"]["gain_options"]:
        subset = [row for row in sweep if row[0]==mode]
        worst = max(subset,key=lambda r:r[3])
        sweep_summary[mode] = {"count":len(subset),"unstable_count":sum(r[3]>0. for r in subset),
                               "least_damped_L":worst[1],"least_damped_p":worst[2],"max_pole_real":worst[3],
                               "max_abs_voltage_error_coupling":max(abs(r[5]) for r in subset)}
    manifest = {
        "doi":cfg["doi"],"scope":cfg["scope"],"source_sha256":read_json(HERE/"sources/source_manifest.json")["sha256"],
        "protocol_sha256":sha256(PROTOCOL.read_bytes()).hexdigest(),
        "implementation_sha256":sha256(Path(__file__).read_bytes()).hexdigest(),
        "python":platform.python_version(),"platform":platform.platform(),
        "packages":{pkg:version(pkg) for pkg in ("numpy","scipy","matplotlib")},
        "started_utc":start.isoformat(),"finished_utc":datetime.now(timezone.utc).isoformat(),
        "wall_seconds":perf_counter()-timer,"scenarios":results,"linearization_cases":lin,
        "local_steps":local_results,"frequency_static":static,"inductance_sweep":sweep_summary,
        "fig9_late_p_pu":p_49,"fig9_late_vhat_pu":float(np.mean(f["vhat"][f["time_s"]>=1.2])),
        "fig10_gfl_max_active_deviation_pu":gfl_p_dev,"do13_vs_rfpsc_active_rmse_pu":comparison,
        "checks":{k:bool(v) for k,v in checks.items()},"numerical_checks_passed":all(checks.values())}
    (args.output/"run_manifest.json").write_text(json.dumps(manifest,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"passed":manifest["numerical_checks_passed"],"checks":len(checks),
                      "failed":[k for k,v in checks.items() if not v],"wall_seconds":manifest["wall_seconds"]}))
    return 0 if manifest["numerical_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
