"""Separate implementation using grid-aligned real coordinates and Eq. (10)."""
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root


HERE = Path(__file__).resolve().parent
CASE = HERE.parent
PROTOCOL = CASE / "configs/protocol.json"
MAXIMUM = 1.3


def j(x):
    return np.array([-x[1], x[0]])


def scalar(x):
    return complex(*x)


def rotate(x, angle):
    return np.cos(angle)*x+np.sin(angle)*j(x)


def stationary_initial(p, voltage, grid, inductance):
    direct = p/grid
    terminal_direct = np.sqrt(voltage**2-(inductance*direct)**2)
    return np.array([direct,(grid-terminal_direct)/inductance]), np.array([terminal_direct,inductance*direct])


def equations(state, kind, p, q, voltage, frequency, grid, inductance, estimate, g):
    """Eq. (1a) plus explicit current derivative in Eq. (10a), in the actual grid frame."""
    nominal = g["omega_hat"]
    i, vh = state[:2],state[2:]
    kp_error = p-np.dot(vh,i)
    gh = vh-nominal*estimate*j(i)
    if kind == "gfl":
        q_error = q-(gh[1]*i[0]-gh[0]*i[1])
        e = (g["gfl_kp"]*kp_error*gh-g["gfl_kq"]*q_error*j(gh))/np.linalg.norm(gh)
    else:
        direction = vh/np.linalg.norm(vh)
        diff = voltage-np.linalg.norm(vh)
        if kind == "do12":
            voltage_feedback = -g["kv"]*diff*j(gh)/np.linalg.norm(gh)
        else:
            voltage_feedback = diff*(direction-g["kv"]*j(direction))
        e = g["kp"]*kp_error*direction+voltage_feedback
    kc = g["alpha_c"]*estimate
    iref = i+e/kc
    size = np.linalg.norm(iref)
    applied = min(size,MAXIMUM)
    if applied < size:
        e = kc*(iref*applied/size-i)
    u = vh+e
    di = (u-np.array([grid,0.]))/inductance-frequency*j(i)
    raw = e+(nominal-frequency)*estimate*j(i)-estimate*di
    bo = 0. if kind == "do12" else -nominal
    dvh = nominal*j(e)+(nominal-frequency)*j(vh)+g["alpha_o"]*raw+bo*j(raw)
    return np.r_[di,dvh], e,u,size,applied


def rfpsc(state,p,v,frequency,grid,L,g):
    i,angle,ifq = state[:2],state[2],state[3]
    local = rotate(i,-angle)
    raw = np.array([p/v,ifq])
    size = np.linalg.norm(raw)
    applied = min(size,MAXIMUM)
    if applied<size:
        raw *= applied/size
    ulocal = np.array([v,0.])+g["rfpsc_ra"]*(raw-local)
    u = rotate(ulocal,angle)
    dcurrent = (u-np.array([grid,0.]))/L-frequency*j(i)
    return np.r_[dcurrent,g["omega_hat"]-frequency+g["rfpsc_kps"]*(p-np.dot(ulocal,local)),
                 g["rfpsc_filter"]*(local[1]-ifq)],u,size,applied


def eq19_audit(state,derivative,e,mode,frequency,Lhat,g):
    """Check rotation to the voltage-estimate frame using the solved plant current derivative."""
    i,h = scalar(state[:2]),scalar(state[2:])
    di,dh = scalar(derivative[:2]),scalar(derivative[2:])
    rotation = h.conjugate()/abs(h)
    delta_rate = (dh*rotation).imag/abs(h)
    wc = frequency+delta_rate
    ia,ea = i*rotation,scalar(e)*rotation
    dia = (di-1j*delta_rate*i)*rotation
    nominal = g["omega_hat"]
    ko = complex(g["alpha_o"],0. if mode=="do12" else -nominal)
    ua = ea+1j*nominal*Lhat*ia-Lhat*dia
    denominator = abs(h)+Lhat*(ko*ia).real
    from_eq19 = (ko*ua+1j*nominal*(abs(h)+ea)).imag/denominator
    mag_rate = (ko*(ua-1j*wc*Lhat*ia)+1j*nominal*ea).real
    return abs(wc-from_eq19),abs(mag_rate-(dh*rotation).real),abs(denominator)


def scenarios(config):
    seq = config["power_sequence"]
    for item in seq["cases"]:
        yield (item["name"],item["L"],[0.,*seq["times_s"],seq["stop_s"]],
               [(item["control"],p,0.,seq["v_ref_pu"]) for p in seq["p_pu"]],None)
    seq = config["frequency_sequence"]
    yield (seq["name"],seq["L"],[0.,*seq["ramp_s"],seq["stop_s"]],
           [(seq["control"],seq["p_pu"],0.,seq["v_ref_pu"])]*3,(seq["ramp_s"],seq["omega_pu"]))
    seq = config["mode_sequence"]
    yield (seq["name"],seq["L"],[0.,*seq["times_s"],seq["stop_s"]],
           [(mode,seq["p_pu"],q,seq["v_ref_pu"]) for mode,q in zip(seq["control"],seq["q_pu"],strict=True)],None)


def integrate(test,config):
    name,L,times,commands,ramp = test
    g,num = config["table_ii"],config["numerics"]
    grid = config["base"]["grid_voltage_pu"]
    angular_base = 2*np.pi*config["base"]["frequency_hz"]
    estimate = g["Lhat"]
    first,p0,_,v0 = commands[0]
    i,v = stationary_initial(p0,v0,grid,L)
    if first=="rfpsc":
        angle = np.arctan2(v[1],v[0])
        x = np.r_[i,angle,rotate(i,-angle)[1]]
    else:
        x = np.r_[i,v]
    sample = num["sample_s"]
    time = np.round(np.arange(round(times[-1]/sample)+1)*sample,12)
    data = {}
    eq19 = []
    def freq(t):
        return 1. if ramp is None else float(np.interp(t,*ramp))
    for k,((start,end),(mode,p,q,vref)) in enumerate(zip(zip(times[:-1],times[1:],strict=True),commands,strict=True)):
        def derivative(t,y):
            if mode=="rfpsc":
                return angular_base*rfpsc(y,p,vref,freq(t),grid,L,g)[0]
            return angular_base*equations(y,mode,p,q,vref,freq(t),grid,L,estimate,g)[0]
        solution = solve_ivp(derivative,(start,end),x,dense_output=True,**num["refined_solver"])
        if not solution.success:
            raise RuntimeError(f"{name}: {solution.message}")
        x = solution.y[:,-1]
        selection = (time>=start)&((time<=end) if k==len(commands)-1 else (time<end))
        for n,(t,y) in enumerate(zip(time[selection],solution.sol(time[selection]).T,strict=True)):
            w = freq(t)
            i = scalar(y[:2])
            if mode=="rfpsc":
                _,u,raw,applied = rfpsc(y,p,vref,w,grid,L,g)
                vh = z = e = None
            else:
                dy,e,u,raw,applied = equations(y,mode,p,q,vref,w,grid,L,estimate,g)
                vh = scalar(y[2:])
                ko = complex(g["alpha_o"],0. if mode=="do12" else -g["omega_hat"])
                z = vh+ko*estimate*i
                if n%97==0:
                    eq19.append(eq19_audit(y,dy,e,mode,w,estimate,g))
            u = scalar(u)
            row = {"p":grid*i.real,"q":-grid*i.imag,"p_ref":p,"q_ref":q,
                   "v_true":abs(grid+1j*w*L*i),"u_net":abs(u),"u_command_comparison":abs(u+.05*i),
                   "current":abs(i),"current_ref_raw":raw,"current_ref_applied":applied,"omega":w,
                   "i_grid":i,"u_grid":u}
            if vh is not None:
                row.update(vhat=abs(vh),vhat_grid=vh,z_grid=z,e_grid=scalar(e))
            for key,value in row.items():
                data.setdefault(key,[]).append(value)
    return {"time_s":time,**{k:np.array(v) for k,v in data.items()}},eq19


def equilibrium_derivatives(cfg):
    g = cfg["table_ii"]
    grid = cfg["base"]["grid_voltage_pu"]
    spec = cfg["small_signal"]["static_frequency_derivative"]
    L,h = spec["L"],spec["omega_difference_pu"]
    rows = []
    for mode in cfg["small_signal"]["gain_options"]:
        for p in spec["power_operating_points"]:
            i,v = stationary_initial(p,1.,grid,L)
            def steady(omega):
                def residual(x):
                    return equations(x,mode,p,0.,1.,omega,grid,L,L,g)[0]
                result = root(residual,np.r_[i,v],tol=1e-11)
                error = float(np.max(abs(residual(result.x))))
                if error > cfg["numerics"]["equilibrium_residual_tolerance"]:
                    raise RuntimeError("Grid-frame equilibrium failed")
                iresult = scalar(result.x[:2])
                return np.array([grid*iresult.real,abs(grid+1j*omega*L*iresult)]),error
            plus,res_plus = steady(1.+h)
            minus,res_minus = steady(1.-h)
            at49,res49 = steady(.98)
            numerical = (plus-minus)/(2*h)
            rows.append({"mode":mode,"p0":p,"p_derivative":float(numerical[0]),"v_derivative":float(numerical[1]),
                         "p_at_49_hz":float(at49[0]),"v_at_49_hz":float(at49[1]),
                         "max_equilibrium_residual":max(res_plus,res_minus,res49)})
    return rows


def main():
    started = datetime.now(timezone.utc)
    timer = perf_counter()
    cfg = json.loads(PROTOCOL.read_text())
    raw,rows = {},{}
    for case in scenarios(cfg):
        name = case[0]
        trace,projection = integrate(case,cfg)
        for key,value in trace.items():
            raw[f"{name}__{key}"] = value
        rows[name] = {"sample_count":len(trace["time_s"]),
                      "finite":all(np.isfinite(x).all() for x in trace.values())}
        if projection:
            rows[name].update(eq19_sample_count=len(projection),eq19_speed_error=max(r[0] for r in projection),
                              eq19_magnitude_derivative_error=max(r[1] for r in projection),
                              eq19_min_abs_denominator=min(r[2] for r in projection))
    frequencies = equilibrium_derivatives(cfg)
    np.savez_compressed(HERE/"ideal_grid_frame.npz",**raw)
    errors = [max(r["eq19_speed_error"],r["eq19_magnitude_derivative_error"]) for r in rows.values() if "eq19_speed_error" in r]
    passes = all(x["finite"] for x in rows.values()) and max(errors) < cfg["numerics"]["trajectory_tolerance_pu"]
    result = {"doi":cfg["doi"],"method":"Grid-aligned real-valued Eq. (1), direct Eq. (10) with solved current derivative, and separately implemented Eq. (30). Shared protocol only.",
              "protocol_sha256":sha256(PROTOCOL.read_bytes()).hexdigest(),
              "implementation_sha256":sha256(Path(__file__).read_bytes()).hexdigest(),
              "started_utc":started.isoformat(),"finished_utc":datetime.now(timezone.utc).isoformat(),
              "wall_seconds":perf_counter()-timer,"cases":rows,"frequency_equilibria":frequencies,
              "max_eq19_projection_error":max(errors),"passed":passes}
    (HERE/"results.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"passed":passes,"cases":len(rows),"max_eq19_projection_error":max(errors),"wall_seconds":result["wall_seconds"]}))
    return 0 if passes else 1


if __name__ == "__main__":
    raise SystemExit(main())
