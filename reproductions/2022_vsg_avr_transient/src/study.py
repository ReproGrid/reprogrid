"""Run the paper's sag, power-angle and gain-selection tests with fixed inputs."""

from dataclasses import replace
from math import pi

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

from model import Vsg


def settings(config):
    p = config["table_i"]
    return Vsg(p["omega_n_rad_s"], p["x_grid_pu"], p["active_reference_pu"],
               p["reactive_reference_pu"], p["voltage_setpoint_pu"], p["droop_p_pu"],
               p["droop_q_pu"], p["inertia_h_s"], p["voltage_integrator_per_s"])


def initial_at(vsg, grid):
    lower, _ = vsg.equilibria(grid)
    return (lower, 0., vsg.algebraic_voltage(lower, 0, grid))


def first_time(sol, event):
    times = sol.t_events[event]
    return float(times[0]) if times.size else None


def summarize(vsg, sol, grid, initial, k, mode, stable_eq, saddle, step):
    t = np.linspace(0, sol.t[-1], int(np.ceil(sol.t[-1] / step))+1)
    angle, speed = sol.sol(t)[:2]
    voltage = np.array([vsg.voltage_at(sol,t0,grid,initial,k,mode) for t0 in t])
    peak_voltage, voltage_peak_time = vsg.voltage_peak(sol,grid,initial,k,mode)
    current = np.abs(voltage*np.exp(1j*angle) - grid)/vsg.x
    turn, crossing = first_time(sol,1), first_time(sol,0)
    max_angle = float(sol.sol(turn)[0]) if turn is not None and (crossing is None or turn < crossing) else None
    tail = float(abs(angle[-1]-stable_eq))
    return {"first_crossing_s":crossing, "first_forward_turn_s":turn,
            "first_forward_turn_rad":max_angle, "uep_rad":saddle, "sep_rad":stable_eq,
            "first_swing_passes":bool(crossing is None and max_angle is not None),
            "settles_in_24_s":bool(crossing is None and tail<1e-3 and abs(speed[-1])<1e-5),
            "peak_u_pu":peak_voltage, "peak_u_at_s":voltage_peak_time,
            "peak_abs_current_pu":float(current.max()), "peak_frequency_pu":float(1+speed.max()),
            "tail_angle_error_rad":tail}, np.column_stack((t,angle,1+speed,voltage,current))


def gain_sweep(vsg, initial, grid, mode, numerical, conditions, verify=False):
    _, saddle = vsg.equilibria(grid)
    dk = conditions["gain_search_step_pu"]
    grid_k = [round(j*dk,10) for j in range(round(numerical["gain_search_upper_k"]/dk)+1)]
    cap = conditions["voltage_headroom_limit_pu"]

    def measure(k, independent=False):
        sol = vsg.simulate(grid,initial,saddle,k,mode,numerical,numerical["sweep_horizon_s"],independent)
        crossing, turn = first_time(sol,0), first_time(sol,1)
        return [k, float(crossing is None and turn is not None), vsg.voltage_peak(sol,grid,initial,k,mode)[0]]

    table = np.array([measure(k) for k in grid_k])
    accepted = table[(table[:,1]==1) & (table[:,2]<=cap)]
    if not len(accepted):
        raise RuntimeError(f"No admissible gains for x={vsg.x}, {mode}")
    upper_k = accepted[-1,0]
    above = table[table[:,0] > upper_k]
    headroom_edge = None
    if len(above) and above[0,1] == 1 and above[0,2]>cap:
        headroom_edge = float(brentq(lambda k: measure(k)[2]-cap,upper_k,above[0,0],xtol=2e-6))
    comparison = {}
    if verify:
        first,last = (int(round(k/dk)) for k in (accepted[0,0],upper_k))
        for i in (first-1,first,last,last+1):
            if 0<=i<len(table):
                k,first_swing,peak = table[i]
                _,check_swing,check_peak = measure(k,independent=True)
                comparison[f"{k:.2f}"] = {"first_swing_agreement":bool(first_swing==check_swing),
                                           "peak_voltage_abs_error_pu":abs(peak-check_peak)}
    return {"x_grid_pu":vsg.x,"mode":mode,"selected_gain_grid_pu":[accepted[0,0],upper_k],
            "gain_count_admitted":len(accepted),"voltage_headroom_edge_pu":headroom_edge,
            "columns":["k_pu","first_swing_passes","peak_u_pu"],
            "independent_boundary_checks":comparison}, table


def calculate(config, original):
    model = settings(config)
    numerical, conditions = config["numerics"], config["literature_conditions"]
    nominal = config["table_i"]["grid_nominal_pu"]
    initial = initial_at(model,nominal)
    traces = {}
    records = {"prefault_state":list(initial),
               "trace_columns":["seconds_since_sag","angle_rad","frequency_pu","u_pu","current_pu"],
               "fig6":{},"fig9":{},"gain_search":{},"independent_checks":{}}

    def run(name, grid, k, mode, independent=False, state_start=initial):
        fixed = state_start[2] if mode=="fixed_voltage" else None
        stable_eq, saddle = model.equilibria(grid,fixed)
        primary = model.simulate(grid,state_start,saddle,k,mode,numerical,numerical["trace_horizon_s"])
        metrics, trace = summarize(model,primary,grid,state_start,k,mode,stable_eq,saddle,numerical["waveform_step_s"])
        traces[name] = trace
        if independent:
            comparator = model.simulate(grid,state_start,saddle,k,mode,numerical,numerical["trace_horizon_s"],independent=True)
            t_end = min(primary.t[-1],comparator.t[-1])
            t = np.linspace(0,t_end,2001)
            gap = np.max(np.abs(primary.sol(t)-comparator.sol(t)),axis=1)
            records["independent_checks"][name] = {"max_abs_state_errors":gap.tolist(),
                 "agree_on_crossing":bool((first_time(primary,0) is None)==(first_time(comparator,0) is None))}
        return metrics

    for grid in conditions["residual_grid_voltages_pu"]:
        key = f"{grid:.2f}"
        records["fig6"][key] = {mode:run(f"fig6_{key}_{mode}",grid,0.,mode,mode == "finite_avr")
                                for mode in ("finite_avr","algebraic_avr","fixed_voltage")}
        published = np.asarray(original["fig6_voltage"][key])
        published = published[published[:,0] >= conditions["step_time_in_fig6_s"]+.08]
        for mode,row in records["fig6"][key].items():
            trace = traces[f"fig6_{key}_{mode}"]
            seconds = published[:,0]-conditions["step_time_in_fig6_s"]
            valid = seconds <= trace[-1,0]
            errors = np.interp(seconds[valid],trace[:,0],trace[:,3])-published[valid,1]
            row["fig6b_voltage_readback_rmse_pu"] = float(np.sqrt(np.mean(errors**2)))
            row["fig6b_voltage_readback_points"] = int(valid.sum())

    published_gains = conditions["fig9_k_pu"]
    grid = min(conditions["residual_grid_voltages_pu"])
    for k in published_gains:
        key = f"{k:.2f}"
        row = run(f"fig9_{key}",grid,k,"finite_avr",k!=0)
        published = np.asarray(original["fig9_phase"][key])
        row["published_curve_start_angle_rad"] = float(published[0,0])
        row["published_curve_visible_max_angle_rad"] = float(published[:,0].max())
        row["published_curve_peak_frequency_pu"] = float(published[:,1].max())
        records["fig9"][key] = row

    initial_published_angle = records["fig9"]["0.30"]["published_curve_start_angle_rad"]
    initial_with_pdf_angle = (initial_published_angle,0.,model.algebraic_voltage(initial_published_angle,0.,nominal))
    records["fig9_pdf_initial_angle_sensitivity"] = {
        "initial_state":list(initial_with_pdf_angle),"k":0.3,
        **run("fig9_0.30_pdf_initial_angle",grid,.3,"finite_avr",True,state_start=initial_with_pdf_angle)}

    stable_eq, saddle = model.equilibria(grid)

    def p_curve(angle, k):
        voltage = model.algebraic_voltage(angle,0,grid,k)
        return model.power(angle,voltage,grid)[0]

    angles = np.linspace(0,pi,501)
    traces["fig8_power_angle"] = np.column_stack((
        angles,[p_curve(a,0) for a in angles],[p_curve(a,1) for a in angles],
        [model.power(a,model.algebraic_voltage(a,0,nominal),nominal)[0] for a in angles]))
    areas = {}
    for k in (0,1):
        mismatch = lambda a: model.pref-p_curve(a,k)
        areas[str(k)] = {"accelerating_pu_rad":quad(mismatch,initial[0],stable_eq,epsabs=1e-12)[0],
                        "decelerating_pu_rad":quad(lambda a:-mismatch(a),stable_eq,saddle,epsabs=1e-12)[0],
                        "equilibria_mismatch_pu":[mismatch(a) for a in (stable_eq,saddle)]}
    records["fig8"] = areas

    scr_model = replace(model,x=1/conditions["stated_scr"])
    for name, vsg, mode in (("table_i",model,"finite_avr"),("text_scr",scr_model,"finite_avr"),
                            ("instantaneous_avr",model,"algebraic_avr")):
        records["gain_search"][name],traces[f"gains_{name}"] = gain_sweep(
            vsg,initial_at(vsg,nominal),grid,mode,numerical,conditions,name=="table_i")

    checks = {f"{name}_independent_Radau":max(r["max_abs_state_errors"])<1e-6 and r["agree_on_crossing"]
              for name,r in records["independent_checks"].items()}
    for g,row in records["gain_search"].items():
        kmin,kmax = row["selected_gain_grid_pu"]
        checks[f"{g}_gain_interval_is_contiguous"] = round((kmax-kmin)/conditions["gain_search_step_pu"])+1==row["gain_count_admitted"]
        checks[f"{g}_voltage_edge_bracketed"] = row["voltage_headroom_edge_pu"] is not None
        for k,c in row["independent_boundary_checks"].items():
            checks[f"{g}_boundary_{k}_independent_Radau"] = c["first_swing_agreement"] and c["peak_voltage_abs_error_pu"]<1e-6
    checks["fig8_equilibria_unchanged"] = max(abs(a) for row in areas.values() for a in row["equilibria_mismatch_pu"])<1e-10
    checks["fig8_area_shift"] = areas["1"]["accelerating_pu_rad"]<areas["0"]["accelerating_pu_rad"] and areas["1"]["decelerating_pu_rad"]>areas["0"]["decelerating_pu_rad"]
    records["checks"] = {key:bool(value) for key,value in checks.items()}
    return records,traces
