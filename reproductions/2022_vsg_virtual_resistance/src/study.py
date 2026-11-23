"""Compare virtual and physical resistance, then compute sag derating limits."""

from dataclasses import replace
from math import pi

import numpy as np

from model import Vsg


TRACE_COLUMNS = ["s","virtual_angle_rad","speed_deviation_rad_s","virtual_voltage_pu",
                 "physical_pcc_active_power_pu","active_power_command_pu","current_pu","physical_pcc_angle_rad"]


def base_model(config):
    p = config["table_i"]
    current_base = p["power_base_w"]/(1.5*p["voltage_phase_peak_v"])
    impedance_base = p["voltage_phase_peak_v"]/current_base
    model = Vsg(p["omega_0_rad_s"],p["omega_0_rad_s"]*p["l_grid_h"]/impedance_base,
                p["p_ref_pu"],p["q_ref_pu"],p["v_ref_pu"],p["j_coeff"],p["dp_coeff"],p["dq_pu"],
                p["r_grid_ohm"]/impedance_base,0.,config["source_tests"]["fault_threshold_voltage_pu"])
    return model,{"current_peak_base_a":current_base,"impedance_base_ohm":impedance_base,
                  "gain_w_per_v_to_pu":p["voltage_phase_peak_v"]/p["power_base_w"]}


def static_curves(model, test, traces):
    axes = np.linspace(0,pi,501)
    records = {"resistance":{},"grid_voltage":{}}
    for kind, settings in (("resistance",[(rg,rv,1.) for rg,rv in test["fig4_pairs_rg_rv_pu"]]),
                           ("grid_voltage",[(0.,0.,voltage) for voltage in test["fig4_grid_voltages_pu"]])):
        for index,(rg,rv,grid) in enumerate(settings,1):
            vsg = replace(model,r_grid=rg,r_virtual=rv)
            peak_angle, peak_margin = vsg.peak_power(grid)
            roots = vsg.positive_equilibria(grid,0.)
            records[kind][str(index)] = {"rg_pu":rg,"rv_pu":rv,"grid_pu":grid,
                 "peak_power_pu":vsg.pref+peak_margin,"peak_power_angle_rad":peak_angle,
                 "equilibria_rad":[] if roots is None else list(roots),
                 "equilibrium_span_rad":None if roots is None else roots[1]-roots[0]}
            traces[f"fig4_{kind}_{index}"] = np.column_stack((axes,[vsg.power_flow(a,grid)[1] for a in axes]))
    return records


def run_curve(model, grid, nominal, gain, numerical, traces, label, compare=False):
    initial = model.initial_angle(nominal)
    equilibria = model.fault_equilibria(grid,gain,initial)
    stable, saddle = equilibria if equilibria is not None else (None,None)
    sol,events = model.simulate(grid,initial,gain,saddle,numerical)
    times = {event.__name__:float(values[0]) if values.size else None for event,values in zip(events,sol.t_events)}
    t = np.linspace(0,sol.t[-1],int(np.ceil(sol.t[-1]/numerical["waveform_step_s"]))+1)
    angle,speed = sol.sol(t)
    voltage,power = np.asarray([model.power_flow(a,grid) for a in angle]).T
    pcc, current = np.asarray([model.circuit(a,grid,v)[2:] for a,v in zip(angle,voltage)]).T
    command = np.array([model.command(v,gain) for v in voltage])
    turn = times["turn"]
    crossed = times.get("cross_saddle")
    tail_error = None if stable is None else float(abs(angle[-1]-stable))
    if times["forward_slip"] is not None:
        outcome = "forward_slip"
    elif times["backward_slip"] is not None:
        outcome = "backward_slip"
    elif tail_error is not None and tail_error<1e-5 and abs(speed[-1])<1e-5:
        outcome = "settles"
    else:
        outcome = "unresolved_within_horizon"
    traces[label] = np.column_stack((t,angle,speed,voltage,power,command,np.abs(current),np.unwrap(np.angle(pcc))))
    checks = {}
    if compare:
        independent,_ = model.simulate(grid,initial,gain,saddle,numerical,independent=True)
        common = t[t<=independent.t[-1]]
        gap = np.max(np.abs(sol.sol(common)-independent.sol(common)),axis=1)
        checks = {"state_max_abs_errors":gap.tolist(),
                  "event_agreement":all(bool(a.size)==bool(b.size) for a,b in zip(sol.t_events,independent.t_events))}
    return {"grid_pu":grid,"rg_pu":model.r_grid,"rv_pu":model.r_virtual,"gain_pu":gain,"initial_angle_rad":initial,
            "initial_post_sag_command_pu":float(command[0]),"equilibria_rad":[] if equilibria is None else list(equilibria),
            "first_forward_turn_rad":None if turn is None or (crossed is not None and crossed < turn) else float(sol.sol(turn)[0]),
            "crossing_s":crossed,"time_at_forward_slip_s":times["forward_slip"],"time_at_backward_slip_s":times["backward_slip"],
            "outcome":outcome,"tail_angle_error_rad":tail_error,"peak_speed_deviation_rad_s":float(speed.max()),
            "peak_current_pu":float(np.abs(current).max()),"final_active_power_pu":float(power[-1]),
            "independent":checks}


def critical_gain(model, grid, nominal, factor, numerical, verify_boundary):
    if model.upper_voltage(grid)>model.detect_voltage:
        raise ValueError("This critical-gain sweep expects the sag detector to stay active.")
    initial = model.initial_angle(nominal)
    low,high = numerical["gain_bracket_w_per_v"]
    tol = numerical["gain_bracket_tolerance_w_per_v"]

    def passes(k,independent=False):
        gain = k*factor
        roots = model.positive_equilibria(grid,gain)
        if roots is None:
            # Large virtual resistance can remove the fault equilibrium at low K.
            return False
        if model.margin(initial,grid,gain)>=0:
            raise ValueError(f"Expected positive initial acceleration at K={k}.")
        sol,_ = model.simulate(grid,initial,gain,roots[1],numerical,first_swing_only=True,independent=independent)
        if sol.t_events[0].size:
            return False
        if sol.t_events[1].size:
            return True
        raise RuntimeError(f"Unresolved first swing at K={k}.")

    if passes(low):
        return [low,low],passes(low,True) if verify_boundary else None
    if not passes(high):
        raise ValueError(f"No first-swing success below K={high} for Rv={model.r_virtual}.")
    while high-low>tol:
        mid = (low+high)/2
        if passes(mid):
            high = mid
        else:
            low = mid
    agreement = not passes(low,True) and passes(high,True) if verify_boundary else None
    return [low,high],agreement


def circuit_checks(model, test):
    max_p_drift,max_v_drift,max_loss_drift = 0.,0.,0.
    for real,virtual in [*test["fig4_pairs_rg_rv_pu"],(model.r_grid,.015)]:
        vsg = replace(model,r_grid=real,r_virtual=virtual)
        for grid in test["fig4_grid_voltages_pu"]:
            for angle in np.linspace(-pi,pi,121):
                voltage,p = vsg.power_flow(angle,grid)
                vi,pii = vsg.power_flow(angle,grid,independent=True)
                actual,imaginary_terminal,_,current = vsg.circuit(angle,grid,voltage)
                max_p_drift = max(max_p_drift,abs(p-pii))
                max_v_drift = max(max_v_drift,abs(voltage-vi))
                max_loss_drift = max(max_loss_drift,abs((imaginary_terminal-actual).real-virtual*abs(current)**2))
    return {"max_p_error_pu":max_p_drift,"max_voltage_error_pu":max_v_drift,
            "virtual_resistance_power_balance_max_error_pu":max_loss_drift}


def calculate(config, original):
    model,bases = base_model(config)
    p,test,numerical = config["table_i"],config["source_tests"],config["numerics"]
    grid,deep,nominal = test["moderate_sag_pu"],test["deep_sag_pu"],p["grid_prefault_pu"]
    factor = bases["gain_w_per_v_to_pu"]
    exact_omega = 2*pi*test["exact_utility_frequency_hz"]
    variants = {
        "table_si":model,
        "printed_pu":replace(model,x=p["l_grid_printed_pu"]),
        "exact_50hz":replace(model,x=exact_omega*p["l_grid_h"]/bases["impedance_base_ohm"],omega=exact_omega)}
    traces = {}
    records = {"bases":bases,"trace_columns":TRACE_COLUMNS,"fig4":static_curves(model,test,traces),
               "fig5":{},"fig10":{},"fig11":{},"deep_sag":{},"gain_unit_sensitivity":{}}
    checks = {}
    for key,b in variants.items():
        table = records["fig5"][key] = {}
        for j,(rg,rv) in enumerate(test["fig5_pairs_rg_rv_pu"],1):
            vsg = replace(b,r_grid=rg,r_virtual=rv)
            table[str(j)] = run_curve(vsg,grid,nominal,0.,numerical,traces,f"fig5_{key}_{j}",key=="table_si")
        table = records["fig10"][key] = {}
        for j,(rv,k) in enumerate(test["fig10_rv_and_gain"],1):
            vsg = replace(b,r_virtual=rv)
            row = run_curve(vsg,grid,nominal,k*factor,numerical,traces,f"fig10_{key}_{j}",key=="table_si")
            row["gain_number_interpreted_as_w_per_v"] = k
            readback = np.asarray(original["fig10_phase"][str(j)])
            row["fig10_published_peak_speed_rad_s"] = float(readback[:,1].max())
            row["fig10_published_start_angle_rad"] = float(readback[0,0])
            table[str(j)] = row

    limiter_model = replace(model,r_virtual=test["deep_sag_rv_pu"])
    for k in test["deep_sag_gain_cases_w_per_v"]:
        records["deep_sag"][str(k)] = run_curve(limiter_model,deep,nominal,k*factor,numerical,traces,
                                                 f"deep_sag_{k}",compare=True)
    for rv,k in test["fig10_rv_and_gain"][2:]:
        vsg = replace(model,r_virtual=rv)
        records["gain_unit_sensitivity"][str(k)] = run_curve(vsg,grid,nominal,k,numerical,traces,
                                                              f"fig10_gain_as_pu_{k}")

    print("Critical derating curves",flush=True)
    for name,b in variants.items():
        row = records["fig11"][name] = {"x_pu":b.x,"omega_rad_s":b.omega,"curves":{}}
        for rg in test["fig11_grid_rg_pu"]:
            rg_label = f"{rg:.3f}"
            ref = np.asarray(original["fig11_gain"][rg_label])
            curve = row["curves"][rg_label] = {}
            dataset = []
            for rv in test["fig11_virtual_resistance_pu"]:
                key = f"{rv:.3f}"
                verify_boundary = key in ("0.020","0.040")
                bracket,agreement = critical_gain(replace(b,r_grid=rg,r_virtual=rv),grid,nominal,factor,numerical,verify_boundary)
                paper = float(np.interp(rv,ref[:,0],ref[:,1])) if ref[0,0]<=rv<=ref[-1,0]+1e-7 else None
                text = original["fig11_text_points"][rg_label].get(f"{rv:.2f}") if round(rv,2)==rv else None
                curve[key] = {"k_bracket_w_per_v":bracket,"pdf_path_readback":paper,"article_text_or_axis_label":text,
                              "verified_bracket_with_rk45_and_kvl":agreement}
                if verify_boundary:
                    checks[f"{name}_{rg_label}_bracket_{key}_independent"] = agreement
                dataset.append([rv,*bracket])
            traces[f"fig11_{name}_rg{rg_label}"] = np.asarray(dataset)
            checks[f"{name}_{rg_label}_critical_gains_nondecreasing"] = bool(np.all(np.diff(np.asarray(dataset)[:,2])>=0))

    identity = records["circuit_checks"] = circuit_checks(model,test)
    checks.update({name:err<1e-11 for name,err in identity.items()})
    for label,states in (("fig5",records["fig5"]["table_si"]),("fig10",records["fig10"]["table_si"]),("deep",records["deep_sag"])):
        for j,state in states.items():
            c = state["independent"]
            checks[f"{label}_{j}_independent_RK45_and_KVL"] = max(c["state_max_abs_errors"])<2e-6 and c["event_agreement"]
    for i,value in records["fig11"]["table_si"]["curves"]["0.003"].items():
        checks[f"fig11_rg_reduces_gain_at_rv_{i}"] = (
            value["k_bracket_w_per_v"][1] <= records["fig11"]["table_si"]["curves"]["0.000"][i]["k_bracket_w_per_v"][1])
    records["checks"] = {key:bool(v) for key,v in checks.items()}
    return records,traces
