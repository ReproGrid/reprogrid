"""Numerical experiments for the paper's balanced faults and switching logic."""

from dataclasses import replace
from math import atan2, expm1, pi, sin

import numpy as np
from scipy.optimize import brentq

from model import Gains, Grid, Implementation, PERIOD, equilibria, marginal_p_solution, phase_run, sample_replay
from stability import critical_damping, forward_outcome


def build_grid(config):
    t = config["table_i"]
    return Grid(t["nominal_hz"], t["line_resistance_pu"], t["line_inductance_pu"], t["nominal_voltage_pu"],
                tuple(t["prefault_current_dq_pu"]), tuple(t["fault_current_dq_pu"]))


def gains_for_cases(config, grid):
    cfg = config["table_ii"]
    damp = (cfg["damping_case_i"], cfg["damping_case_ii_and_iii"])
    low, high = [Gains.from_damping(grid.voltage_prefault, d, cfg["settling_time_s"]) for d in damp]
    return {"case_i": low, "case_ii": high, "case_iii_p_mode": Gains(high.kp, 0.)}


def study_critical(config, grid, digitized, arrays):
    ep = config["numerics"]["separatrix_seed_rad"]
    r = abs(grid.fault_static)
    initial = grid.initial_angle
    nominal = grid.voltage_prefault
    ratios = np.array(config["variants"]["damping_ratio_curve_disturbance_ratios"])
    reproduced = [critical_damping(r/ratio, r, initial, nominal, ep) for ratio in ratios]
    original = np.interp(ratios, *np.array(digitized["fig9"]).T)
    arrays["fig9"] = np.column_stack((ratios, reproduced, original))
    target = config["balanced_cases"]["residual_voltage_pu"][0]
    critical = critical_damping(target, r, initial, nominal, ep)
    refined = critical_damping(target, r, initial, nominal, ep/4)
    bracket = []
    for damping in (critical-.001, critical+.001, config["table_ii"]["damping_case_i"]):
        bracket.append({"damping": damping, "forward_outcome": forward_outcome(damping, target, r, initial, nominal)})
    sensitivity = []
    for angle in config["variants"]["reference_initial_phase_offsets_deg"]:
        phase = initial + angle*pi/180
        sensitivity.append({"phase_offset_deg": angle, "initial_phase_rad": phase,
                            "critical_zeta": critical_damping(target, r, phase, nominal, ep)})
    marker = digitized["fig13"][f"{target:.2f}"]["pre_event_marker_rad_hz"][0]
    marker_zeta = critical_damping(target, r, marker, nominal, ep)
    def case_i_gap(a):
        return critical_damping(target, r, a, nominal, ep) - config["table_ii"]["damping_case_i"]

    upper_angle = initial + 10*pi/180
    threshold_angle = brentq(case_i_gap, initial, upper_angle, xtol=1e-8) if case_i_gap(initial)*case_i_gap(upper_angle) < 0 else None
    return {"fault_v_pu": target, "ratio": r/target, "zeta": critical, "zeta_finer_seed": refined,
            "article_text_zeta": config["balanced_cases"]["reported_critical_damping_at_0p14"],
            "article_vector_zeta": float(np.interp(r/target, *np.array(digitized["fig9"]).T)),
            "forward_boundary_checks": bracket,
            "starting_angle_sensitivity": sensitivity,
            "published_marker_sensitivity": {"angle_rad":marker,"critical_zeta":marker_zeta},
            "critical_initial_phase_offset_deg_for_case_i": None if threshold_angle is None else (threshold_angle-initial)*180/pi,
            "array_columns": ["disturbance_ratio", "same_topology_critical_zeta", "figure9_vector_zeta"]}


def study_phases(config, grid, digitized, arrays):
    horizons = config["numerics"]
    gains = gains_for_cases(config, grid)
    residuals = config["balanced_cases"]["residual_voltage_pu"]
    results = {}
    for voltage in residuals:
        key = f"{voltage:.2f}"
        pdf = digitized["fig13"][key]
        equilibrium = equilibria(voltage, grid.fault_static)
        saddle = equilibrium["lower_separatrix_rad"]
        stable = equilibrium["stable_rad"]
        cases = {}
        for case, gain in gains.items():
            h = horizons["continuous_phase_horizon_s"]
            direct, escaped = phase_run(grid, gain, voltage, grid.current_fault, grid.initial_angle, h, horizons)
            swing, swing_escaped = phase_run(grid, gain, voltage, grid.current_fault, grid.initial_angle, h, horizons, "swing")
            stop = min(len(direct),len(swing))-1
            compare = float(np.max(np.abs(direct[:stop,1:]-swing[:stop,1:])))
            arrays[f"fig13_{key}_{case}"] = direct
            min_angle = np.min(direct[:,1])
            row = {"gains_kp_ki": [gain.kp,gain.ki], "initial_frequency_deviation_hz": float(direct[0,2]),
                   "min_phase_rad_before_-pi_or_time_limit": float(min_angle), "crosses_fault_basin_edge": bool(min_angle < saddle),
                   "angle_at_last_sample_rad": float(direct[-1,1]), "last_time_s": float(direct[-1,0]),
                   "time_to_minus_pi_s": float(direct[-1,0]) if escaped else None,
                   "converges_to_hyperbolic_equilibrium": bool(equilibrium["count"] == 2 and abs(direct[-1,1]-stable)<1e-4),
                   "independent_swing_sequence_abs_error": compare,
                   "reformulations_agree_on_crossing_minus_pi": escaped == swing_escaped}
            if abs(voltage+grid.fault_static) < 1e-13 and gain.ki == 0.:
                exact = marginal_p_solution(direct[:,0], grid.initial_angle, abs(grid.fault_static), gain.kp)
                row["independent_closed_form_max_angle_error_rad"] = float(np.max(np.abs(direct[:,1]-exact)))
            cases[case] = row
        results[key] = {"equilibrium": equilibrium, "cases": cases,
                        "prefault_angle_from_fig13_marker_rad": pdf["pre_event_marker_rad_hz"][0],
                        "adaptive_fig13_curve_frequency_at_phase_zero_hz": pdf["adaptive_frequency_at_zero_angle_hz"],
                        "p_mode_equation_frequency_at_phase_zero_hz": gains["case_iii_p_mode"].kp*grid.fault_static/PERIOD}
    return results


def characterize_samples(chart, events, duration, grid, voltage, dt):
    time = chart[:,0]
    fault = time < duration-1e-12
    indices = np.flatnonzero(fault)
    clear = np.flatnonzero(~fault)[0]
    phase, freq = chart[:,1], chart[:,2]
    fold = equilibria(voltage, grid.fault_static)["lower_separatrix_rad"]

    def first_for(mask):
        candidates = indices[mask[indices]]
        return float(time[candidates[0]]) if len(candidates) else None

    nearest = round((phase[-1]-grid.initial_angle)/PERIOD)
    post = abs(phase[-1]-grid.initial_angle-nearest*PERIOD)
    pq = chart[clear-1]
    iq_angle = atan2(pq[10], pq[9])
    pcc_angle = atan2(pq[5], pq[4])
    return {"total_duration_s": duration, "phase_at_clearing_rad": float(phase[clear]),
            "fault_minimum_frequency_hz": float(np.min(freq[fault])),
            "time_on_lower_frequency_clamp_during_fault_s": float(np.sum(freq[fault]<=45.+1e-10)*dt),
            "fault_separatrix_or_fold_rad": fold,
            "first_crossing_of_edge_s": first_for(phase < fold) if fold is not None else None,
            "first_negative_full_cycle_slip_s": first_for(phase < grid.initial_angle-PERIOD),
            "max_filtered_rocof_in_selected_units_during_fault": float(np.max(chart[fault,6])),
            "p_mode_fraction_during_fault": float(np.mean(chart[fault,7])),
            "at_fault_entry_rocof_in_selected_units": float(chart[0,6]),
            "clearance_current_angle_from_pcc_deg": (pcc_angle-iq_angle)*180/pi,
            "current_reactive_alignment_error_deg": (pcc_angle-iq_angle)*180/pi-90,
            "final_postfault_frequency_hz": float(freq[-1]), "final_nearest_nominal_rotation": nearest,
            "final_nominal_equilibrium_residual_rad": float(post),
            "switching_events": events}


def study_samples(config, grid, arrays):
    gains = gains_for_cases(config, grid)
    gain0, gain1 = gains["case_i"], gains["case_ii"]
    ref = config["rocof"]
    num = config["numerics"]
    dt = num["sample_period_s"]
    sag = config["balanced_cases"]
    published_duration = round(sag["fault_clear_s"]-sag["fault_on_s"], 8)
    options = {"case_i_fix_pi": (gain0, Implementation("pi")),
               "case_ii_fix_pi": (gain1, Implementation("pi")),
               "case_iii_fig12_order": (gain1, Implementation("adaptive")),
               "adaptive_abs_rocof_first": (gain1, Implementation("adaptive", rocof_filter="derivative_then_abs")),
               "adaptive_angular_thresholds": (gain1, Implementation("adaptive", threshold_units="angular")),
               "adaptive_omit_held_integral": (gain1, Implementation("adaptive", integral_policy="hold_omit")),
               "case_i_pi_conditional_integration": (gain0, Implementation("pi", antiwindup=True)),
               "case_ii_pi_conditional_integration": (gain1, Implementation("pi", antiwindup=True)),
               "p_mode_all_the_time": (gain1, Implementation("p")),
               "pll_freeze": (gain1, Implementation("freeze"))}

    def run(voltage, duration, gain, option, key, time_step=dt, chosen_grid=grid):
        chart, events = sample_replay(chosen_grid, gain, voltage, duration, num["postfault_horizon_s"], time_step, ref, option)
        if np.max(np.abs(chart[:,2]-chosen_grid.nominal_hz)) > 5.+1e-10:
            raise AssertionError("Frequency clipping is inconsistent with the reported bounds")
        if key:
            arrays[key] = chart
        return characterize_samples(chart, events, duration, chosen_grid, voltage,time_step)

    short_cases = {f"{v:.2f}": {name: run(v,published_duration,*opts,f"sag_{v:.2f}_{name}")
                                for name,opts in options.items()}
                   for v in sag["residual_voltage_pu"]}
    fold_v = min(sag["residual_voltage_pu"])
    margin_v = max(sag["residual_voltage_pu"])
    alternate_option = options["adaptive_abs_rocof_first"][1]
    refinement = {label: run(fold_v,published_duration,gain,setting, f"refined_{label}",num["sample_period_refined_s"])
                  for label,(gain,setting) in options.items() if label in ("case_iii_fig12_order","adaptive_abs_rocof_first")}
    long_faults = [{"duration_s": duration,
                    "fig12_order": run(fold_v,duration,gain1,Implementation("adaptive"),f"long_{duration}_fig12"),
                    "abs_rocof_first": run(fold_v,duration,gain1,alternate_option,f"long_{duration}_abs_first"),
                    "fixed_p_mode": run(fold_v,duration,gain1,Implementation("p"),f"long_{duration}_p_only")}
                   for duration in config["variants"]["long_fault_duration_s"] if duration>published_duration]
    ramps = [{"ramp_s": ramp,
              "case_i_0p14": run(margin_v,published_duration,gain0,Implementation("pi",current_ramp_s=ramp),f"ramp_{ramp}_case_i"),
              "adaptive_0p10": run(fold_v,published_duration,gain1,Implementation("adaptive",current_ramp_s=ramp),f"ramp_{ramp}_adaptive")}
             for ramp in config["variants"]["fault_current_ramp_s"]]
    r_small = ref["section_iv_b_reference_line_r_pu"]
    small = replace(grid, resistance_pu=r_small)
    diagnostic = run(fold_v,published_duration,gain1,Implementation("adaptive"),"small_r_fig12",chosen_grid=small)
    return {"duration_s": published_duration, "reported_cases": short_cases, "adaptive_refinements": refinement,
            "duration_sensitivity_at_fold": long_faults, "slew_sensitivity": ramps, "small_r_diagnostic": diagnostic}


def study_units(config, grid):
    # Eq. (20) operates on d(δdot)/dt in radians. Scaling to a frequency
    # measurement in cycles/s divides the angular rate derivative by 2 pi.
    rcf = config["rocof"]
    edge = rcf["section_iv_b_reference_fault_edge_s"]
    kp = gains_for_cases(config,grid)["case_ii"].kp
    reference_r = rcf["section_iv_b_reference_line_r_pu"]
    current = config["table_i"]["rated_current_pu"]
    gain = -expm1(-edge/rcf["lpf_s"])/edge
    bound = kp*current*reference_r*gain
    actual = []
    for sag in config["balanced_cases"]["residual_voltage_pu"]:
        angular_step = kp*abs(grid.fault_static-sag*sin(grid.initial_angle))
        actual.append({"sag_pu":sag,"frequency_jump_hz":-angular_step/PERIOD,
                       "eq17_edge_estimate_hz_per_s": angular_step*gain/PERIOD})
    return {"kp_rad_s_per_pu":kp,"reference_r_pu":reference_r,"fault_edge_s":edge,
            "equation20_angular_rad_per_s2":bound,"equation20_hz_per_s":bound/PERIOD,
            "manuscript_hz_per_s":rcf["article_example_hz_per_s"],
            "published_switch_on_hz_per_s":rcf["switch_on_hz_per_s"],
            "table_i_table_ii_sags_eq17":actual}
