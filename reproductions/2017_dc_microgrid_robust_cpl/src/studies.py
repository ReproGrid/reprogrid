"""Numerical experiments for Section IV and independently verified comparisons."""

from dataclasses import replace
from itertools import product

import numpy as np
from scipy.linalg import eigvals
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, linear_sum_assignment, root

from certificates import CertificateBox, normalized_lyapunov, spectral_abscissa, vertex_decay
from network import Network, graph_symmetries, simulate


def leading_mode(A):
    modes = eigvals(A)
    pole = max(modes[modes.imag >= 0], key=lambda z: z.real)
    return {"rightmost_real_per_s": float(pole.real), "frequency_hz": float(pole.imag / (2 * np.pi))}


def operating_point(network, power):
    x = network.equilibrium(power)
    load = x[network.load_v]
    bus = x[network.buses]
    line_current = network.incidence.T @ bus / network.c.line_r_ohm
    return {"load_v": load.tolist(), "bus_v": bus.tolist(),
            "max_line_current_a": float(np.max(np.abs(line_current), initial=0)),
            "delta_per_s": network.delta(x, power).tolist(), **leading_mode(network.jacobian(network.delta(x, power)))}


def assembly_diagnostics(network, power):
    equilibrium = network.equilibrium(power)
    # The branch solver starts from a distinct, full-dimensional seed and
    # uses neither the reduced DC conductance system nor the assembled A matrix.
    zero_seed = equilibrium.copy()
    zero_seed[network.sources] = 0
    zero_seed[network.loads] = 0
    zero_seed[network.buses] = network.c.reference_v
    zero_seed[network.load_v] = network.c.reference_v
    branch = root(lambda x: network.branch_rhs(x, power), zero_seed, tol=1e-11)
    check = equilibrium + np.linspace(-1.5, 2, network.n)
    h = 1e-3
    directions = h * np.eye(network.n)
    finite_diff = np.column_stack([(network.branch_rhs(check + dx, power) - network.branch_rhs(check - dx, power)) / (2*h)
                                   for dx in directions])
    return {"branch_equilibrium_state_max_error": float(np.max(np.abs(equilibrium - branch.x))),
            "equilibrium_branch_residual_si_per_s": float(np.max(np.abs(network.branch_rhs(equilibrium, power)))),
            "off_equilibrium_rhs_error_si_per_s": float(np.max(np.abs(network.rhs(check, power) - network.branch_rhs(check, power)))),
            "finite_difference_jacobian_error_per_s": float(np.max(np.abs(finite_diff - network.jacobian(network.delta(check, power))))),
            "equilibrium_power_balance_error_w": abs(network.power_imbalance_w(equilibrium, power))}


def voltage_envelope(network, low_power, high_power, v_floor):
    lo = network.equilibrium(high_power)[network.load_v]
    hi = network.equilibrium(low_power)[network.load_v]
    H = network.load_impedance
    lipschitz = np.linalg.norm(H, ord=np.inf) * high_power / v_floor**2
    return {"min_load_voltage_v": lo.tolist(), "max_load_voltage_v": hi.tolist(),
            "dc_impedance_min_ohm": float(np.min(H)), "lipschitz_bound_in_paper_voltage_range": float(lipschitz),
            "delta_lower_per_s": (low_power / (network.c.load_c_f * hi**2)).tolist(),
            "delta_upper_per_s": (high_power / (network.c.load_c_f * lo**2)).tolist()}


def store_certificate(box, result, P, name, arrays):
    if P is not None:
        arrays[f"{name}_P"] = P
        arrays[f"{name}_decay_per_s"] = vertex_decay(box.matrices, P)
    result["matrix_key"] = f"{name}_P" if P is not None else None
    return result


def check_box(network, low, high, numerics, permutations, name, arrays):
    box = CertificateBox(network, low, high, numerics)
    results = {"summary": box.summary(), "equations": {}}
    arrays[f"{name}_deltas_per_s"] = box.deltas
    arrays[f"{name}_rates_per_s"] = box.rates
    for equation, run in (("7", lambda: box.eq7(permutations=permutations)), ("9", box.eq9), ("10", box.eq10)):
        row, P = run()
        results["equations"][equation] = store_certificate(box, row, P, f"{name}_eq{equation}", arrays)
    return results


def nine_buses(components, protocol, arrays):
    cfg = protocol["nine_bus"]
    low, high = cfg["cpl_limits_w"]
    vf, vc = cfg["equilibrium_load_voltage_bounds_v"]
    box_min = low / (components.load_c_f * vc**2)
    box_max = high / (components.load_c_f * vf**2)
    output = {}
    for droop in cfg["droop_ohm"]:
        print(f"Nine-bus network and certificates, d={droop:.2f} ohm", flush=True)
        id_ = f"{droop:.2f}"
        name = f"nine_{round(100 * droop):02}"
        network = Network(cfg, components, droop)
        powers = np.linspace(*cfg["uniform_power_grid_w"])
        modes = []
        alignment = 0
        for power in powers:
            delta = network.delta(network.equilibrium(power), power)
            modal = network.modal_eigenvalues(delta[0])
            full = eigvals(network.jacobian(delta))
            distances = np.abs(full[:, None] - modal.flatten())
            i, j = linear_sum_assignment(distances)
            alignment = max(alignment, float(np.max(distances[i, j])))
            modes.append(np.max(modal.real, axis=1))
        arrays[f"{name}_uniform_powers_w"] = powers
        arrays[f"{name}_modal_rates_per_s"] = np.array(modes)
        arrays[f"{name}_conductance_modes_siemens"] = np.linalg.eigvalsh(network.laplacian)
        envelope = voltage_envelope(network, low, high, vf)
        permutations = graph_symmetries(network, cfg)
        mixed = np.array(list(product((low, high), repeat=network.n_loads)))
        points = [operating_point(network, power) for power in mixed]
        rates = np.array([row["rightmost_real_per_s"] for row in points])
        load_voltages = np.array([row["load_v"] for row in points])
        arrays[f"{name}_power_corners_w"] = mixed
        arrays[f"{name}_power_corner_rates_per_s"] = rates
        arrays[f"{name}_power_corner_load_voltages_v"] = load_voltages
        first = np.where(np.max(modes, axis=1) >= 0)[0]
        boundary = None
        if len(first) and first[0] > 0:
            index = int(first[0])
            critical = brentq(lambda p: spectral_abscissa(network.jacobian(network.delta(network.equilibrium(p), p))),
                             powers[index-1], powers[index], xtol=1e-8)
            delta = network.delta(network.equilibrium(critical), critical)
            modal_rates = np.max(network.modal_eigenvalues(delta[0]).real, axis=1)
            boundary = {"watts_per_load": critical, "critical_mode_index": int(np.argmax(modal_rates)),
                        "next_spatial_mode_real_per_s": float(np.max(modal_rates[1:])), **operating_point(network, critical)}
        output[id_] = {"uniform_power_boundary": boundary, "modal_full_matrix_error_per_s": alignment,
                       "at_15kw": operating_point(network, cfg["nominal_cpl_w"]), "at_20kw": operating_point(network, high),
                       "power_corner_survey": {"count": len(mixed), "unstable": int(np.sum(rates > 0)),
                                               "min_load_v": float(load_voltages.min()), "max_load_v": float(load_voltages.max()),
                                               "max_real_per_s": float(max(rates)), "worst_power_w": mixed[np.argmax(rates)].tolist(),
                                               "largest_line_current_a": max(row["max_line_current_a"] for row in points)},
                       "dc_equilibrium_envelope": envelope,
                       "assembly_checks": assembly_diagnostics(network, np.linspace(low, high, network.n_loads)),
                       "paper_delta_box": check_box(network, box_min, box_max, protocol["numerics"], permutations,
                                                    f"{name}_paper_box", arrays),
                       "physical_delta_box": check_box(network, min(envelope["delta_lower_per_s"]), max(envelope["delta_upper_per_s"]),
                                                       protocol["numerics"], permutations, f"{name}_physical_box", arrays)}
    return output


def eight_buses(components, protocol, arrays):
    cfg = protocol["eight_bus"]
    numerics = protocol["numerics"]
    output = {}
    for line_r in cfg["line_r_ohm"]:
        print(f"Eight-bus paper table and certificate limits, R={line_r:.1f} ohm", flush=True)
        network = Network(cfg, replace(components, line_r_ohm=line_r), cfg["droop_ohm"])
        permutations = graph_symmetries(network, cfg)
        paper = cfg["paper_delta_upper_per_s"][str(line_r)]
        def upper_real(delta):
            return spectral_abscissa(network.jacobian(delta))

        upper_eigen_boundary = brentq(upper_real, 0, 350)
        eq9_boundary = brentq(lambda d: 1 / normalized_lyapunov(network.jacobian(d))[1] - 2*d,
                              0, upper_eigen_boundary - 1e-6)
        trace = {"7": {}, "9": {}, "10": {}}

        def trial(equation, delta):
            key = f"{delta:.8f}"
            if key not in trace[equation]:
                box = CertificateBox(network, cfg["box_delta_floor_per_s"], delta, numerics)
                if equation == "7":
                    row, P = box.eq7(permutations=permutations)
                else:
                    row, P = getattr(box, "eq" + equation)()
                name = f"eight_{line_r:g}_eq{equation}_{delta:.8f}".replace(".", "p")
                trace[equation][key] = store_certificate(box, row, P, name, arrays)
            return trace[equation][key]["verdict"]

        eq7_candidate = paper["7"] - 1
        for delta in [paper["7"], eq7_candidate, eq7_candidate - 1, eq7_candidate - 2]:
            if trial("7", delta) == "certified":
                break
        if "9" in paper:
            trial("9", paper["9"])
        for delta in (int(eq9_boundary), int(eq9_boundary) + 1):
            trial("9", delta)

        # The Eq. (10) upper and lower witnesses are checked after solving.
        # Unresolved points are kept in the sweep but cannot shrink the bracket.
        trial("10", paper["10"])
        lo, hi = 0.0, float(np.ceil(upper_eigen_boundary))
        trial("10", lo)
        trial("10", hi)
        while hi - lo > cfg["boundary_resolution_per_s"]:
            mid = (lo + hi) / 2
            verdict = trial("10", mid)
            if verdict == "certified":
                lo = mid
            elif verdict in ("strict_lmi_infeasible", "rhp_vertex"):
                hi = mid
            else:
                break
        eq7_certified = [float(k) for k, r in trace["7"].items() if r["verdict"] == "certified"]
        eq7_negative = [float(k) for k, r in trace["7"].items() if r["verdict"] in ("rhp_vertex", "strict_lmi_infeasible")]
        box = CertificateBox(network, cfg["box_delta_floor_per_s"], paper["10"], numerics)
        output[str(line_r)] = {"paper_upper_limits_per_s": paper, "states": network.n, "vertices": 2**network.n_loads,
                               "upper_point_hopf_per_s": upper_eigen_boundary, "eq9_analytic_boundary_per_s": eq9_boundary,
                               "eq7_box_bound_per_s": [max(eq7_certified, default=0), min([upper_eigen_boundary, *eq7_negative])],
                               "eq10_boundary_bracket_per_s": [lo, hi], "eq10_direct_full_lmi_at_published_delta": box.eq10_direct(),
                               "trials": trace}
    return output


def eight_bus_practice(components, protocol, eight):
    cfg = protocol["eight_bus"]
    c = replace(components, line_r_ohm=0.3)
    droop = cfg["droop_ohm"]
    watts = 20_000

    def get_network(voltage):
        return Network(cfg, replace(c, reference_v=voltage), droop)

    def uniform_boundary(network):
        return brentq(lambda p: spectral_abscissa(network.jacobian(network.delta(network.equilibrium(p), p))), 11_000, 24_000)

    at_400, at_440 = get_network(400), get_network(440)
    claimed_delta = cfg["paper_delta_upper_per_s"]["0.3"]["7"]
    verified_delta = eight["0.3"]["eq7_box_bound_per_s"][0]
    result = {"load_power_w_per_load": watts, "load_buses": cfg["loads"],
              "nominal_440v_operating_point": operating_point(at_440, watts),
              "uniform_power_hopf_400v_w": uniform_boundary(at_400),
              "uniform_power_hopf_440v_w": uniform_boundary(at_440),
              "reference_v_where_20kw_equilibrium_changes_stability": brentq(
                  lambda v: operating_point(get_network(v), watts)["rightmost_real_per_s"], 440, 480),
              "reference_v_needed_for_all_loads_at_least_410_v": brentq(
                  lambda v: min(get_network(v).equilibrium(watts)[-len(cfg["loads"]):]) - 410, 440, 480),
              "voltage_floor_from_paper_eq7_delta_v": np.sqrt(watts / (c.load_c_f * claimed_delta)),
              "kw_limit_at_360_v_from_paper_eq7_delta": claimed_delta * c.load_c_f * 360**2 / 1000,
              "assembly_checks": assembly_diagnostics(at_440, watts)}
    if verified_delta > 0:
        v_floor = np.sqrt(watts / (c.load_c_f * verified_delta))
        result["voltage_floor_from_certified_eq7_delta_v"] = v_floor
        result["reference_for_certified_delta_at_20kw_v"] = brentq(
            lambda v: min(get_network(v).equilibrium(watts)[-len(cfg["loads"]):]) - v_floor, 440, 490)
    return result


def time_domain(components, protocol, readback, arrays):
    output = {}
    for key, case in protocol["trajectories"].items():
        print(f"Replaying Fig. {key}", flush=True)
        topology = protocol["nine_bus"] if case["buses"] == 9 else protocol["eight_bus"]
        c = replace(components, line_r_ohm=case["line_r_ohm"], reference_v=case["reference_v"])
        network = Network(topology, c, case["droop_ohm"])
        time, states, power, segments = simulate(network, case, protocol["numerics"])
        if case["probe_state"] == "bus_voltage":
            probe = network.buses.start + case["probe_bus"] - 1
        else:
            probe = network.load_v.start + list(network.load_buses).index(case["probe_bus"])
        plotted_v = states[:, probe]
        arrays[f"fig{key}_time_s"] = time
        arrays[f"fig{key}_power_w"] = power
        arrays[f"fig{key}_states_si"] = states
        arrays[f"fig{key}_probe_v"] = plotted_v
        arrays[f"fig{key}_min_cpl_v"] = np.min(states[:, network.load_v], axis=1)

        points = []
        for ref in readback["figures"][key]["plateau_windows"]:
            p = ref["power_kw"] * 1000
            a, b = ref["window_s"]
            in_window = (time >= a) & (time <= b)
            v = plotted_v[in_window]
            eq = network.equilibrium(p)
            points.append({**ref, "equilibrium_probe_v": float(eq[probe]),
                           "equilibrium_min_cpl_v": float(np.min(eq[network.load_v])),
                           **leading_mode(network.jacobian(network.delta(eq, p))),
                           "model_median_v": float(np.median(v)) if len(v) else None,
                           "model_span_v": float(np.ptp(v)) if len(v) else None})
        read = readback["figures"][key]
        intended = case["first_step_s"] + case["step_period_s"] * np.arange(case["steps"])
        shown_and_settled = [row for row in points if row["pdf_median_v"] is not None and row["pdf_span_v"] < .1]
        result = {"profiles": segments, "probe": case["probe_state"], "probe_bus": case["probe_bus"],
                  "reference_v": case["reference_v"], "states": network.n, "stop_s": float(time[-1]),
                  "minimum_probe_voltage_v": float(np.min(plotted_v)),
                  "minimum_cpl_voltage_v": float(np.min(states[:, network.load_v])),
                  "max_schedule_readback_error_s": float(np.max(np.abs(intended - read["step_pdf_midpoint_s"]))),
                  "max_settled_apparent_plateau_error_v": max(abs(row["equilibrium_probe_v"] - row["pdf_median_v"]) for row in shown_and_settled),
                  "comparison_by_power": points}
        if key in ("4", "8"):
            # Check the sensitive final segment with a different integrator.
            # Starting from the same pre-step state isolates numerical growth.
            last = segments[-1]["start_s"]
            sample_index = int(np.flatnonzero(time == last)[-1])
            step_time = time[sample_index:]
            p = power[sample_index]
            bdf = solve_ivp(lambda _, x: network.branch_rhs(x, p), (last, time[-1]), states[sample_index],
                            method="BDF", rtol=protocol["numerics"]["integration_rtol"] / 5,
                            atol=protocol["numerics"]["integration_atol"] / 5, t_eval=step_time)
            if not bdf.success:
                raise RuntimeError(bdf.message)
            result["last_step_independent_branch_bdf_max_probe_error_v"] = float(np.max(np.abs(bdf.y[probe] - plotted_v[sample_index:])))
            arrays[f"fig{key}_bdf_probe_v"] = bdf.y[probe]
        output[key] = result
    return output
