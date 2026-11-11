"""Reconstruct robust PSC tuning from the manuscript transfer functions and dq model."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Polynomial
from scipy import signal
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


ROOT = Path(__file__).resolve().parents[1]
S = Polynomial([0., 1.])


def geometry(op, L, V):
    id0, iq0 = op
    current = complex(id0, iq0)
    grid = V - 1j * L * current
    return current, abs(grid), -float(np.angle(grid))


def plant(op, L, V, Ra, wb, kappa):
    """Eq. (9), in tau = omega1*t time. Return ascending-power polynomials."""
    id0, iq0 = op
    a = L * iq0 / V
    b = -Ra**2 / V * (iq0 / L + (id0**2 + iq0**2) / V)
    if wb == 0:
        n = a*S**2 + (1+a+b)
        d = S**2 + (2*Ra/L)*S + 1 + (Ra/L)**2
    else:
        f = S + wb
        n = (a*S**2 + 1+a)*f**2 + b*S**2
        d = (S**2 + 1)*f**2 + (2*Ra/L)*S**2*f + (Ra/L)**2*S**2
    return kappa*V**2/L*n, d, a, b


def loops(op, L, nominal, Kp, Kd, filter_=(Polynomial([1.]), Polynomial([1.]))):
    n, d, a, b = plant(op, L, **nominal)
    lowpass_n, lowpass_d = filter_
    p = Kp*n*lowpass_n, S*d*lowpass_d
    c = p[0], p[0] + p[1]
    dc = Kd*c[0], S*c[1]
    dc_closed = dc[0], dc[0] + dc[1]
    return {"gp": p, "gc": c, "gd": dc, "wc": dc_closed}, (a, b)


def response(tf, w):
    return tf[0](1j*w) / tf[1](1j*w)


def poles(tf):
    return tf[1].roots()


def crossings(w, y, function):
    candidates = np.flatnonzero(np.signbit(y[:-1]) != np.signbit(y[1:]))
    return [brentq(function, w[k], w[k+1], xtol=1e-13) for k in candidates]


def margins(tf, w):
    values = response(tf, w)
    phase_unwrapped = np.unwrap(np.angle(values))
    gain_crossings = []
    for k in crossings(w, abs(values)-1, lambda f: abs(response(tf, f))-1):
        phase = np.angle(response(tf, k))
        reference = np.interp(k, w, phase_unwrapped)
        phase += 2*np.pi*round((reference-phase)/(2*np.pi))
        gain_crossings.append({"w_pu": k, "phase_margin_deg": float(180+np.degrees(phase))})
    phase_crossings = []
    for k in crossings(w, values.imag, lambda f: response(tf, f).imag):
        real = response(tf, k).real
        if real < 0:
            phase_crossings.append({"w_pu": k, "gain_margin": float(-1/real)})
    closed = poles((tf[0], tf[0]+tf[1]))
    return {"gain_margin": min((x["gain_margin"] for x in phase_crossings), default=None),
            "phase_margin_deg": min((x["phase_margin_deg"] for x in gain_crossings), default=None),
            "gain_crossings": gain_crossings, "negative_real_crossings": phase_crossings,
            "closed_poles_pu": [[float(z.real), float(z.imag)] for z in closed],
            "max_closed_real_pu": float(max(closed.real))}


def bandwidth(tf, w):
    dc = abs(tf[0](0)/tf[1](0))
    threshold = dc / np.sqrt(2)
    roots = crossings(w, abs(response(tf, w))-threshold, lambda f: abs(response(tf, f))-threshold)
    return roots[0] if roots else None


def vsm_filter(params, omega, damping):
    # Normalize the physical frequency first; H remains a physical time in seconds.
    M = 2*params["inertia_seconds"]*omega
    Kg = 1/params["droop"]
    if not damping:
        return Polynomial([Kg]), M*S + Kg
    alpha = params["damping_filter_rad_s"]/omega
    return Kg*(S+alpha), (M*S+Kg)*(S+alpha) + params["damping_pu"]*S


def series_csv(path, names, rows):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(names)
        writer.writerows(rows)


def nyquist(ax, tf, w, label, color, style="-"):
    z = response(tf, w)
    ax.plot(z.real, z.imag, color=color, ls=style, label=label)
    ax.plot(z.real, -z.imag, color=color, ls=style)
    ax.plot(-1, 0, "x", color=".15", ms=6)
    ax.set(xlim=(-1.55, .06), ylim=(-.69, .69), xlabel="Real", ylabel="Imaginary")
    ax.grid(alpha=.2)


def local_step(op, L, nominal, Kp, Kd, channel, solver, tau, eps):
    V, Ra, wb, kappa = (nominal[k] for k in ("V", "Ra", "wb", "kappa"))
    i0, Vg, delta = geometry(op, L, V)
    P0 = kappa * V * i0.real
    equilibrium = np.array([i0.real, i0.imag, delta, i0.real, i0.imag, 0.])

    def evaluate(state, amplitude):
        i = state[0]+1j*state[1]
        h = state[3]+1j*state[4]
        v = V-Ra*(i-h)
        p = kappa*np.real(v*np.conj(i))
        pref = P0+amplitude if channel == "active" else P0+Kd*(state[5]-amplitude)
        delta_rate = Kp*(pref-p)
        di = (v-Vg*np.exp(-1j*state[2]))/L - 1j*(1+delta_rate)*i
        dh = wb*(i-h)
        dw = 0. if channel == "active" else P0-p
        return np.array([di.real, di.imag, delta_rate, dh.real, dh.imag, dw]), p

    residual = float(np.max(np.abs(evaluate(equilibrium, 0.)[0])))
    sol = solve_ivp(lambda _, y: evaluate(y, eps)[0], (tau[0], tau[-1]), equilibrium,
                    t_eval=tau, **solver)
    if not sol.success:
        raise RuntimeError(sol.message)
    if channel == "active":
        trace = np.array([(evaluate(y, eps)[1]-P0)/eps for y in sol.y.T])
    else:
        trace = sol.y[5]/eps
    return trace, residual


def appendix_audit(cfg, Kd, w, output):
    params = cfg["appendix_audit"]
    nominal = {**cfg["nominal"], "wb": params["filter"]}
    Ra, V, kappa = (nominal[k] for k in ("Ra", "V", "kappa"))
    Kp = Ra/(kappa*V**2)
    rows = []
    cases = [(float(scr), op) for scr in params["scr"] for op in params["current_points"]]
    cases.append((1/(np.sqrt(2)*Ra), [0., 0.]))
    for scr, op in cases:
        L = 1/scr
        systems, (a, b) = loops(op, L, nominal, Kp, Kd)
        q = 1+a+b
        alpha = Ra/L
        wp = np.sqrt(1+alpha**2)
        wd = np.sqrt(q/(2+a))
        printed_wd = np.sqrt((1-b)/(2+a))
        gm_p = 2*(1+alpha**2)/(1+b-a*alpha**2)
        printed_gm_p = 2*(1+alpha**2)/(1-(Ra*np.linalg.norm(op)/V)**2)
        gm_d = ((1-b)/(2*(2+a)*alpha) + alpha/2)/Kd
        printed_gm_d = ((1-b)/((2+a)*alpha) + alpha/2)/Kd
        eq43 = np.sqrt((1-b)/(2+a))/Kd
        mp, md = [margins(systems[k], w) for k in ("gp", "gd")]
        rows.append({
            "scr": float(scr), "L_pu": float(L), "id": op[0], "iq": op[1], "a": a, "b": b,
            "p_w_pi_formula": float(wp), "p_w_pi_numeric": mp["negative_real_crossings"][0]["w_pu"],
            "p_gm_eq39_first": float(gm_p), "p_gm_printed_eq40": float(printed_gm_p), "p_gm_numeric": mp["gain_margin"],
            "dc_w_pi_derived": float(wd), "dc_w_pi_printed": float(printed_wd),
            "dc_w_pi_numeric": md["negative_real_crossings"][0]["w_pu"],
            "dc_gm_derived": float(gm_d), "dc_gm_printed_eq42": float(printed_gm_d),
            "dc_gm_numeric": md["gain_margin"], "dc_gm_lower_bound_eq43": float(eq43)})
    series_csv(output / "appendix_audit.csv", list(rows[0]), [x.values() for x in rows])
    tol = cfg["numerics"]["closed_form_abs_tolerance"]
    checks = {
        "appendix_II_crossover_matches": all(abs(r["p_w_pi_numeric"]-r["p_w_pi_formula"])<tol for r in rows),
        "appendix_II_eq39_first_expression_matches_and_ge_2": all(abs(r["p_gm_numeric"]-r["p_gm_eq39_first"])<tol and r["p_gm_numeric"]>=2-tol for r in rows),
        "appendix_IV_corrected_crossover_matches": all(abs(r["dc_w_pi_numeric"]-r["dc_w_pi_derived"])<tol for r in rows),
        "appendix_IV_corrected_gain_margin_matches": all(abs(r["dc_gm_numeric"]-r["dc_gm_derived"])<tol for r in rows),
        "appendix_IV_eq43_bound_holds": all(r["dc_gm_numeric"]>=r["dc_gm_lower_bound_eq43"]-tol for r in rows),
        "appendix_IV_zero_current_minimum_equals_four": abs(rows[-1]["dc_gm_numeric"]-4)<tol}
    return rows, checks


def reactive_sweep(cfg, Kp, Kd, w, output, figures):
    spec = cfg["extensions"]["reactive_sweep"]
    L = 1/spec["scr"]
    rows = []
    for iq in np.linspace(spec["iq_min"], spec["iq_max"], spec["count"]):
        op = [spec["id"], float(iq)]
        _, Vg, theta = geometry(op, L, cfg["nominal"]["V"])
        systems, (a, _) = loops(op, L, cfg["nominal"], Kp, Kd)
        md = margins(systems["gd"], w)
        rows.append({"iq_pu": float(iq), "a": a, "Vg_pu": Vg, "delta0_deg": float(np.degrees(theta)),
                     "bandwidth_gc_pu": bandwidth(systems["gc"], w),
                     "bandwidth_eq21_pu": (1+a)*cfg["nominal"]["Ra"]/L,
                     "dc_phase_margin_deg": md["phase_margin_deg"],
                     "dc_max_closed_real_pu": md["max_closed_real_pu"]})
    series_csv(output / "reactive_sweep.csv", list(rows[0]), [x.values() for x in rows])
    x = [r["iq_pu"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), layout="constrained")
    axes[0].semilogy(x, [r["bandwidth_gc_pu"] for r in rows], label="Finite filter, −3 dB")
    axes[0].semilogy(x, [r["bandwidth_eq21_pu"] for r in rows], "--", label="Approx. Eq. (21)")
    axes[0].set(ylabel="Active-power bandwidth (pu)")
    axes[0].legend()
    axes[1].plot(x, [r["dc_phase_margin_deg"] for r in rows])
    axes[1].axhline(45, color=".4", ls=":", label="45° reference")
    axes[1].axhline(0, color="#aa452d", ls="--")
    axes[1].set(ylabel="DC-link phase margin (degrees)")
    for ax in axes:
        ax.set(xlabel="Steady q-axis current (pu)")
        ax.grid(alpha=.22)
    fig.suptitle("High reactive injection extension (SCR = 1, id = 0)")
    fig.savefig(figures / "reactive_boundary.png", dpi=160)
    plt.close(fig)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--figures", type=Path, default=ROOT / "figures")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.figures.mkdir(parents=True, exist_ok=True)
    start = datetime.now(timezone.utc)
    timer = perf_counter()
    cfg_path = ROOT / "configs/protocol.json"
    cfg = json.loads(cfg_path.read_text())
    source = json.loads((ROOT / "sources/source_manifest.json").read_text())
    nominal, num = cfg["nominal"], cfg["numerics"]
    Kp, Kd = nominal["Ra"]/(nominal["kappa"]*nominal["V"]**2), 1/(4*np.sqrt(2))
    omega = 2*np.pi*cfg["base_frequency_hz"]
    w = np.geomspace(*num["margin_search_pu"])
    common = np.geomspace(*num["cross_frequency_pu"])
    spectra = {"w_pu": common}
    tau = np.linspace(0, num["local_end_tau"], num["local_samples"])
    traces = {"tau": tau, "time_s": tau/omega}
    checks = {}
    fig5, axes5 = plt.subplots(2, 2, figsize=(8, 7), layout="constrained")
    figtime, ax_time = plt.subplots(2, 2, figsize=(9, 6), layout="constrained")
    fig5_cases, steps, alternate = {}, {}, {}
    colors = {1: "#306c97", 3: "#ba632d", 10: "#287e55"}
    for row, group in enumerate(("active", "reactive")):
        op = cfg["fig5"][group]
        for scr in cfg["fig5"]["scr"]:
            key = f"fig5_{group}_scr{scr}"
            L = 1/scr
            current, Vg, delta = geometry(op, L, nominal["V"])
            systems, (a, b) = loops(op, L, nominal, Kp, Kd)
            for channel in ("gp", "gc", "gd"):
                spectra[f"{key}_{channel}"] = response(systems[channel], common)
            fig5_cases[key] = {"op": op, "current_magnitude_pu": abs(current), "scr": scr, "Vg_pu": Vg,
                               "delta0_deg": float(np.degrees(delta)), "a": a, "b_frozen": b,
                               "gp": margins(systems["gp"], w), "gd": margins(systems["gd"], w),
                               "active_bandwidth_pu": bandwidth(systems["gc"], w)}
            for col, name in enumerate(("gp", "gd")):
                nyquist(axes5[row, col], systems[name], w, f"SCR {scr}", colors[scr])
                axes5[row, col].set_title(f"{group.capitalize()} case, {name.upper()}")
            for col, channel in enumerate(("active", "dc")):
                tf = systems["gc" if channel == "active" else "wc"]
                _, linear = signal.step((tf[0].coef[::-1], tf[1].coef[::-1]), T=tau)
                nonlinear, residual = local_step(op, L, nominal, Kp, Kd, channel, num["solver"], tau, num["local_step_pu"])
                refined, _ = local_step(op, L, nominal, Kp, Kd, channel, num["refinement"], tau, num["local_step_pu"])
                step_key = f"{key}_{channel}"
                traces[f"{step_key}_linear"] = linear
                traces[f"{step_key}_nonlinear"] = nonlinear
                error = float(np.max(abs(linear-nonlinear)))
                integrator = float(np.max(abs(nonlinear-refined)))
                steps[step_key] = {"linear_nonlinear_max_relative_to_step": error,
                                   "DOP853_Radau_max_relative_to_step": integrator, "equilibrium_residual": residual}
                checks[f"{step_key}_local_agreement"] = error < num["local_relative_to_step_tolerance"]
                checks[f"{step_key}_solver_refinement"] = integrator < num["integrator_relative_to_step_tolerance"]
                checks[f"{step_key}_equilibrium"] = residual < 1e-10
                ax = ax_time[row, col]
                ax.plot(tau/omega, nonlinear, color=colors[scr], label=f"SCR {scr}")
                ax.plot(tau/omega, linear, "--", color=colors[scr], alpha=.65)
                ax.set(title=f"{group.capitalize()} case, {channel} reference",
                       xlabel="Time (s)", ylabel="Response / step amplitude")
                ax.grid(alpha=.2)
            alt = cfg["extensions"]["table_I_Ra"]
            alt_ra = alt["actual_ohm"]/alt["base_ohm"]
            alt_systems, _ = loops(op, L, {**nominal, "Ra": alt_ra}, alt_ra/(nominal["kappa"]*nominal["V"]**2), Kd)
            alternate[key] = {name: margins(alt_systems[name], w) for name in ("gp", "gd")}
    for ax in (*axes5.flat, *ax_time.flat):
        ax.legend(fontsize=8)
    fig5.suptitle("Fig. 5 reconstruction with the finite active-resistance filter")
    fig5.savefig(args.figures / "fig5_nyquist.png", dpi=160)
    plt.close(fig5)
    figtime.suptitle("Small-step response: ideal nonlinear dq (solid) and linear (dashed)")
    figtime.savefig(args.figures / "local_step_response.png", dpi=160)
    plt.close(figtime)

    example = cfg["fig7_8"]
    op = [example["id"], example["iq"]]
    L = 1/example["scr"]
    psc, _ = loops(op, L, nominal, Kp, Kd)
    vsm, _ = loops(op, L, nominal, example["droop"], Kd)
    vsm_reduced, _ = loops(op, L, nominal, example["droop"], Kd*example["Kd_reduced_fraction"])
    inertial, _ = loops(op, L, nominal, example["droop"], Kd, vsm_filter(example, omega, False))
    damped, _ = loops(op, L, nominal, example["droop"], Kd, vsm_filter(example, omega, True))
    fig78, axes78 = plt.subplots(1, 3, figsize=(12, 3.6), layout="constrained")
    variants = {"psc": psc, "vsm0": vsm, "vsm0_kd_30pct": vsm_reduced, "vsm_H5": inertial, "vsm_H5_KD50": damped}
    for ax, title, channel, names in zip(
        axes78, ("Fig. 7(a)", "Fig. 7(b)", "Fig. 8"), ("gp", "gd", "gp"),
        (("psc", "vsm0"), ("psc", "vsm0", "vsm0_kd_30pct"), ("vsm0", "vsm_H5", "vsm_H5_KD50"))):
        for name, color, style in zip(names, ("#306c97", "#ba632d", "#287e55"), ("-", "--", "-.")):
            nyquist(ax, variants[name][channel], w, name, color, style)
        ax.set_title(title)
        ax.legend(fontsize=7)
    fig78.savefig(args.figures / "fig7_8_nyquist.png", dpi=160)
    plt.close(fig78)
    fig78_cases = {name: {loop: margins(system[loop], w) for loop in ("gp", "gd")} for name, system in variants.items()}
    for name, system in variants.items():
        for channel in ("gp", "gc", "gd"):
            spectra[f"fig7_8_{name}_{channel}"] = response(system[channel], common)
    appendix, appendix_checks = appendix_audit(cfg, Kd, w, args.output)
    checks.update(appendix_checks)
    sweep = reactive_sweep(cfg, Kp, Kd, w, args.output, args.figures)
    np.savez_compressed(args.output / "frequency_responses.npz", **spectra)
    np.savez_compressed(args.output / "local_steps.npz", **traces)
    series_csv(args.output / "local_steps.csv", list(traces), zip(*traces.values()))
    result = {
        "doi": cfg["doi"], "scope": cfg["scope"], "source_sha256": source["sha256"],
        "protocol_sha256": hashlib.sha256(cfg_path.read_bytes()).hexdigest(),
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "started_utc": start.isoformat(), "finished_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": perf_counter()-timer, "python": platform.python_version(), "platform": platform.platform(),
        "packages": {pkg: version(pkg) for pkg in ("numpy", "scipy", "matplotlib")},
        "gains_pu": {"Ra": nominal["Ra"], "Kp": Kp, "Kd": float(Kd)},
        "fig5": fig5_cases, "fig7_8": fig78_cases, "local_steps": steps, "appendix_audit": appendix,
        "table_I_Ra_sensitivity": {"Ra_pu": alt_ra, "fig5_cases": alternate},
        "reactive_sweep": {"strongest_injection": sweep[0], "zero_current": sweep[-1],
                           "unstable_dc_cases": sum(r["dc_max_closed_real_pu"]>0 for r in sweep), "count": len(sweep)},
        "checks": checks, "numerical_checks_passed": all(checks.values())}
    (args.output / "run_manifest.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"checks": len(checks), "failed": [k for k,v in checks.items() if not v],
                      "wall_seconds": result["wall_seconds"]}))
    return 0 if result["numerical_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
