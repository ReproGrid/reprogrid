"""Independent RFPSC verification from the accepted paper, no primary code imports.

State variables use the GRID synchronous frame. Time is tau=omega_base*t.
H=0 means zero filter PERTURBATION, not zero equilibrium filter current.
Run: verification/.venv/bin/python verification/verifier.py
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from dataclasses import dataclass, replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".mplconfig"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import signal
from scipy.integrate import solve_ivp
from scipy.linalg import expm
import sympy as sp

OMEGA_BASE = 2 * np.pi * 50
PROTOCOL = json.loads((HERE / "protocol.json").read_text())


@dataclass(frozen=True)
class Parameters:
    L: float = 0.5
    Ra: float = 0.2
    V: float = 1.0
    omega: float = 1.0
    kappa: float = 1.0
    wb: float = 0.1
    id0: float = 0.0
    iq0: float = 0.0
    xi: int = 1
    full_filter: bool = False

    @property
    def kp(self):
        return self.omega * self.Ra / (self.kappa * self.V**2)

    @property
    def i0(self):
        return self.id0 + 1j * self.iq0

    @property
    def r0(self):
        return self.kappa * self.V * self.id0

    @property
    def equilibrium_grid_voltage(self):
        return self.V - 1j * self.omega * self.L * self.i0

    @property
    def Vg(self):
        return abs(self.equilibrium_grid_voltage)

    def equilibrium(self):
        delta = -np.angle(self.equilibrium_grid_voltage)
        current = self.i0 * np.exp(1j * delta)
        x = [delta, current.real, current.imag]
        if self.full_filter:
            x.extend([self.id0, self.iq0])
        return np.array(x, dtype=float)


def electrical_output(x, reference, p):
    delta = x[0]
    I = x[1] + 1j * x[2]
    current = I * np.exp(-1j * delta)
    if p.full_filter:
        zd, zq = x[3], x[4]
    else:
        zd, zq = p.id0, p.iq0
    iref = ((reference / (p.kappa * p.V)) if p.xi else zd) + 1j * zq
    voltage = p.V + p.Ra * (iref - current)
    power = p.kappa * np.real(voltage * np.conj(current))
    return float(power), current, voltage


def grid_frame_rhs(tau, x, reference, p):
    power, current, voltage = electrical_output(x, reference, p)
    I = x[1] + 1j * x[2]
    u = voltage * np.exp(1j * x[0])
    dI = (u - p.Vg) / p.L - 1j * p.omega * I
    out = [p.kp * (reference - power), dI.real, dI.imag]
    if p.full_filter:
        out.extend([p.wb * (current.real - x[3]), p.wb * (current.imag - x[4])])
    return np.array(out)


def analytic_state_space(p):
    """Linearize the physical grid-frame equations; does not use paper Eq.19."""
    assert not p.full_filter
    delta = p.equilibrium()[0]
    rot = np.exp(1j * delta)
    C_current = p.kappa * (p.V - p.Ra * p.i0) * rot
    C = np.array([p.kappa * p.V * p.iq0, C_current.real, C_current.imag])
    D = p.xi * p.Ra * p.id0 / p.V
    du_delta = 1j * (p.V + p.Ra * p.i0) * rot
    du_ref = p.xi * p.Ra / (p.kappa * p.V) * rot
    A = np.zeros((3, 3))
    A[0] = -p.kp * C
    A[1:, 0] = [du_delta.real / p.L, du_delta.imag / p.L]
    A[1:, 1:] = [[-p.Ra / p.L, p.omega], [-p.omega, -p.Ra / p.L]]
    B = np.array([p.kp * (1-D), du_ref.real / p.L, du_ref.imag / p.L])
    return A, B, C, D


def finite_difference_state_space(p, h=1e-6):
    x = p.equilibrium()
    n = len(x)
    eye = np.eye(n)
    A = np.column_stack([(grid_frame_rhs(0, x+h*eye[k], p.r0, p) -
                          grid_frame_rhs(0, x-h*eye[k], p.r0, p))/(2*h)
                         for k in range(n)])
    B = (grid_frame_rhs(0, x, p.r0+h, p) - grid_frame_rhs(0, x, p.r0-h, p))/(2*h)
    C = np.array([(electrical_output(x+h*eye[k], p.r0, p)[0] -
                   electrical_output(x-h*eye[k], p.r0, p)[0])/(2*h)
                  for k in range(n)])
    D = (electrical_output(x, p.r0+h, p)[0] - electrical_output(x, p.r0-h, p)[0])/(2*h)
    return A, B, C, D


def paper_polynomials(p):
    """Direct transcription of (17)-(19), independent of the state realization."""
    alpha = p.Ra / p.L
    a = p.omega * p.L * p.iq0 / p.V
    b = -p.Ra**2 / p.V * (p.iq0 / (p.omega*p.L) + abs(p.i0)**2 / p.V)
    c = p.Ra * p.id0 / p.V
    d = p.Ra**2 * p.iq0 / (p.omega*p.L*p.V)
    k2 = (a + p.xi*(1+c))*alpha
    k1 = p.xi*((c+d)*p.omega**2 + alpha**2)
    k0 = (1+a+b)*alpha*p.omega**2
    num = np.array([p.xi*c, k2, k1, k0])
    den = np.array([1, (2+a)*alpha, alpha**2+p.omega**2, k0])
    return num, den


def resolvent_response(A, B, C, D, s):
    return C @ np.linalg.solve(s*np.eye(len(A))-A, B) + D


def linear_step(p, times):
    A, B, C, D = analytic_state_space(p)
    # Augmented exponential works even if A is singular; unlike direct A^-1.
    M = np.zeros((len(A)+1, len(A)+1))
    M[:-1, :-1] = A
    M[:-1, -1] = B
    return np.array([C @ expm(M*t)[:-1, -1] + D for t in times])


def encode_complex(values):
    return [[float(z.real), float(z.imag)] for z in values]


def first_step_metrics(times, output, reference=0.4):
    mask = (times >= 0.2) & (times < 0.4)
    t = times[mask]
    y = output[mask]
    def crossing(level):
        hit = np.flatnonzero(y >= level*reference)
        if not len(hit):
            return None
        i = hit[0]
        if i == 0:
            return float(t[0])
        return float(np.interp(level*reference, [y[i-1], y[i]], [t[i-1], t[i]]))
    t10, t90 = crossing(.1), crossing(.9)
    return {
        "rise_10_90_ms": None if t10 is None or t90 is None else (t90-t10)*1e3,
        "overshoot_percent_reference": float(max(0, y.max()-reference)/reference*100),
        "final_window_mean_pu": float(y[-200:].mean()),
    }


def sequence_run(p, method="DOP853", rtol=2e-10, atol=2e-12):
    assert p.full_filter and p.id0 == p.iq0 == 0
    times = np.linspace(0, 1, 40001)
    states = np.zeros((len(times), 5))
    refs = np.zeros(len(times))
    boundaries = [0, .2, .4, .6, .8, 1.0]
    levels = [0, .4, .8, 1, 0]
    x = p.equilibrium()
    total_nfev = 0
    for a, b, r in zip(boundaries[:-1], boundaries[1:], levels):
        mask = (times >= a-1e-12) & (times <= b+1e-12)
        tau = np.clip(times[mask], a, b) * OMEGA_BASE
        sol = solve_ivp(lambda t, state: grid_frame_rhs(t, state, r, p),
                        (a*OMEGA_BASE, b*OMEGA_BASE), x, method=method,
                        t_eval=tau, rtol=rtol, atol=atol)
        if not sol.success:
            raise RuntimeError(sol.message)
        states[mask] = sol.y.T
        refs[mask] = r
        x = sol.y[:, -1]
        total_nfev += sol.nfev
    values = [electrical_output(x, r, p) for x, r in zip(states, refs)]
    power = np.array([v[0] for v in values])
    current = np.array([v[1] for v in values])
    voltage = np.array([v[2] for v in values])
    return times, refs, states, power, current, voltage, total_nfev


def main():
    rows = []
    points = PROTOCOL["figure3"]["operating_points_id_iq"]
    w = np.logspace(-3, 2, 300)
    for id0, iq0 in points:
        for xi in [0, 1]:
            p = Parameters(id0=id0, iq0=iq0, xi=xi)
            A, B, C, D = analytic_state_space(p)
            Af, Bf, Cf, Df = finite_difference_state_space(p)
            num, den = paper_polynomials(p)
            tf_error = max(abs(resolvent_response(A, B, C, D, 1j*wi) -
                               np.polyval(num, 1j*wi)/np.polyval(den, 1j*wi)) for wi in w)
            j_error = max(np.max(np.abs(A-Af)), np.max(np.abs(B-Bf)),
                          np.max(np.abs(C-Cf)), abs(D-Df))
            residual = float(np.max(np.abs(grid_frame_rhs(0, p.equilibrium(), p.r0, p))))
            Ap, *_ = analytic_state_space(replace(p, xi=1-xi))
            pole_difference = float(np.max(np.abs(A-Ap)))
            rows.append({"id0": id0, "iq0": iq0, "xi": xi, "Vg": p.Vg,
                         "delta0_rad": float(p.equilibrium()[0]),
                         "equilibrium_max_residual": residual,
                         "jacobian_max_abs_error": float(j_error),
                         "eq19_resolvent_max_abs_error": float(tf_error),
                         "xi_state_matrix_difference": pole_difference,
                         "poles": encode_complex(np.linalg.eigvals(A)),
                         "zeros": encode_complex(np.roots(np.trim_zeros(num, "f"))),
                         "passed": bool(residual < 1e-12 and j_error < 2e-8 and
                                        tf_error < 2e-8 and pole_difference < 1e-12)})

    # Exact symbolic identity for i0=0, arbitrary positive Ra/L=alpha and omega.
    s, alpha, om = sp.symbols("s alpha omega", positive=True)
    numerator = alpha*s**2 + alpha**2*s + alpha*om**2
    denominator = s**3 + 2*alpha*s**2 + (alpha**2+om**2)*s + alpha*om**2
    symbolic_residual = sp.simplify(numerator*(s+alpha)-alpha*denominator)
    symbolic_check = {"identity_residual": str(symbolic_residual),
                      "factored_numerator": str(sp.factor(numerator)),
                      "factored_denominator": str(sp.factor(denominator)),
                      "passed": symbolic_residual == 0}

    step_rows = []
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, L in zip(axes, [.1, .5, 1.]):
        p = Parameters(L=L)
        tau = np.linspace(0, 40, 1001)
        response = linear_step(p, tau)
        exact = 1-np.exp(-p.Ra/p.L*tau)
        error = float(np.max(np.abs(response-exact)))
        conventional = linear_step(replace(p, xi=0), tau)
        ax.plot(tau/OMEGA_BASE*1000, conventional, label="PSC")
        ax.plot(tau/OMEGA_BASE*1000, response, label="RFPSC")
        ax.plot(tau/OMEGA_BASE*1000, exact, "k--", alpha=.6, label="exact first order")
        ax.set(title=f"L={L} pu, i0=0, frozen filter", xlabel="Time (ms)", ylabel="Normalized response")
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
        epsilon = 1e-5
        sol = solve_ivp(lambda t, x: grid_frame_rhs(t, x, epsilon, p), (0, 40),
                        p.equilibrium(), t_eval=tau, method="DOP853", rtol=1e-11, atol=1e-13)
        nonlinear = np.array([electrical_output(x, epsilon, p)[0] for x in sol.y.T])/epsilon
        small_error = float(np.max(np.abs(nonlinear-response)))
        step_rows.append({"L": L, "first_order_max_abs_error": error,
                          "small_signal_step": epsilon,
                          "nonlinear_normalized_max_abs_error": small_error,
                          "theoretical_rise_10_90_ms": float(np.log(9)/(p.Ra/p.L*OMEGA_BASE)*1000),
                          "passed": bool(sol.success and error < 1e-9 and small_error < .001)})
    fig.tight_layout()
    fig.savefig(HERE/"independent_zero_current_steps.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(8, 7))
    polezero_rows = []
    for ax, (id0, iq0) in zip(axes.flat, points):
        p = Parameters(id0=id0, iq0=iq0)
        A, *_ = analytic_state_space(p)
        num, _ = paper_polynomials(p)
        poles = np.linalg.eigvals(A)
        zeros = np.roots(np.trim_zeros(num, "f"))
        pp = poles[poles.imag > 0]
        zz = zeros[zeros.imag > 0]
        ax.plot(pp.real, pp.imag, "x", ms=8, label="poles: physical-state Jacobian")
        ax.plot(zz.real, zz.imag, "o", mfc="none", ms=9, label="zeros: paper Eq.19")
        ax.set(xlim=(-.3, -.1), ylim=(.9, 1.1), title=f"id0={id0}, iq0={iq0}",
               xlabel="Real part (pu)", ylabel="Imaginary part (pu)")
        ax.grid(alpha=.3)
        ax.legend(fontsize=7)
        polezero_rows.append({"id0": id0, "iq0": iq0, "pole_positive_imag": encode_complex(pp),
                              "zero_positive_imag": encode_complex(zz),
                              "distance": float(abs(pp[0]-zz[0]))})
    fig.suptitle("Independent reconstruction of Figure 3 parameters (H perturbation = 0)")
    fig.tight_layout()
    fig.savefig(HERE/"independent_figure3_poles_zeros.png", dpi=180)
    plt.close(fig)

    sequence_rows = []
    for Ra in [.2, 4.4/12.6]:
        fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True)
        for column, L in enumerate([1., .1]):
            for xi in [0, 1]:
                p = Parameters(L=L, Ra=Ra, xi=xi, full_filter=True)
                run = sequence_run(p)
                t, r, x, power, current, voltage, nfev = run
                cross = sequence_run(p, method="Radau")
                cross_error = float(np.max(np.abs(power-cross[3])))
                label = "RFPSC" if xi else "PSC"
                tag = f"Ra_{Ra:.6f}_L_{L:.1f}_{label}"
                np.savez_compressed(HERE/f"sequence_{tag}.npz", time_s=t, reference=r,
                                    states_grid_frame=x, power_pu=power,
                                    current_controller_frame=current, voltage_controller_frame=voltage)
                metrics = first_step_metrics(t, power)
                metrics.update({"L": L, "Ra": Ra, "controller": label,
                                "power_solver_max_abs_difference": cross_error,
                                "voltage_magnitude_min": float(abs(voltage).min()),
                                "voltage_magnitude_max": float(abs(voltage).max()),
                                "function_evaluations_DOP853": nfev,
                                "function_evaluations_Radau": cross[-1],
                                "passed": cross_error < 2e-7})
                sequence_rows.append(metrics)
                axes[0, column].plot(t, power, label=label)
                axes[1, column].plot(t, current.imag, label=label)
                axes[2, column].plot(t, abs(voltage), label=label)
            axes[0, column].plot(t, r, "k--", label="plot-inferred reference")
            axes[0, column].set_title(f"L={L} pu; ideal source Vg=1 pu")
            for row, ylabel in enumerate(["P (pu)", "iq (pu)", "|v| (pu)"]):
                axes[row, column].set_ylabel(ylabel)
                axes[row, column].grid(alpha=.25)
                axes[row, column].legend(fontsize=8)
            axes[2, column].set_xlabel("Time (s)")
        fig.suptitle(f"Continuous average model; full filters; Ra={Ra:.6f} pu. NOT experimental replication")
        fig.tight_layout()
        fig.savefig(HERE/f"independent_sequence_Ra_{Ra:.6f}.png", dpi=180)
        plt.close(fig)

    filter_rows = []
    for L in [.1, .5, 1.]:
        for xi in [0, 1]:
            p0 = Parameters(L=L, xi=xi)
            A0, B0, C0, D0 = analytic_state_space(p0)
            Af, Bf, Cf, Df = finite_difference_state_space(replace(p0, full_filter=True))
            errors = [abs(resolvent_response(A0, B0, C0, D0, 1j*wi) -
                          resolvent_response(Af, Bf, Cf, Df, 1j*wi)) for wi in w]
            filter_rows.append({"L": L, "controller": "RFPSC" if xi else "PSC",
                                "frozen_poles": encode_complex(np.linalg.eigvals(A0)),
                                "full_filter_poles_including_unobservable_zd_for_RFPSC": encode_complex(np.linalg.eigvals(Af)),
                                "max_transfer_difference_on_frequency_grid": float(max(errors)),
                                "status": "approximation characterization, not equality test"})

    results = {"independence": "Only accepted manuscript and local protocol read; no primary implementation or author code imported",
               "environment": {"python":sys.version, "platform": platform.platform(),
                               "numpy": np.__version__, "scipy": scipy.__version__, "sympy": sp.__version__,
                               "matplotlib": matplotlib.__version__},
               "source_sha256": hashlib.sha256((HERE/"rfpsc_2020_author_manuscript.pdf").read_bytes()).hexdigest(),
               "protocol_sha256": hashlib.sha256((HERE/"protocol.json").read_bytes()).hexdigest(),
               "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "state_space_vs_paper_checks": rows,
               "symbolic_zero_current_factorization": symbolic_check,
               "zero_current_steps": step_rows,
               "figure3_polezero": polezero_rows,
               "full_filter_sequences": sequence_rows,
               "finite_filter_approximation": filter_rows,
               "table_I_arithmetic": {"Ra_ohm":4.4,"Zbase_ohm":12.6,"ratio":4.4/12.6,
                                      "stated_Ra_pu":.2,"ohms_implied_by_stated_pu":.2*12.6,
                                      "finding":"internal normalization inconsistency in inspected manuscript; cause unknown"},
               "overall_checks_passed": bool(all(x["passed"] for x in rows+step_rows+sequence_rows) and symbolic_check["passed"]),
               "scope_limit": "No hardware/data replication, no validation of switching, control delay, finite-resistance grids, or exact original experimental traces"}
    (HERE/"verification_results.json").write_text(json.dumps(results, indent=2)+"\n")
    print(json.dumps({"overall_checks_passed": results["overall_checks_passed"],
                      "linear_cases":len(rows), "step_cases":len(step_rows),
                      "sequence_cases":len(sequence_rows),
                      "max_tf_error":max(r["eq19_resolvent_max_abs_error"] for r in rows),
                      "max_solver_difference":max(r["power_solver_max_abs_difference"] for r in sequence_rows)}, indent=2))
    return 0 if results["overall_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
