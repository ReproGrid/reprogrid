"""Reproduce nominal synchronverter equilibria and the Example VI-A stability contours."""

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import root

CASE = Path(__file__).resolve().parents[1]
PROTOCOL = CASE / "configs/protocol.json"


@dataclass(frozen=True)
class Plant:
    v: float
    w: float
    r: float
    l: float
    m: float
    j: float
    dp: float
    k: float

    @classmethod
    def from_example(cls, example, frequency):
        scale = example["virtual_impedance_scale"]
        return cls(example["phase_rms_v"] * np.sqrt(3), 2 * np.pi * frequency,
                   example["physical_r_ohm"] * scale, example["physical_l_h"] * scale,
                   example["m_h"], example["j"], example["dp"],
                   example["k_a"] * example["m_h"] * np.sqrt(2 / 3))

    def configured_torque(self, p, q):
        return (p + self.r * (p*p + q*q) / self.v**2) / self.w

    def powers(self, state):
        i_d, i_q, _, angle, _ = state
        s, c = np.sin(angle), np.cos(angle)
        return -self.v * (i_d*s + i_q*c), self.v * (i_q*s - i_d*c)

    def state_for_power(self, p, q):
        # Use a frame in which the grid phasor is real, then recover the rotor frame.
        internal = self.v + (self.r + 1j*self.w*self.l) * (p - 1j*q) / self.v
        angle = np.angle(internal)
        s, c = np.sin(angle), np.cos(angle)
        return np.array([-(p*s + q*c)/self.v, -(p*c - q*s)/self.v,
                         np.full_like(angle, self.w), angle, abs(internal)/(self.m*self.w)])

    def rhs(self, state, torque, q_set):
        i_d, i_q, w, angle, field = state
        return np.array([
            -self.r/self.l*i_d + w*i_q + self.v/self.l*np.sin(angle),
            -w*i_d - self.r/self.l*i_q - self.m*field*w/self.l + self.v/self.l*np.cos(angle),
            (torque + self.m*field*i_q - self.dp*(w-self.w))/self.j,
            w-self.w,
            (q_set-self.powers(state)[1])/self.k,
        ])

    def jacobian(self, state, k):
        i_d, i_q, w, angle, field = state
        s, c = np.sin(angle), np.cos(angle)
        jac = np.zeros((*np.shape(angle), 5, 5))
        jac[..., 0, 0] = jac[..., 1, 1] = -self.r/self.l
        jac[..., 0, 1], jac[..., 1, 0] = w, -w
        jac[..., 0, 2], jac[..., 1, 2] = i_q, -i_d-self.m*field/self.l
        jac[..., 0, 3], jac[..., 1, 3] = self.v*c/self.l, -self.v*s/self.l
        jac[..., 1, 4] = -self.m*w/self.l
        jac[..., 2, 1], jac[..., 2, 2], jac[..., 2, 4] = self.m*field/self.j, -self.dp/self.j, self.m*i_q/self.j
        jac[..., 3, 2] = 1
        jac[..., 4, 0], jac[..., 4, 1] = self.v*c/k, -self.v*s/k
        jac[..., 4, 3] = -self.v*(i_q*c+i_d*s)/k
        return jac


def printed_units(x):
    return np.array([x[0], x[1], x[2], np.degrees(x[3]), x[4]])


def check_root_from_dynamics(plant, reference, state, torque, q_set):
    seed = np.array(reference)
    seed[3] = np.radians(seed[3])
    scale = np.maximum(abs(seed), 1.)
    result = root(lambda point: plant.rhs(point * scale, torque, q_set), seed/scale, tol=1e-11)
    solved = result.x * scale
    if np.max(abs(plant.rhs(solved, torque, q_set))) > 1e-8:
        raise RuntimeError("Nonlinear equilibrium solve did not converge")
    return float(np.max(abs(solved-state)/scale))


def nominal(cfg):
    records, checks = {}, {}
    limits = cfg["checks"]
    for index, (name, example) in enumerate(cfg["examples"].items()):
        plant = Plant.from_example(example, cfg["frequency_hz"])
        q_set, torque = example["q_set_var"], example["printed_torque_nm"]
        coefficient = plant.r/plant.v**2
        p_roots = sorted(np.roots([coefficient, 1., coefficient*q_set**2-plant.w*torque]), reverse=True)
        branches = []
        for sign, p, ref, printed_p in zip(("right", "left"), p_roots,
                                           example["printed_states_i_d_i_q_omega_delta_deg_i_f"],
                                           example["printed_p_w"], strict=True):
            state = plant.state_for_power(p, q_set)
            mirror = state * [-1, -1, 1, 1, -1]
            mirror[3] += np.pi
            jac = plant.jacobian(state, plant.k)
            steps = np.eye(5) * np.maximum(abs(state), 1.) * 1e-5
            numerical = np.column_stack([
                (plant.rhs(state + step, torque, q_set) - plant.rhs(state - step, torque, q_set))
                / (2 * step[i])
                for i, step in enumerate(steps)
            ])
            eig = np.linalg.eigvals(jac)
            maximum = float(max(eig.real))
            mirror_max = float(max(np.linalg.eigvals(plant.jacobian(mirror, plant.k)).real))
            residual = float(max(np.max(abs(plant.rhs(state, torque, q_set))),
                                 np.max(abs(plant.rhs(mirror, torque, q_set)))))
            root_error = check_root_from_dynamics(plant, ref, state, torque, q_set)
            jac_error = float(np.max(abs(jac-numerical)/np.maximum(1., abs(jac))))
            printed_error = float(np.max(abs(printed_units(state)-ref)))
            label = f"{name}_{sign}"
            checks[label] = (
                printed_error <= limits["printed_state_max_error_per_component"]
                and abs(p-printed_p) <= limits["printed_power_max_error_w"][index]
                and residual <= limits["equilibrium_rhs_max_error"]
                and root_error <= limits["independent_root_scaled_max_error"]
                and jac_error <= limits["numeric_jacobian_relative_max_error"]
                and (maximum < 0) == (sign == "right")
            )
            branches.append({
                "branch": sign, "state_in_paper_units": printed_units(state).tolist(),
                "mirrored_state_in_paper_units": printed_units(mirror).tolist(),
                "p_w": float(p), "q_var": q_set,
                "max_real_fourth_order_per_s": float(max(np.linalg.eigvals(jac[:4, :4]).real)),
                "max_real_fifth_order_per_s": maximum, "eigenvalues_per_s": [[z.real, z.imag] for z in eig],
                "mirrored_max_real_fifth_order_per_s": mirror_max,
                "equilibrium_max_rhs_residual": residual, "numeric_jacobian_scaled_error": jac_error,
                "independent_root_scaled_error": root_error, "max_component_difference_from_printed": printed_error,
            })
        exact_torque = plant.configured_torque(example["setpoint_w"], q_set)
        records[name] = {"printed_torque_nm": torque, "eq44_torque_nm": exact_torque,
                         "computed_ktilde_a_h": plant.k, "branches": branches}
    return records, checks


def plot_contours(cfg, output):
    spec = cfg["fig11"]
    plant = Plant.from_example(cfg["examples"][spec["example"]], cfg["frequency_hz"])
    p_kw = np.linspace(*spec["p_kw"], spec["grid_samples"])
    q_kvar = np.linspace(*spec["q_kvar"], spec["grid_samples"])
    p, q = np.meshgrid(p_kw * 1e3, q_kvar * 1e3)
    states = plant.state_for_power(p, q)
    # Eq. (43)'s right branch lies on or to the right of the circle centre.
    valid = (p > -plant.v**2/(2*plant.r)) & (states[-1] > 0)
    arrays = {"p_kw": p_kw, "q_kvar": q_kvar, "right_branch": valid}
    rows = []
    stable_regions = []
    fig, ax = plt.subplots(figsize=(6.7, 5.3), layout="constrained")
    for index, (gain, color) in enumerate(zip(spec["plotted_integrator_gain_ka_h"],
                                              ("#2166ac", "#b2182b", "#ce8000", "#762a83"), strict=True)):
        max_real = np.linalg.eigvals(plant.jacobian(states, gain * 1000)).real.max(axis=-1)
        arrays[f"max_real_{index}"] = max_real
        stable = valid & (max_real < 0)
        stable_regions.append(stable)
        boundary = np.ma.masked_where(~valid, max_real)
        ax.contour(p_kw, q_kvar, boundary, levels=[0.], colors=[color], linewidths=1.8)
        ax.plot([], [], color=color, label=f"{gain:g} kA·H")
        rows.append({"ktilde_ka_h": gain, "stable_grid_points": int(np.count_nonzero(stable))})
    # Report witnesses with the largest shared margin from the two zero-eigenvalue contours.
    lower_gain, higher_gain = arrays["max_real_0"], arrays["max_real_1"]
    witnesses = {}
    for direction, mask, margin in (
        ("lost_when_increasing_gain", stable_regions[0] & ~stable_regions[1], np.minimum(-lower_gain, higher_gain)),
        ("gained_when_increasing_gain", stable_regions[1] & ~stable_regions[0], np.minimum(lower_gain, -higher_gain)),
    ):
        if not np.any(mask):
            continue
        index = np.unravel_index(np.argmax(np.where(mask, margin, -np.inf)), mask.shape)
        witnesses[direction] = {
            "count": int(np.count_nonzero(mask)), "p_kw": float(p[index]/1e3), "q_kvar": float(q[index]/1e3),
            "max_real_2_5_per_s": float(lower_gain[index]), "max_real_14_3_per_s": float(higher_gain[index]),
        }
    ax.set(xlabel="Active power P (kW)", ylabel="Reactive power Q (kvar)",
           title="Synchronverter · Example VI-A local stability contours")
    ax.grid(alpha=.2)
    ax.legend(title="Field integrator K̃", fontsize=8)
    fig.savefig(output / "fig11a_stability.png", dpi=165)
    plt.close(fig)
    np.savez_compressed(output / "fig11a_mesh.npz", **arrays)
    return {"sampled_stable_counts": rows, "non_nested_witnesses": witnesses}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=CASE / "results")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(PROTOCOL.read_text())
    records, checks = nominal(cfg)
    contour = plot_contours(cfg, args.output)
    checks["fig11a_nonnested"] = len(contour["non_nested_witnesses"]) == 2
    data = {"doi": cfg["doi"], "protocol_sha256": sha256(PROTOCOL.read_bytes()).hexdigest(),
            "implementation_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
            "examples": records, "checks": checks, "fig11a": contour, "passed": all(checks.values())}
    (args.output / "run_manifest.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"passed": data["passed"], "checks": checks, "fig11a": contour}))
    return 0 if data["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
