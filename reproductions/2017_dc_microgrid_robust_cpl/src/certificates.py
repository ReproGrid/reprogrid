"""Normalized versions of Eqs. (7), (9), and (10), with a posteriori checks."""

from itertools import product

import cvxpy as cp
import numpy as np
from scipy.linalg import solve_continuous_lyapunov
from scipy.sparse import csr_matrix


def symmetric(matrix):
    return (matrix + matrix.T) / 2


def eigenmin(matrix):
    return float(np.linalg.eigvalsh(symmetric(matrix))[0])


def spectral_abscissa(matrix):
    return float(np.max(np.linalg.eigvals(matrix).real))


def box_matrices(network, lower, upper):
    indices = np.array(list(product((False, True), repeat=network.n_loads)))
    deltas = np.where(indices, upper, lower)
    return deltas, np.array([network.jacobian(delta) for delta in deltas])


def vertex_decay(matrices, P):
    product_pa = P @ matrices
    return np.linalg.eigvalsh(-product_pa - product_pa.swapaxes(-1, -2))[:, 0]


def psd_projection(matrix):
    values, vectors = np.linalg.eigh(symmetric(matrix))
    return (vectors * np.maximum(0, values)) @ vectors.T


def normalized_lyapunov(A):
    P = symmetric(solve_continuous_lyapunov(A.T, -np.eye(len(A))))
    return P, float(np.linalg.eigvalsh(P)[-1])


def solve_problem(problem, solver):
    if solver == "CLARABEL":
        options = dict(tol_gap_abs=2e-10, tol_gap_rel=1e-9, tol_feas=2e-10, max_iter=130, max_threads=2)
    elif solver == "CVXOPT":
        options = dict(abstol=1e-9, reltol=1e-8, feastol=1e-9, refinement=2, kktsolver="robust")
    else:
        raise ValueError(f"Unsupported SDP solver {solver}")
    problem.solve(solver=solver, **options)


def invariant_matrix(n, permutations):
    if not permutations:
        return cp.Variable((n, n), symmetric=True)
    entry_orbits = {}
    ids = []
    for row in range(n):
        for col in range(n):
            key = min(tuple(sorted((row, col))), *(tuple(sorted((p[row], p[col]))) for p in permutations))
            ids.append(entry_orbits.setdefault(key, len(entry_orbits)))
    expand = csr_matrix((np.ones(n*n), (np.arange(n*n), ids)), shape=(n*n, len(entry_orbits)))
    return cp.reshape(expand @ cp.Variable(len(entry_orbits)), (n, n), order="C")


class CertificateBox:
    """One interval box of incremental CPL gains, with physical matrices in SI."""

    def __init__(self, network, lower, upper, numerics):
        self.net = network
        self.deltas, self.matrices = box_matrices(network, lower, upper)
        self.plus = self.matrices[-1]
        self.upper = self.deltas[-1]
        self.rates = np.array([spectral_abscissa(matrix) for matrix in self.matrices])
        self.scale = numerics["sdp_rate_scale_per_s"]
        self.tol = numerics["sdp_strict_margin_per_s"]
        self.p_floor = numerics["sdp_pd_floor"]

    def summary(self):
        worst = int(np.argmax(self.rates))
        return {"delta_min_per_s": self.deltas[0].tolist(), "delta_max_per_s": self.upper.tolist(),
                "states": self.net.n, "vertices": len(self.deltas), "vertices_with_rhp": int(np.sum(self.rates > self.tol)),
                "max_vertex_real_per_s": float(self.rates[worst]), "worst_vertex_index": worst,
                "worst_vertex_delta_per_s": self.deltas[worst].tolist(),
                "all_upper_real_per_s": float(self.rates[-1])}

    def result(self, method, P, strict_margin, **details):
        if P is None:
            return {"verdict": "rhp_vertex", "method": method, **details}, None
        minimum_p = eigenmin(P)
        worst = float(np.min(vertex_decay(self.matrices, P)))
        certified = minimum_p > self.p_floor and worst > self.tol and strict_margin > self.tol
        upper_bound = details.get("dual_upper_per_s")
        status = "certified" if certified else "unresolved"
        if not certified and upper_bound is not None and upper_bound < -self.tol:
            status = "strict_lmi_infeasible"
        return {"verdict": status, "method": method, "strict_margin_per_s": float(strict_margin),
                "p_min_eigenvalue": minimum_p, "min_vertex_decay_per_s": worst, **details}, P

    def eq7(self, solver="CLARABEL", permutations=(), max_solves=12):
        if np.max(self.rates) > self.tol:
            return self.result("vertex_eigenvalue", None, 0)
        A = self.matrices / self.scale
        # If A_lo A_hi v = -a v with a > 0 and w = A_hi v, the two
        # strict Lyapunov inequalities give opposite signs for v^T P w.
        pair = A[0] @ A[-1]
        eigenvalues, vectors = np.linalg.eig(pair)
        negative = np.flatnonzero((eigenvalues.real < -1e-9) & (eigenvalues.imag == 0))
        if len(negative):
            index = negative[np.argmin(eigenvalues.real[negative])]
            eigenvalue = eigenvalues[index].real
            v = vectors[:, index].real
            residual = np.linalg.norm(pair @ v - eigenvalue * v) / np.linalg.norm(v)
            if residual < 1e-9:
                return {"verdict": "strict_lmi_infeasible", "method": "negative_real_eigenvalue_of_vertex_product",
                        "vertex_indices": [0, len(A) - 1], "product_eigenvalue_per_s2": float(eigenvalue * self.scale**2),
                        "eigenvector": v.tolist(), "scaled_eigenpair_residual": float(residual)}, None
        P_seed, _ = normalized_lyapunov(self.plus)
        P_seed /= np.trace(P_seed)
        lower = vertex_decay(self.matrices, P_seed)
        if eigenmin(P_seed) > self.p_floor and float(np.min(lower)) > self.tol:
            return self.result("all_vertex_check_of_upper_point_lyapunov_solution", P_seed,
                               np.min(lower), vertices_in_sdp=0, sdp_solves=0)

        # Each auxiliary SDP uses a subset of the original constraints. If its
        # optimum is negative, the subset also supplies a dual bound for Eq. (7).
        n = self.net.n
        roots = np.arange(len(A))
        # Averaging all conjugates preserves feasibility. Under a group of
        # graph symmetries only one vertex per orbit is needed by the solver.
        for permutation in permutations:
            if not np.array_equal(self.plus, self.plus[np.ix_(permutation, permutation)]):
                raise ValueError("State permutation must preserve the upper matrix")
            load_map = permutation[self.net.diagonal] - self.net.diagonal[0]
            if not np.array_equal(self.deltas[0], self.deltas[0, load_map]):
                raise ValueError("State permutation must preserve the lower matrix")
            bits = (self.deltas[:, load_map] > self.deltas[0, load_map]).astype(int)
            roots = np.minimum(roots, bits @ (2 ** np.arange(self.net.n_loads - 1, -1, -1)))
        active = sorted({0, len(A) - 1, int(roots[np.argmin(lower)])})
        P_var = invariant_matrix(n, permutations)
        gamma = cp.Variable()
        I = np.eye(n)
        for iteration in range(1, max_solves + 1):
            lm = [-P_var @ A[j] - A[j].T @ P_var - gamma * I >> 0 for j in active]
            problem = cp.Problem(cp.Maximize(gamma), [P_var >> 0, cp.trace(P_var) == 1, *lm])
            solve_problem(problem, solver)
            P = symmetric(P_var.value)
            margins = vertex_decay(self.matrices, P)
            duals = [psd_projection(constraint.dual_value) for constraint in lm]
            traces = sum(np.trace(dual) for dual in duals)
            dual_matrix = -sum(A[j] @ dual + dual @ A[j].T for j, dual in zip(active, duals, strict=True)) / traces
            upper = float(np.linalg.eigvalsh(symmetric(dual_matrix))[-1] * self.scale)
            result, value = self.result("working_set_sdp", P, np.min(margins), solver=solver,
                                        solver_status=problem.status, sdp_solves=iteration, vertices_in_sdp=len(active),
                                        dual_upper_per_s=upper)
            if result["verdict"] != "unresolved":
                return result, value
            unseen = list(dict.fromkeys(int(roots[j]) for j in np.argsort(margins)
                                        if roots[j] not in active and margins[j] <= self.tol))
            if not unseen:
                return result, value
            active = sorted([*active, *unseen[:4]])
        return result, value

    def eq9(self):
        if self.rates[-1] > self.tol:
            return self.result("upper_vertex_eigenvalue", None, 0)
        # If A+ is Hurwitz and H solves H A+ + A+^T H = -I, then any
        # feasible P must satisfy P >= gamma H. Thus gamma <= 1/lambda_max(H)
        # when t=1, and P=H/lambda_max(H) attains the bound.
        H, hmax = normalized_lyapunov(self.plus)
        P = H / hmax
        attained = eigenmin(-P @ self.plus - self.plus.T @ P)
        bound = 2 * np.max(self.upper)
        return self.result("analytical_lyapunov_t_equals_1", P, attained - bound,
                           gamma_per_s=attained, twice_delta_max_per_s=float(bound),
                           dual_upper_per_s=float(1 / hmax - bound))

    def eq10(self, solver="CLARABEL"):
        if self.rates[-1] > self.tol:
            return self.result("upper_vertex_eigenvalue", None, 0)
        n = self.net.n
        A = self.plus / self.scale
        upper = self.upper / self.scale
        P_var = cp.Variable((n, n), symmetric=True)
        gamma = cp.Variable()
        bounds = cp.Variable(self.net.n_loads)
        lyapunov = -P_var @ A - A.T @ P_var - gamma * np.eye(n) >> 0
        cones = [cp.SOC(bound / delta + P_var[index, index], P_var[:, index]) if delta > 0 else bound >= 0
                 for index, delta, bound in zip(self.net.diagonal, upper, bounds, strict=True)]
        problem = cp.Problem(cp.Maximize(gamma - cp.sum(bounds)),
                             [P_var >> 0, cp.trace(P_var) == 1, lyapunov, *cones])
        solve_problem(problem, solver)

        # Recover the original inequalities independently using full Gk matrices.
        P = symmetric(P_var.value)
        I = np.eye(n)
        gamma_zero = eigenmin(-P @ self.plus - self.plus.T @ P)
        penalties = []
        dual = psd_projection(lyapunov.dual_value)
        dual /= np.trace(dual)
        dual_matrix = -A @ dual - dual @ A.T
        for index, delta, cone in zip(self.net.diagonal, self.upper, cones, strict=True):
            selector = np.outer(I[index], I[index])
            penalties.append(-eigenmin(delta * (P @ selector + selector @ P)))
            # A unit direction supports ||P[:,index]||, so it also gives a
            # global upper bound on the normalized SDP objective.
            if delta > 0:
                weight, vector = cone.dual_value
                support = -np.ravel(vector)
                direction = support / max(float(np.ravel(weight)[0]), np.linalg.norm(support), 1e-300)
                dual_matrix += (delta / self.scale) * (selector - symmetric(np.outer(direction, I[index])))
        penalty = np.maximum(0, penalties)
        margin = gamma_zero - np.sum(penalty)
        eta = penalty + margin / len(penalty)
        gamma_k = eta - penalty
        upper_bound = float(np.linalg.eigvalsh(symmetric(dual_matrix))[-1] * self.scale)
        return self.result("rank_two_soc_sdp", P, min(margin, gamma_zero),
                           solver=solver, solver_status=problem.status, gamma0_per_s=gamma_zero,
                           delta_gk_penalty_per_s=penalty.tolist(),
                           eta_per_s=eta.tolist(), gamma_k_per_s=gamma_k.tolist(),
                           original_gamma_sum_error_per_s=float(abs(np.sum(eta) - gamma_zero)),
                           dual_upper_per_s=upper_bound)

    def eq10_direct(self):
        """Independently solve the original full LMIs with a strict slack variable."""
        n = self.net.n
        P_var = cp.Variable((n, n), symmetric=True)
        eta = cp.Variable(self.net.n_loads, nonneg=True)
        gamma = cp.Variable()
        I = np.eye(n)
        A = self.plus / self.scale
        conditions = [P_var >> 0, cp.trace(P_var) == 1, cp.sum(eta) >= gamma, eta >= gamma,
                      -P_var @ A - A.T @ P_var - cp.sum(eta) * I >> 0]
        for index, delta, e in zip(self.net.diagonal, self.upper / self.scale, eta, strict=True):
            selector = np.outer(I[index], I[index])
            conditions.append(delta * (P_var @ selector + selector @ P_var) - (gamma - e) * I >> 0)
        problem = cp.Problem(cp.Maximize(gamma), conditions)
        solve_problem(problem, "CVXOPT")
        P = symmetric(P_var.value)
        eta_si = eta.value * self.scale
        slacks = [eigenmin(-P @ self.plus - self.plus.T @ P) - np.sum(eta_si)]
        recovered = []
        for index, delta, e in zip(self.net.diagonal, self.upper, eta_si, strict=True):
            selector = np.outer(I[index], I[index])
            recovered.append(eigenmin(delta * (P @ selector + selector @ P)) + e)
        return {"solver": "CVXOPT", "status": problem.status, "gamma_proposed_per_s": float(gamma.value * self.scale),
                "gamma_recovered_per_s": float(np.min(recovered)), "p_min_eigenvalue": eigenmin(P),
                "upper_lyapunov_slack_per_s": float(slacks[0]), "eta_min_per_s": float(np.min(eta_si))}
