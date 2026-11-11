"""Compile strict diode regions and linear state maps for the trapezoidal RSM."""
from dataclasses import dataclass
from fractions import Fraction as Q
from itertools import product

import numpy as np


def canonical(row):
    """Normalize by a positive scalar, preserving the half-space."""
    scale = max(map(abs, row), default=0)
    return tuple(x / scale for x in row) if scale else tuple(row)


def fme_witness(rows, n):
    """Exact Fourier–Motzkin elimination of homogeneous A x > 0.

    Back substitution returns a rational point in the open region, or None.
    One-sided bounds drop out when their variable is eliminated.
    """
    rows = list(dict.fromkeys(canonical(row) for row in rows))
    levels = []
    for _ in range(n):
        if any(not any(row) for row in rows):
            return None
        lower = [r for r in rows if r[-1] > 0]
        upper = [r for r in rows if r[-1] < 0]
        levels.append((lower, upper))
        projected = [r[:-1] for r in rows if r[-1] == 0]
        projected.extend(tuple(p[-1] * a - m[-1] * b for a, b in zip(m[:-1], p[:-1]))
                         for p, m in product(lower, upper))
        rows = list(dict.fromkeys(canonical(row) for row in projected))
    if rows:  # a constant strict inequality 0 > 0
        return None
    x = []
    for lower, upper in reversed(levels):
        lo = max((-sum(a * b for a, b in zip(r[:-1], x)) / r[-1] for r in lower), default=None)
        hi = min((-sum(a * b for a, b in zip(r[:-1], x)) / r[-1] for r in upper), default=None)
        if lo is None and hi is None:
            x.append(Q(0))
        elif lo is None:
            x.append(hi - 1)
        elif hi is None:
            x.append(lo + 1)
        else:
            assert lo < hi
            x.append((lo + hi) / 2)
    return tuple(x)


def irredundant(rows, n):
    """Removing row r is safe if other strict inequalities exclude -r > 0."""
    rows = list(dict.fromkeys(map(canonical, rows)))
    i = 0
    while i < len(rows):
        other = rows[:i] + rows[i+1:]
        if fme_witness(other + [tuple(-x for x in rows[i])], n) is None:
            rows = other
        else:
            i += 1
    return rows


def solve_exact(a, b):
    """Gauss–Jordan with exact rational entries, multiple right sides."""
    n = len(a)
    aug = [list(map(Q, x + y)) for x, y in zip(a, b)]
    for j in range(n):
        k = next(k for k in range(j, n) if aug[k][j])
        aug[k], aug[j] = aug[j], aug[k]
        pivot = aug[j][j]
        aug[j] = [x / pivot for x in aug[j]]
        for k, row in enumerate(aug):
            if k != j and row[j]:
                aug[k] = [aij - row[j] * pij for aij, pij in zip(row, aug[j])]
    return [tuple(row[n:]) for row in aug]


def bits(mask, n):
    """Bit 0 is the first named diode (D1 for the resonant boost)."""
    return [(mask >> j) & 1 for j in range(n)]


@dataclass
class Mode:
    mask: int
    voltage: np.ndarray
    physical: np.ndarray
    update: np.ndarray
    diode_voltage: np.ndarray
    exact_diode_voltage: tuple
    witness: tuple
    reduced_rows: list


@dataclass
class GateMap:
    modes: list
    w: np.ndarray
    row_indices: np.ndarray
    row_polarities: np.ndarray

    def select(self, z, relative_tie):
        # Four signs per mode at most; unused columns point to index -1,
        # whose value is +inf after the minimum is formed using a mask.
        signs = self.w @ z
        margins = np.where(self.row_polarities, signs[self.row_indices] * self.row_polarities, np.inf)
        scores = margins.min(axis=1)
        k = int(scores.argmax())
        tol = relative_tie * max(1., np.max(np.abs(z)))
        if scores[k] < -tol or np.count_nonzero(scores > tol) > 1:
            raise ArithmeticError(f'No unique diode region at z={z}; margins={scores}')
        return self.modes[k], bool(scores[k] <= tol)


class DMM:
    """A gate-indexed bank of W matrices, sign rules and F state-update maps."""
    def __init__(self, protocol, step_ns):
        self.p = protocol
        self.step_ns = step_ns
        n = protocol['network']
        self.nodes = n['unknown_nodes'] + [n['source']['p']]
        self.dynamics = n['dynamic']
        self.diodes = n['diodes']
        self.switches = n['controlled']
        self.fixed = n['resistors']
        self.ports = len(self.dynamics) + 1
        dt = Q(step_ns, 10**9)
        self.gdyn = [dt / (2 * Q(b['value'])) if b['kind'] == 'L' else 2 * Q(b['value']) / dt
                     for b in self.dynamics]
        self.eye = [[Q(j == k) for k in range(self.ports)] for j in range(self.ports)]
        self.gates = {mask: self.compile_gate(mask) for mask in range(2**len(self.switches))}

    def edge(self, branch):
        return tuple(int(x == branch['p']) - int(x == branch['n']) for x in self.nodes)

    def compile_mode(self, gate, mask):
        n = len(self.nodes)
        # The known DC source node is the last column of Y. Each h is a
        # Norton injection on its own port, oriented from p to n.
        fixed = [(b, 1 / Q(b['value'])) for b in self.fixed]
        r = self.p['resistance_model_ohm']
        switches = [(b, 1 / Q(r['on'] if bit else r['off']))
                    for b, bit in zip(self.switches + self.diodes,
                                      bits(gate, len(self.switches)) + bits(mask, len(self.diodes)))]
        edges = [(self.edge(b), g) for b, g in list(zip(self.dynamics, self.gdyn)) + fixed + switches]
        y = [[sum(g * a[j] * a[k] for a, g in edges) for k in range(n)] for j in range(n - 1)]
        e = [self.edge(b) for b in self.dynamics]
        rhs = [[-a[j] for a in e] + [-y[j][-1]] for j in range(n - 1)]
        v = solve_exact([row[:-1] for row in y], rhs) + [tuple(self.eye[-1])]

        def voltage(branch):
            return tuple(sum(a * v[j][k] for j, a in enumerate(self.edge(branch))) for k in range(self.ports))

        diode_q = tuple(voltage(b) for b in self.diodes)
        inequality = [tuple((2 * bit - 1) * x for x in row)
                      for bit, row in zip(bits(mask, len(diode_q)), diode_q)]
        w = fme_witness(inequality, self.ports)
        if w is None:
            return None
        assert all(sum(a*b for a,b in zip(row,w)) > 0 for row in inequality)
        update, states = [], []
        for j, b in enumerate(self.dynamics):
            vb = voltage(b)
            h = self.eye[j]
            g = self.gdyn[j]
            if b['kind'] == 'L':
                states.append(tuple(hk + g*vk for hk,vk in zip(h,vb)))
                update.append(tuple(hk + 2*g*vk for hk,vk in zip(h,vb)))
            else:
                states.append(vb)
                update.append(tuple(-hk - 2*g*vk for hk,vk in zip(h,vb)))
        return Mode(mask, np.array(v, dtype=float), np.array(states, dtype=float),
                    np.array(update, dtype=float), np.array(diode_q, dtype=float), diode_q, w,
                    irredundant(inequality, self.ports))

    def compile_gate(self, gate):
        modes = [m for mask in range(2**len(self.diodes)) if (m := self.compile_mode(gate, mask)) is not None]
        rows = {}
        rules = []
        for mode in modes:
            rule = []
            for row in mode.reduced_rows:
                polarity = 1 if next(x for x in row if x) > 0 else -1
                key = tuple(polarity * x for x in row)
                rule.append((rows.setdefault(key, len(rows)), polarity))
            rules.append(rule)
        j = np.zeros((len(modes), max(map(len, rules))), dtype=int)
        p = np.zeros_like(j)
        for k, rule in enumerate(rules):
            for m, (idx, polarity) in enumerate(rule):
                j[k,m] = idx
                p[k,m] = polarity
        return GateMap(modes, np.array(list(rows), dtype=float), j, p)

    def solve(self, gate, z):
        mode, boundary = self.gates[gate].select(z, self.p['criteria']['lookup_relative_tie'])
        return mode.mask, mode.physical @ z, mode.update @ z, mode.voltage @ z, mode.diode_voltage @ z, boundary

    def tables(self):
        return {str(g): {'mode_count': len(m.modes), 'diode_masks': [r.mask for r in m.modes],
                         'W_shape': list(m.w.shape), 'F_shape': list(m.modes[0].update.shape),
                         'W': m.w.tolist(), 'rule_indices': m.row_indices.tolist(),
                         'rule_polarities': m.row_polarities.tolist(),
                         'F': {str(r.mask): r.update.tolist() for r in m.modes}}
                for g, m in self.gates.items()}
