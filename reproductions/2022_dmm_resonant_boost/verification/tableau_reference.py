"""Independent branch-current tableau, with implicit diode active-set iteration.

This module takes only the circuit description. It does not use precomputed
nodal responses, feasibility tests, lookup rows or state matrices from DMM.
"""
from itertools import product

import numpy as np


class TableauReference:
    def __init__(self, protocol, step_ns):
        n = protocol['network']
        self.dynamic = n['dynamic']
        self.diodes = n['diodes']
        self.controlled = n['controlled']
        self.resistors = n['resistors']
        self.branches = self.dynamic + self.resistors + self.controlled + self.diodes + [n['source']]
        self.nodes = n['unknown_nodes'] + [n['source']['p']]
        self.nd = len(self.dynamic)
        self.nn = len(self.nodes)
        self.diode_slice = slice(-1-len(self.diodes), -1)
        self.nmasks = 2**len(self.diodes)
        self.eps = protocol['criteria']['reference_tie_voltage_v']
        self.incidence = np.zeros((self.nn, len(self.branches)))
        for j, branch in enumerate(self.branches):
            for side, polarity in [('p', 1), ('n', -1)]:
                if branch[side] != '0':
                    self.incidence[self.nodes.index(branch[side]), j] += polarity
        dt = step_ns * 1e-9
        self.g = np.array([dt/(2*float(b['value'])) if b['kind']=='L' else 2*float(b['value'])/dt
                           for b in self.dynamic])
        self.inductors = np.array([b['kind']=='L' for b in self.dynamic])
        self.directions = np.array([[(mask >> j & 1)*2-1 for j in range(len(self.diodes))]
                                    for mask in range(self.nmasks)])
        self.bits = 2 ** np.arange(len(self.diodes))
        self.matrix = {}
        r = protocol['resistance_model_ohm']
        rstates = [1 / float(r['off']), 1 / float(r['on'])]
        for gate, mask in product(range(2**len(self.controlled)), range(self.nmasks)):
            cond = [*self.g, *(1/float(b['value']) for b in self.resistors),
                    *(rstates[(gate >> j) & 1] for j in range(len(self.controlled))),
                    *(rstates[(mask >> j) & 1] for j in range(len(self.diodes)))]
            mat = np.zeros((self.nn+len(self.branches),)*2)
            mat[:self.nn,self.nn:] = self.incidence
            # KCL uses branch-current unknowns. Separate branch laws use
            # scaled Norton rows; high conductance does not enter KCL.
            scale = 1 / np.maximum(cond, 1)
            mat[self.nn:-1,:self.nn] = (-np.asarray(cond)*scale)[:,None] * self.incidence[:,:-1].T
            j = self.nn + np.arange(len(cond))
            mat[j,j] = scale
            mat[-1,:self.nn] = self.incidence[:,-1]  # independent voltage source
            self.matrix[gate,mask] = (mat, scale[:self.nd])

    def trial(self, gate, mask, z):
        mat, scale = self.matrix[gate,mask]
        rhs = np.zeros(mat.shape[0])
        rhs[self.nn:self.nn+self.nd] = z[:-1] * scale
        rhs[-1] = z[-1]
        sol = np.linalg.solve(mat, rhs)
        v = sol[:self.nn]
        i = sol[self.nn:]
        return v, i, self.incidence.T @ v

    def solve(self, gate, z, previous=0):
        """Start from previous diode bits; fall back to all modes on a cycle."""
        mask, trials, solves = previous, {}, 0
        while mask not in trials:
            v,i,vb = trials[mask] = self.trial(gate,mask,z)
            solves += 1
            vd = vb[self.diode_slice]
            inconsistent = self.directions[mask] * vd < -self.eps
            if not inconsistent.any():
                break
            mask ^= int(self.bits @ inconsistent)
        else:
            for k in range(self.nmasks):
                if k not in trials:
                    trials[k] = self.trial(gate,k,z)
                    solves += 1
            scores = {k: float((self.directions[k] * x[2][self.diode_slice]).min())
                      for k, x in trials.items()}
            mask = max(scores, key=scores.get)
            if scores[mask] < -self.eps:
                raise ArithmeticError(f"No diode-consistent tableau solution for {gate=}, {z=}: {scores}")
            v,i,vb = trials[mask]
            return self.outputs(mask,v,i,vb,solves,True)
        return self.outputs(mask,v,i,vb,solves,False)

    def outputs(self, mask, v, i, vb, solves, fallback):
        vd = vb[self.diode_slice]
        cv, ci = vb[:self.nd], i[:self.nd]
        x = np.where(self.inductors, ci, cv)
        hnew = np.where(self.inductors, 1., -1.) * (ci + self.g * cv)
        return mask, x, hnew, v, vd, i, solves, fallback
