"""Independent stamp of all eight switches and the ideal transformer in Fig. 1."""

from itertools import product

import numpy as np


def stamp_resistor(matrix, rhs, a, b, conductance, known_voltage=None, history=0.0):
    matrix[a, a] += conductance
    rhs[a] -= history
    if b is None:
        rhs[a] += conductance * known_voltage
    else:
        matrix[b, b] += conductance
        matrix[a, b] -= conductance
        matrix[b, a] -= conductance
        rhs[b] += history


def step(case, step_ps, ron, roff, gate, previous, tie, rectifier_ron=None):
    dt = step_ps * 1e-12
    n, vin = case['ratio_primary_secondary'], case['input_v']
    lr, lm, cr, co, r = [case[key] for key in ('lr_h', 'lm_h', 'cr_f', 'co_f', 'load_ohm')]
    pa, pb, mid, dot, vx, vy, vo, port_current = range(8)
    matrix, rhs = np.zeros((8, 8)), np.zeros(8)
    for i, (node, rail) in enumerate(((pa, vin), (pa, 0), (pb, vin), (pb, 0))):
        resistance = ron if (gate >> i) & 1 else roff
        stamp_resistor(matrix, rhs, node, None, 1 / resistance, rail)
    stamp_resistor(matrix, rhs, pa, mid, dt / lr, history=previous[0])
    stamp_resistor(matrix, rhs, dot, pb, dt / lm, history=previous[1])
    stamp_resistor(matrix, rhs, mid, dot, cr / dt, history=-cr / dt * previous[2])
    if r != 0:
        stamp_resistor(matrix, rhs, vo, None, co / dt + 1 / r, 0.0, history=-co / dt * previous[3])
    matrix[port_current, [dot, pb, vx, vy]] = [1, -1, -n, n]
    matrix[[dot, pb, vx, vy], port_current] = [1, -1, -n, n]

    best = None
    branches = ((vx, vo), (vx, None), (vy, vo), (vy, None))
    # Use secondary common mode and differential voltage as coordinates. Replace
    # the two AC-node KCL rows with their analytic sum and difference. The sum has
    # no transformer or inductor current and stays well scaled for four off diodes.
    coordinates = np.eye(8)
    coordinates[vx, vy] = 0.5
    coordinates[vy, vx] = 1.0
    coordinates[vy, vy] = -0.5
    for flags in product((0, 1), repeat=4):
        y = matrix.copy()
        b = rhs.copy()
        for (anode, cathode), on in zip(branches, flags, strict=True):
            stamp_resistor(y, b, anode, cathode, 1 / ((rectifier_ron or ron) if on else roff), 0.0)
        if r == 0:
            y[vo, :] = 0.0
            y[vo, vo] = 1.0
            b[vo] = 0.0
        row_x, row_y = y[vx].copy(), y[vy].copy()
        y[vx], y[vy] = row_x + row_y, row_x - row_y
        b[vx], b[vy] = b[vx] + b[vy], b[vx] - b[vy]
        y = y @ coordinates
        scale = np.max(np.abs(y), axis=1)
        volts = coordinates @ np.linalg.solve(y / scale[:, None], b / scale)
        vd = np.array([volts[vx] - volts[vo], -volts[vx], volts[vy] - volts[vo], -volts[vy]])
        margin = np.min(vd * (2 * np.array(flags) - 1))
        if best is None or margin > best[0]:
            best = (margin, volts, sum(on << i for i, on in enumerate(flags)), vd)
    margin, volts, mask, vd = best
    if margin < -tie:
        raise RuntimeError(f'Full bridge has no admissible state, margin {margin} V')
    state = [previous[0] + dt / lr * (volts[pa] - volts[mid]),
             previous[1] + dt / lm * (volts[dot] - volts[pb]),
             volts[mid] - volts[dot], volts[vo], volts[vx], volts[vy]]
    return np.array(state), mask, float(np.min(np.abs(vd)))


def compare(case, wave, step_ps, ron, roff, tie, seed=2998746):
    endpoints = np.flatnonzero(np.diff(wave[:, 0]) == step_ps) + 1
    adjacent = wave[endpoints]
    changed = adjacent[np.any(adjacent[:, 7:] != wave[endpoints - 1, 7:], axis=1), 0]
    selected = set(np.searchsorted(wave[:, 0], changed[:24]))
    selected.update(np.random.default_rng(seed).choice(endpoints, size=24, replace=False))
    selected.update(endpoints[:8])
    differences = []
    mismatches = 0
    for i in sorted(selected):
        result, mask, boundary = step(case, step_ps, ron, roff, int(wave[i, 8]), wave[i - 1, 1:5], tie)
        differences.append(np.abs(result - wave[i, 1:7]))
        mismatches += mask != int(wave[i, 7]) and boundary > tie
    maximum = np.max(differences, axis=0)
    return {'tested_steps': len(selected), 'max_current_error_a': float(max(maximum[:2])),
            'max_voltage_error_v': float(max(maximum[2:])), 'mode_mismatches_outside_tie': mismatches}
