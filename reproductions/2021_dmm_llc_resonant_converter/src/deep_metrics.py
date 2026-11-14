"""Quantitative comparison with fixed axes and no waveform alignment."""

import numpy as np

from network_check import step


SIGNALS = ['vo', 'ir', 'im']
COL = {'vo': 4, 'ir': 1, 'im': 2}


def load_curves(directory):
    curves = {}
    for file in sorted(directory.glob('fig*.csv')):
        a = np.loadtxt(file, skiprows=1, delimiter=',')
        _, back = np.unique(a[::-1, 0], return_index=True)
        curves[file.stem] = a[::-1][back]
    return curves


def compare_window(wave, paper, panel, step_ps):
    c = COL[panel.rsplit('_', 1)[1]]
    t_ps = wave[:, 0]
    o = wave[(t_ps * 1e-12 >= paper[0, 0]) & (t_ps * 1e-12 <= paper[-1, 0])]
    # The source panel spans 0.8 ms. A 1 µs grid represents its slower envelope.
    if panel == 'fig11e_vo':
        t = np.arange(round(paper[0, 0] * 1e6) + 1, round(paper[-1, 0] * 1e6)) * 1e-6
        signal = np.interp(t, t_ps * 1e-12, wave[:, c])
    else:
        if panel == 'fig11h_im' or panel == 'fig11h_ir':
            o = o[np.abs(o[:, 0] * 1e-12 - 0.55) <= 3e-6 + 1e-13]
        t = o[:, 0] * 1e-12
        if len(o) < 12 or not np.all(np.diff(o[:, 0]) == step_ps):
            raise ValueError(f'Missing dense samples in {panel}')
        signal = o[:, c]
    target = np.interp(t, paper[:, 0], paper[:, 1])
    delta = signal - target
    rms = np.sqrt(np.mean(delta**2))
    return {
        'reference': 'solid published PDF curve, no time or phase correction',
        'samples': len(t), 'start_s': float(t[0]), 'end_s': float(t[-1]),
        'reference_min': float(np.min(paper[:, 1])), 'reference_max': float(np.max(paper[:, 1])),
        'simulated_min': float(np.min(o[:, c])), 'simulated_max': float(np.max(o[:, c])),
        'reference_uniform_mean': float(np.mean(target)), 'simulated_uniform_mean': float(np.mean(signal)),
        'simulation_peak_time_s': float(o[np.argmax(o[:, c]), 0] * 1e-12),
        'reference_peak_time_s': float(paper[np.argmax(paper[:, 1]), 0]),
        'rms_error': float(rms), 'nrmse_pct': float(rms / np.sqrt(np.mean(target**2)) * 100),
        'ref_pp': float(np.ptp(target)), 'sim_pp': float(np.ptp(signal)),
        'centered_ripple_nrmse_pct': float(np.std(delta) / np.std(target) * 100),
    }


def full_network_checks(wave, motor, device, trial):
    dt = trial['step_ps']
    adjacent = np.diff(wave[:, 0]) == dt
    changed = np.flatnonzero(adjacent & (np.diff(wave[:, 10]) != 0)) + 1
    dense = np.flatnonzero(adjacent) + 1
    rng = np.random.default_rng(1925)
    ids = np.unique(np.r_[changed, rng.choice(dense, 30, replace=False)])
    mismatch = 0
    best = np.zeros(6)
    for i in ids:
        local = motor.copy()
        if wave[i, 10]:
            r = trial['short_ohm']
            local['load_ohm'] = 1 / (1 / r + 1 / motor['load_ohm']) if r else 0
        predicted, mask, min_edge = step(local, dt, device['on'], device['off'], int(wave[i, 8]), wave[i-1, 1:5],
                                         1e-9, rectifier_ron=trial.get('rectifier_ron_ohm'))
        best = np.maximum(best, np.abs(wave[i, 1:7] - predicted))
        mismatch += mask != int(wave[i, 7]) and min_edge > 1e-9
    return {'checked_steps': len(ids), 'checked_fault_edges': len(changed),
            'state_max_abs_delta': best.tolist(), 'diode_states_changed_outside_tie': mismatch}


def compare_schedule(wave, paper, bounds):
    t = wave[:, 0] * 1e-12
    y = wave[:, 4]
    # Uniform grid at 0.1 ms. PDF strokes do not have uniform sampling.
    grid = np.linspace(0.005, 0.6, 5951)
    sim = np.interp(grid, t, y)
    ref = np.interp(grid, paper[:, 0], paper[:, 1])
    errors = {}
    for lo, hi in zip(bounds[:-1], bounds[1:], strict=True):
        indices = (grid >= lo) & (grid < hi)
        residual = sim[indices] - ref[indices]
        errors[f'{lo:g}–{hi:g}'] = {'voltage_rmse_v': float(np.sqrt(np.mean(residual**2))),
                                     'squared_error_fraction_of_total': float(np.dot(residual, residual) / np.dot(sim-ref, sim-ref))}
    return {'start_s': float(grid[0]), 'end_s': float(grid[-1]), 'time_step_s': float(grid[1] - grid[0]),
            'vo_rms_error_v': float(np.sqrt(np.mean((sim - ref)**2))),
            'vo_nrmse_pct': float(np.linalg.norm(sim - ref) / np.linalg.norm(ref) * 100), 'per_sequence': errors}


def paper_and_cases(waves, source, protocol):
    result = {}
    for trial in protocol['cases']:
        name = trial['id']
        wave = waves[name]
        panels = {panel: compare_window(wave, series, panel, trial['step_ps'])
                  for panel, series in source.items() if panel.startswith('fig11')}
        bounds = [ps * 1e-12 for ps in protocol['timing_ps']['metric_boundaries']]
        result[name] = {'fig11': panels, 'fig10a': compare_schedule(wave, source['fig10a_vo'], bounds)}
    return result
