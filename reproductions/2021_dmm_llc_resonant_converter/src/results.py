"""Metrics on a common hardware-clock grid for the fixed-frequency experiments."""

import numpy as np


def describe(wave, raw, start_ps):
    result = {'status': raw['status'], 'simulated_steps': raw['steps'], 'end_s': raw['last_ps'] * 1e-12,
              'stop_state': {name: float(x) for name, x in zip(('ir_a', 'im_a', 'vcr_v', 'vo_v'), wave[-1, 1:5])},
              'm1': raw['m1'], 'm2': raw['m2'], 'solver_checks': raw['checks']}
    steady = wave[wave[:, 0] >= start_ps]
    if raw['status'] != 'completed' or len(steady) < 3:
        result['late_window'] = None
        return result

    dt = np.diff(steady[:, 0])
    duration = sum(dt)
    current, output = steady[:, 1], steady[:, 4]
    mask = steady[:, 7]
    blocked = mask[1:] == 0
    result['late_window'] = {
        'start_s': start_ps * 1e-12,
        'ir_abs_peak_a': float(np.max(np.abs(current))),
        'ir_rms_a': float(np.sqrt(np.trapezoid(current**2, steady[:, 0]) / duration)),
        'vo_mean_v': float(np.trapezoid(output, steady[:, 0]) / duration),
        'vo_pp_v': float(np.ptp(output)),
        'blocked_pct': float(np.sum(dt[blocked]) / duration * 100),
        'rectifier_mode_transitions': int(np.count_nonzero(np.diff(mask))),
        'direct_pair_flips': int(np.count_nonzero(pair_flip(mask))),
    }
    return result


def pair_flip(mask):
    return ((mask[:-1] == 6) & (mask[1:] == 9)) | ((mask[:-1] == 9) & (mask[1:] == 6))


def compare(left, right, start_ps, blocking_reference=False):
    if min(left[-1, 0], right[-1, 0]) <= start_ps:
        return None
    first = max(start_ps, left[0, 0], right[0, 0])
    end = min(left[-1, 0], right[-1, 0])
    matched = left[(left[:, 0] >= first) & (left[:, 0] <= end)]
    if len(matched) < 3 or end < start_ps + 0.95 * (left[-1, 0] - start_ps):
        return None
    times = matched[:, 0]
    ir = np.interp(times, right[:, 0], right[:, 1])
    voltage = np.interp(times, right[:, 0], right[:, 4])
    residual = matched[:, 1] - ir
    rms_scale = float(np.linalg.norm(matched[:, 1]))
    result = {'window_s': [float(times[0] * 1e-12), float(times[-1] * 1e-12)],
              'max_abs_ir_error_a': float(np.max(np.abs(residual))),
              'ir_nrmse_pct': float(np.linalg.norm(residual) / rms_scale * 100),
              'vo_mean_error_v': float(np.mean(voltage - matched[:, 4])),
              'vo_nrmse_pct': float(np.linalg.norm(voltage - matched[:, 4]) / np.linalg.norm(matched[:, 4]) * 100)}
    if blocking_reference:
        idx = np.clip(np.searchsorted(right[:, 0], times), 0, len(right) - 1)
        end_modes, candidate_modes = matched[:, 7], right[idx, 7]
        # Interior samples remove the actual edges of the reference blocked intervals.
        blocked = end_modes == 0
        interior = np.zeros(len(matched), dtype=bool)
        interior[1:-1] = blocked[:-2] & blocked[1:-1] & blocked[2:]
        if np.any(interior):
            result['ir_rmse_inside_block_a'] = float(np.sqrt(np.mean(residual[interior]**2)))
        direct_flips = pair_flip(candidate_modes)
        result['direct_pair_flips_inside_reference_block'] = int(np.count_nonzero(direct_flips & blocked[:-1] & blocked[1:]))
    return result


def ramps(wave, centers):
    features = {}
    t = wave[:, 0] * 1e-12
    windows = [(f'near_{center:g}', center - 3e-6, center + 3e-6) for center in centers]
    windows.append(('near_end', 0.59999, 0.6))
    for name, left, right in windows:
        selected = wave[(t >= left) & (t <= right)]
        features[name] = {'mean_vo_v': float(np.mean(selected[:, 4])), 'max_abs_ir_a': float(np.max(np.abs(selected[:, 1]))),
                          'ripple_vo_pp_v': float(np.ptp(selected[:, 4]))}
    return features


def verify(raw, criteria):
    output = {'finished': raw['status'] == 'completed'}
    for name in ('checks', 'probes'):
        value = raw[name]
        prefix = 'trajectory' if name == 'checks' else 'probes'
        output[prefix + '_current'] = value['max_current_delta_a'] <= criteria['max_dmm_reference_current_error_a']
        output[prefix + '_voltage'] = value['max_voltage_delta_v'] <= criteria['max_dmm_reference_voltage_error_v']
        output[prefix + '_kcl'] = value['max_kcl_relative'] <= criteria['max_relative_rectifier_kcl_residual']
        output[prefix + '_energy'] = value['max_be_energy_relative'] <= criteria['max_relative_discrete_energy_residual']
        output[prefix + '_diode_polarity'] = value['max_diode_sign_violation_v'] <= criteria['max_diode_violation_v']
        if criteria['require_no_mode_mismatches_outside_nodal_tie']:
            output[prefix + '_admissible_mode'] = value['max_mode_mismatch_boundary_distance_v'] <= criteria['nodal_tie_v']
    output['single_stage_history'] = (raw['checks']['max_single_stage_history_rel'] <=
                                      criteria['max_relative_single_stage_history_error'])
    output['single_stage_masks'] = raw['checks']['single_stage_mode_mismatches'] == 0
    return output
