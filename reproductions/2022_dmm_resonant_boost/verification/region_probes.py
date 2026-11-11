"""Feasibility certificates from target node voltages and tableau KCL."""
import numpy as np


def verify_regions(protocol, model, ref):
    """Construct z for prescribed diode voltages, independently of DMM's Q."""
    p = protocol['probes']
    rng = np.random.default_rng(p['seed'])
    source = float(protocol['network']['source']['value'])
    r = protocol['resistance_model_ohm']
    gs = [1/float(r['off']), 1/float(r['on'])]
    free = np.array([0, 1, 3, 4])  # Lg, Lr, Cr, Co; keep Cs history at zero
    B = ref.incidence[:-1, :ref.nd]
    assert np.linalg.matrix_rank(B[:, free]) == ref.nn - 1
    rows, count, worst, mismatch, fallback = [], 0, 0., 0, 0
    for gate in model.gates:
        probes = [(None, np.r_[row, source], None) for row in
                  rng.standard_normal((p['random_per_gate'],ref.nd))*p['history_scales_a']]
        probes.append((None, np.zeros(ref.nd+1), np.zeros(ref.nn)))
        for mask, signs in enumerate(ref.directions):
            vd = signs * p['witness_diode_voltage_v']
            # Solve the four diode drop equations geometrically. This
            # explicitly checks the published anode/cathode ordering.
            sw = -vd[0]
            out = sw - vd[2]
            v = np.array([sw, sw-vd[3], out-vd[1], out, source])
            cond = [*ref.g, *(1/float(b['value']) for b in ref.resistors),
                    *(gs[gate >> j & 1] for j in range(len(ref.controlled))),
                    *(gs[mask >> j & 1] for j in range(len(ref.diodes)))]
            i = (ref.incidence[:,:-1].T @ v) * cond
            h = np.zeros(ref.nd)
            h[free] = np.linalg.solve(B[:,free], -ref.incidence[:-1,:-1] @ i)
            probes.append((mask, np.r_[h,source], v))
        for expected, z, known_voltage in probes:
            a,_,_,v1,vd1,_ = model.solve(gate,z)
            b,_,_,v2,vd2,_,_,fb = ref.solve(gate,z)
            fallback += int(fb)
            mismatch += int(a != b and min(np.min(np.abs(vd1)),np.min(np.abs(vd2))) >
                            protocol['criteria']['mode_compare_min_abs_diode_v'])
            norm = max(1., np.max(np.abs(v2)))
            peak = np.max(np.abs(v1-v2)) / norm
            if known_voltage is not None:
                peak = max(peak, np.max(np.abs(v2-known_voltage))/norm)
            worst = max(worst, peak)
            if expected is not None:
                rows.append({'gate_mask': gate, 'diode_mask': expected, 'z_history_a_then_vg_v': z.tolist(),
                             'target_node_voltages_v': known_voltage.tolist(),
                             'dmm_mask': a, 'tableau_mask': b,
                             'tableau_diode_voltages_v': vd2.tolist(), 'relative_voltage_error': float(peak)})
            count += 1
    return {'probes_total': count, 'strict_constructed_witnesses': len(rows),
            'all_witness_masks_agree': all(r['diode_mask']==r['dmm_mask']==r['tableau_mask'] for r in rows),
            'mismatches_away_from_transition': mismatch, 'max_relative_peak_voltage_error': float(worst),
            'reference_fallbacks': fallback, 'witnesses': rows}
