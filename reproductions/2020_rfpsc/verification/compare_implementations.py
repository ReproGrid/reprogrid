"""Read-only cross-comparison of separately generated RFPSC output files."""
from pathlib import Path
import hashlib
import io
import json
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
PROTOCOL=json.loads((HERE/'cross_implementation_protocol.json').read_text())

def sha(data):
    return hashlib.sha256(data).hexdigest()

def snapshot_npz(path):
    data=path.read_bytes()
    with np.load(io.BytesIO(data),allow_pickle=False) as z:
        arrays={k:z[k].copy() for k in z.files}
    return arrays,sha(data),{k:sha(v.tobytes()) for k,v in arrays.items()}

main_manifest=json.loads((ROOT/'results/run_manifest.json').read_text())
independent_manifest=json.loads((HERE/'verification_results.json').read_text())
rows=[]
for branch,ra in [('paper_pu',.2),('table_ohm',4.4/12.6)]:
    for L in [1.,.1]:
        for ctl in ['PSC','RFPSC']:
            main_path=ROOT/'results'/f'{branch}_L{L:g}_{ctl.lower()}.npz'
            independent_path=HERE/f'sequence_Ra_{ra:.6f}_L_{L:.1f}_{ctl}.npz'
            a,ha,na=snapshot_npz(main_path)
            b,hb,nb=snapshot_npz(independent_path)
            ta,tb=a['time_s'],b['time_s']
            # Rounded index arithmetic is used only for mapping times, never signal interpolation.
            indices=np.rint((ta-tb[0])/(tb[1]-tb[0])).astype(int)
            if np.any(indices<0) or np.any(indices>=len(tb)):
                raise RuntimeError('Primary time samples outside independent output range')
            time_error=float(np.max(np.abs(ta-tb[indices])))
            if time_error>PROTOCOL['time_alignment_tolerance_s']:
                raise RuntimeError('No direct common sampling grid: refusing waveform interpolation')
            tref=np.array([0.,.2,.4,.6,.8])
            levels=np.array([0.,.4,.8,1.,0.])
            expected=levels[np.searchsorted(tref,ta,side='right')-1]
            reference_equal=bool(np.array_equal(a['reference_pu'],b['reference'][indices]))
            right_continuous=bool(np.array_equal(a['reference_pu'],expected) and
                                  np.array_equal(b['reference'][indices],expected))
            err=a['power_pu']-b['power_pu'][indices]
            k=int(np.argmax(np.abs(err)))
            event_mask=np.zeros(len(ta),dtype=bool)
            events=[]
            for event,level in zip(tref[1:],levels[1:]):
                j=int(np.argmin(abs(ta-event)))
                if abs(ta[j]-event)>PROTOCOL['time_alignment_tolerance_s']:
                    raise RuntimeError('Missing exact event sample')
                event_mask[j]=True
                events.append({'time_s':float(event),'primary_index':j,
                               'independent_index':int(indices[j]),
                               'reference_left_primary':float(a['reference_pu'][j-1]),
                               'reference_right_primary':float(a['reference_pu'][j]),
                               'reference_right_independent':float(b['reference'][indices[j]]),
                               'expected_right_reference':float(level),
                               'power_abs_error_pu':float(abs(err[j]))})
            current=b['current_controller_frame'][indices]
            states=b['states_grid_frame'][indices]
            aux={'id_pu':float(np.max(abs(a['id_pu']-current.real))),
                 'iq_pu':float(np.max(abs(a['iq_pu']-current.imag))),
                 'delta_rad':float(np.max(abs(a['delta_rad']-states[:,0]))),
                 'voltage_abs_pu':float(np.max(abs(a['voltage_abs_pu']-abs(b['voltage_controller_frame'][indices])))),
                 'filter_d_pu':float(np.max(abs(a['h_d']-states[:,3]))),
                 'filter_q_pu':float(np.max(abs(a['h_q']-states[:,4])))}
            mx=float(np.max(abs(err)))
            rows.append({'case_id':f'{branch}_L{L:g}_{ctl.lower()}', 'L':L,'Ra':ra,'controller':ctl,
                         'primary_file':str(main_path),'independent_file':str(independent_path),
                         'primary_file_sha256':ha,'independent_file_sha256':hb,
                         'primary_numeric_array_sha256':na,'independent_numeric_array_sha256':nb,
                         'comparison_samples':len(ta),'independent_samples':len(tb),
                         'index_stride':int(indices[1]-indices[0]),
                         'max_timestamp_error_s':time_error,'interpolation_used':False,
                         'reference_vectors_exactly_equal':reference_equal,
                         'event_convention_right_continuous_verified':right_continuous,
                         'max_power_abs_error_pu':mx,'rms_power_error_pu':float(np.sqrt(np.mean(err**2))),
                         'max_power_error_time_s':float(ta[k]),
                         'max_event_power_error_pu':float(np.max(abs(err[event_mask]))),
                         'max_non_event_power_error_pu':float(np.max(abs(err[~event_mask]))),
                         'events':events,'auxiliary_max_abs_errors':aux,
                         'tolerance_power_pu':PROTOCOL['power_max_abs_tolerance_pu'],
                         'passed':bool(reference_equal and right_continuous and mx<PROTOCOL['power_max_abs_tolerance_pu'])})
result={'protocol_sha256':sha((HERE/'cross_implementation_protocol.json').read_bytes()),
        'comparison_script_sha256':sha(Path(__file__).read_bytes()),
        'primary_output_manifest_sha256':sha((ROOT/'results/run_manifest.json').read_bytes()),
        'primary_generation_code_sha256':main_manifest['implementation_sha256'],
        'primary_current_code_sha256':sha((ROOT/'src/reproduce.py').read_bytes()),
        'primary_generation_protocol_sha256':main_manifest['protocol_sha256'],
        'independent_generation_code_sha256':independent_manifest['verifier_sha256'],
        'independent_generation_protocol_sha256':independent_manifest['protocol_sha256'],
        'independence':'Independent derivation, code, protocol and numerical outputs completed before primary source inspection. Primary source was read only for this cross-comparison.',
        'basis':'Primary code integrates converter-dq current including delta_dot frame term; independent code integrates fixed-grid-frame current.',
        'cases':rows,'all_cases_passed':all(r['passed'] for r in rows),
        'maximum_power_difference_pu':max(r['max_power_abs_error_pu'] for r in rows),
        'scope_limit':'Cross-implementation agreement checks mathematical/software consistency, not correspondence to original experimental measurements.'}
(HERE/'cross_implementation.json').write_text(json.dumps(result,indent=2)+'\n')
lines=['# Cross-implementation verification','',
       'Two independently written models were compared: primary converter-dq frame and verifier fixed-grid frame. The independent derivation and outputs were completed before reading the primary source.','',
       'The primary grid is 100 microseconds (10,001 points), and the verifier grid is 25 microseconds (40,001 points). Every fourth verifier sample is selected. Every timestamp is checked within 1e-12 seconds; no signal interpolation is used.','',
       'Both outputs use post-step references at 0.2, 0.4, 0.6, and 0.8 seconds while the physical states remain continuous. Full reference vectors and all four event samples are checked. Algebraic power may jump for RFPSC at nonzero current; those event samples are included, not discarded.','',
       'The power threshold of 2e-7 pu was recorded in cross_implementation_protocol.json before computing paired differences; it matches the independent solver gate.','',
       '| Case | Max power difference (pu) | Time of maximum (s) | Event max (pu) | Passed |',
       '|---|---:|---:|---:|---|']
for r in rows:
    lines.append(f"| {r['case_id']} | {r['max_power_abs_error_pu']:.6g} | {r['max_power_error_time_s']:.6f} | {r['max_event_power_error_pu']:.6g} | {r['passed']} |")
lines.extend(['',f"All eight cases passed: **{result['all_cases_passed']}**. Maximum power difference: **{result['maximum_power_difference_pu']:.6g} pu**.",'',
              'Raw file hashes and numeric-array hashes are recorded in cross_implementation.json. The generating code hash from the primary run manifest is retained separately from the current inspected source hash, because source edits may occur after the outputs were generated. Re-run this comparison after a clean replay to attach the new output/code version.','',
              'This establishes implementation agreement for the stated model and protocol. It does not resolve Table I ambiguity, supply missing experimental data, or establish hardware validation.',''])
(HERE/'cross_implementation.md').write_text('\n'.join(lines))
print(json.dumps({'all_cases_passed':result['all_cases_passed'],'maximum_power_difference_pu':result['maximum_power_difference_pu'],
                  'cases':[(r['case_id'],r['max_power_abs_error_pu']) for r in rows]},indent=2))
raise SystemExit(0 if result['all_cases_passed'] else 1)
