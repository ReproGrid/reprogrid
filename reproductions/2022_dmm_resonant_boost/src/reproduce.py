"""Reproduce the offline RBC DMM computation, and compare independent tableau solves."""
import argparse
from datetime import datetime, timezone
from fractions import Fraction
from hashlib import sha256
from math import lcm
from pathlib import Path
import json
import platform
import sys
import time

import numpy as np

from dmm import DMM, bits, fme_witness

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'verification'))
from tableau_reference import TableauReference
from region_probes import verify_regions

PROTOCOL = ROOT/'configs/protocol.json'


def utc():
    return datetime.now(timezone.utc).isoformat()


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def gate_arrays(p, step, stop, fixed_edges=False):
    time_ns = np.arange(stop//step+1, dtype=np.int64)*step
    u = p['pwm']
    phase = time_ns % u['period_ns']
    snap = p['primary']['step_ns'] if fixed_edges else step
    edges = {name: (u[name]+snap-1)//snap*snap
             for name in ['S1_on_ns','S1_off_ns','S2_on_ns','S2_off_ns']}
    gate = ((phase >= edges['S1_on_ns']) & (phase < edges['S1_off_ns'])).astype(np.uint8)
    gate += 2*((phase >= edges['S2_on_ns']) & (phase < edges['S2_off_ns'])).astype(np.uint8)
    gate[0] = 0
    return time_ns,gate,edges


def run_dmm(p, d, ref, step, stop, fixed_edges=False, checks=False):
    t,g,e = gate_arrays(p,step,stop,fixed_edges)
    z = np.zeros(d.ports)
    z[-1] = float(p['network']['source']['value'])
    x = np.zeros((len(t),d.ports-1))
    masks = np.zeros(len(t),dtype=np.uint8)
    boundary = np.zeros(len(t),dtype=bool)
    if checks:
        volt = np.zeros((len(t),ref.nn))
        diodes = np.zeros((len(t),len(ref.diodes)))
        history = np.zeros((len(t),ref.nd))
    start = time.perf_counter()
    for k, gate in enumerate(g[1:],1):
        m, x[k], h, v, vd, b = d.solve(int(gate),z)
        masks[k], boundary[k] = m,b
        if checks:
            volt[k],diodes[k],history[k] = v,vd,z[:-1]
        z[:-1] = h
    elapsed = time.perf_counter()-start
    extras = None
    if checks:
        # Re-evaluate actual branch laws, KCL and trapezoidal residual from
        # the node/state trace; no use of the precomputed F in these checks.
        bv = volt @ ref.incidence
        r = p['resistance_model_ohm']
        on,off = 1/float(r['on']),1/float(r['off'])
        dsel = (masks[:,None] >> np.arange(len(ref.diodes))) & 1
        gsel = (g[:,None] >> np.arange(len(ref.controlled))) & 1
        dyn_i = bv[:,:ref.nd] * ref.g + history
        irl = bv[:,ref.nd:ref.nd+1]/float(ref.resistors[0]['value'])
        igate = bv[:,ref.nd+len(ref.resistors):ref.diode_slice.start]*np.where(gsel,on,off)
        idiode = bv[:,ref.diode_slice]*np.where(dsel,on,off)
        currents = np.column_stack((dyn_i,irl,igate,idiode))
        kres = currents @ ref.incidence[:-1,:-1].T
        kcl = np.max(np.abs(kres)/np.maximum(1,np.max(np.abs(currents),axis=1,keepdims=True)))
        vl,il = bv[:,:ref.nd][:,ref.inductors],dyn_i[:,ref.inductors]
        vc,ic = bv[:,:ref.nd][:,~ref.inductors],dyn_i[:,~ref.inductors]
        trap_l = np.diff(il,axis=0)-(vl[1:]+vl[:-1])*ref.g[ref.inductors]
        trap_c = ic[1:]+ic[:-1]-np.diff(vc,axis=0)*ref.g[~ref.inductors]
        extras = {'max_relative_kcl': float(kcl),
                  'max_relative_trapezoid_residual': float(max(np.abs(trap_l).max()/max(1.,np.abs(il).max()),
                                                                np.abs(trap_c).max()/max(1.,np.abs(ic).max()))),
                  'max_diode_sign_violation_v': float(max(0.,-(diodes*(2*dsel-1)).min()))}
    return {'t_ns':t,'gate':g,'x':x,'mask':masks,'boundary':boundary,
            'edges':e,'wall_s':elapsed,'checks':extras}


def run_reference(p, ref, trace):
    z = np.zeros(ref.nd+1)
    z[-1] = float(p['network']['source']['value'])
    mask = 0
    shape = trace['x'].shape
    x = np.zeros(shape)
    masks = np.zeros(len(x),dtype=np.uint8)
    vd_min = np.zeros(len(x))
    solves = np.zeros(len(x),dtype=np.uint8)
    fallbacks = np.zeros(len(x),dtype=bool)
    start = time.perf_counter()
    for k,gate in enumerate(trace['gate'][1:],1):
        mask, x[k], z[:-1], _, vd, _, solves[k], fallbacks[k] = ref.solve(int(gate),z,mask)
        masks[k] = mask
        vd_min[k] = np.min(np.abs(vd))
    return {'x':x,'mask':masks,'vd_min':vd_min,'solves':solves,'fallbacks':fallbacks,
            'wall_s':time.perf_counter()-start}


def l2(a,b):
    return (np.linalg.norm(a-b,axis=0)/np.maximum(np.linalg.norm(b,axis=0),np.finfo(float).tiny)).tolist()


def refine(p, args, primary):
    records = {}
    u = p['refinement']
    results = {}
    for fixed, steps in [(False,u['steps_ns']), (True,u['fixed_100ns_edges_steps_ns'])]:
        name = 'fixed_edges' if fixed else 'published_edges'
        traces = {}
        for dt in steps:
            if dt == p['primary']['step_ns']:
                cut = u['stop_ns']//dt+1
                traces[dt] = {'t_ns':primary['t_ns'][:cut],'x':primary['x'][:cut],'edges':primary['edges'],'wall_s':None}
            else:
                compiled = DMM(p,dt)
                traces[dt] = run_dmm(p,compiled,None,dt,u['stop_ns'],fixed)
        records[name] = {'edges_ns':{str(dt):t['edges'] for dt,t in traces.items()},
                         'wall_seconds':{str(dt):t['wall_s'] for dt,t in traces.items()},
                         'pairwise_relative_l2':{}}
        for a,b in zip(steps,steps[1:]):
            stride = lcm(a,b)
            aa,bb = traces[a]['x'][::stride//a],traces[b]['x'][::stride//b]
            records[name]['pairwise_relative_l2'][f'{a}ns_vs_{b}ns'] = l2(aa,bb)
        for step,tr in traces.items():
            results[f'{name}_{step}ns_t_ns'] = tr['t_ns']
            results[f'{name}_{step}ns_x'] = tr['x']
    np.savez_compressed(args.out/'refinement.npz',**results)
    return records


def topology_summary(d):
    summary = {}
    for gate,t in d.gates.items():
        positive = []
        for m in t.modes:
            q = m.exact_diode_voltage
            o = tuple(-a-c for a,c in zip(q[0],q[2]))  # vo = -vd1 - vd3
            signed = [tuple((2*b-1)*x for x in row) for b,row in zip(bits(m.mask,len(q)),q)]
            if fme_witness([*signed,o],d.ports) is not None:
                positive.append(m.mask)
        summary[str(gate)] = {'modes':len(t.modes),'W_shape':list(t.w.shape),
                              'F_shape':list(t.modes[0].update.shape),
                              'positive_output_mode_count':len(positive),'positive_output_masks':positive}
    return summary


def plot(out, trace, ref, p):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(3, 2, figsize=(11.2,7.2), layout='constrained')
    t = trace['t_ns']
    all_labels = [r'$i_g$ (A)',r'$i_r$ (A)',r'$v_{Cs}$ (V)',r'$v_{Cr}$ (V)',r'$v_o$ (V)']
    indices = np.flatnonzero((t >= p['primary']['view_start_ns']) & (t <= p['primary']['view_stop_ns']))
    for i,a in enumerate(ax.flat[:5]):
        full = i in (0,4)
        jj = np.arange(0,len(t),100) if full else indices
        scale = 1e6 if full else 1e3
        a.plot(t[jj]/scale, ref['x'][jj,i], color='#e17928', lw=2.7, alpha=.65, label='Iterative tableau')
        a.plot(t[jj]/scale, trace['x'][jj,i], color='#13678f', lw=1.1, label='DMM')
        a.set(xlabel=f'Time ({"ms" if full else "μs"})',ylabel=all_labels[i])
        a.grid(alpha=.23)
    ax[0,1].legend(loc='best')
    a = ax.flat[-1]
    a.step(t[indices]/1e3,trace['mask'][indices],where='post',color='#3f5589',label='Diode mask')
    a.step(t[indices]/1e3,trace['gate'][indices],where='post',color='#c27821',label='Gate mask')
    a.set(xlabel='Time (μs)',ylabel='Binary mask (decimal)')
    a.legend(loc='best')
    a.grid(alpha=.23)
    fig.suptitle('Resonant boost converter | offline replay at 100 ns')
    fig.savefig(out/'rbc_waveforms.png', dpi=185)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'results')
    parser.add_argument('--figures',type=Path,default=ROOT/'figures')
    args = parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    args.figures.mkdir(parents=True,exist_ok=True)
    started = utc()
    clock = time.perf_counter()
    p = json.loads(PROTOCOL.read_text())
    c = p['criteria']
    step,stop = p['primary']['step_ns'],p['primary']['stop_ns']
    tic = time.perf_counter()
    d = DMM(p,step)
    compile_wall = time.perf_counter()-tic
    r = TableauReference(p,step)
    modes = topology_summary(d)
    dump(args.out/'compiled_maps.json',d.tables())
    dump(args.out/'topology_summary.json',modes)
    print('Compiled maps:', {g:(m['modes'],m['W_shape']) for g,m in modes.items()},flush=True)
    probes = verify_regions(p,d,r)
    dump(args.out/'region_probes.json',probes)
    print(f"Probes: {probes['probes_total']}, independent strict witnesses: {probes['strict_constructed_witnesses']}",flush=True)
    trace = run_dmm(p,d,r,step,stop,checks=True)
    print(f"DMM replay: {len(trace['x'])-1} steps in {trace['wall_s']:.2f}s",flush=True)
    ref = run_reference(p,r,trace)
    print(f"Tableau replay: {ref['wall_s']:.2f}s",flush=True)
    cmp_l2 = l2(trace['x'],ref['x'])
    mismatches = trace['mask'] != ref['mask']
    boundary = trace['boundary'] | (ref['vd_min'] <= c['mode_compare_min_abs_diode_v'])
    by_gate = {str(g):sorted(set(map(int,trace['mask'][1:][trace['gate'][1:]==g]))) for g in d.gates}
    gates_and_diodes = {(int(g),int(m)) for g,m in zip(trace['gate'][1:],trace['mask'][1:])}
    metrics = {
        'state_order':[b['state'] for b in p['network']['dynamic']],
        'step_ns':step,'time_steps':len(trace['x'])-1,'stop_ns':stop,'effective_edges_ns':trace['edges'],
        'trajectory_relative_l2':cmp_l2,
        'trajectory_max_abs_error':np.max(np.abs(trace['x']-ref['x']),axis=0).tolist(),
        'trajectory_mask_mismatches':int(mismatches.sum()),
        'trajectory_mask_mismatches_away_from_transition':int(np.count_nonzero(mismatches & ~boundary)),
        'trajectory_mask_mismatches_near_transition':int(np.count_nonzero(mismatches & boundary)),
        'min_diode_voltage_at_mask_mismatches_max_v':float(np.max(ref['vd_min'][mismatches],initial=0)),
        'visited_diode_masks':sorted(set(map(int,trace['mask'][1:]))),
        'visited_gate_diode_pairs':len(gates_and_diodes),'visited_diode_masks_by_gate':by_gate,
        'reference_linear_solves_total':int(ref['solves'].sum()),
        'reference_linear_solves_per_step_max':int(ref['solves'].max()),
        'reference_linear_solves_per_step_histogram':{str(k):int(np.count_nonzero(ref['solves'][1:]==k)) for k in np.unique(ref['solves'][1:])},
        'reference_fallback_steps':int(ref['fallbacks'].sum()),
        'dmm_online_linear_solves':0,
        'network_residuals':trace['checks'],
        'final_dmm_state':trace['x'][-1].tolist(),
    }
    dump(args.out/'metrics.json',metrics)
    np.savez_compressed(args.out/'rbc_100ns.npz',t_ns=trace['t_ns'],gate_mask=trace['gate'],
                        dmm=trace['x'],tableau=ref['x'],dmm_diode_mask=trace['mask'],tableau_diode_mask=ref['mask'],
                        dmm_boundary=trace['boundary'],tableau_min_abs_diode_voltage=ref['vd_min'],
                        tableau_linear_solves=ref['solves'],tableau_fallback=ref['fallbacks'])
    view = (trace['t_ns']>=p['primary']['view_start_ns']) & (trace['t_ns']<=p['primary']['view_stop_ns'])
    np.savetxt(args.out/'rbc_zoom.csv',np.column_stack([trace['t_ns'][view],trace['gate'][view],trace['mask'][view],trace['x'][view],ref['x'][view]]),
               delimiter=',',header='t_ns,gate_mask,diode_mask,'+','.join('dmm_'+k for k in metrics['state_order'])+','+
               ','.join('tableau_'+k for k in metrics['state_order']),comments='',fmt=['%d']*3+['%.13e']*10)
    ref_clock = time.perf_counter()
    step_refinement = refine(p,args,trace)
    refinement_wall = time.perf_counter()-ref_clock
    dump(args.out/'step_refinement.json',step_refinement)
    plot(args.figures,trace,ref,p)
    checks = {
        'states_finite':bool(np.isfinite(trace['x']).all() and np.isfinite(ref['x']).all()),
        'state_l2_limit':max(cmp_l2) <= c['max_relative_state_l2'],
        'trajectory_diode_masks_away_from_boundaries':metrics['trajectory_mask_mismatches_away_from_transition']==0,
        'probe_diode_masks_away_from_boundaries':probes['mismatches_away_from_transition']==0,
        'probe_voltage':probes['max_relative_peak_voltage_error'] <= c['max_relative_probe_voltage_peak_error'],
        'all_constructed_strict_masks':probes['all_witness_masks_agree'],
        'kcl':trace['checks']['max_relative_kcl'] <= c['max_relative_kcl'],
        'trapezoid':trace['checks']['max_relative_trapezoid_residual'] <= c['max_relative_trapezoid_residual'],
        'diode_sign':trace['checks']['max_diode_sign_violation_v'] <= c['max_diode_sign_violation_v'],
        'F_shape':all(v['F_shape']==p['published_targets']['F_shape'] for v in modes.values()),
        'all_gates_exercised':all(by_gate.values()),
    }
    outputs = sorted(x for x in args.out.iterdir() if x.name in {
        'compiled_maps.json','topology_summary.json','region_probes.json','metrics.json',
        'rbc_100ns.npz','rbc_zoom.csv','refinement.npz','step_refinement.json'})
    manifest = {
        'doi':p['doi'],'started_utc':started,'finished_utc':utc(),'wall_seconds':time.perf_counter()-clock,
        'elapsed_seconds':{'offline_compile':compile_wall,'dmm_steps':trace['wall_s'],
                           'tableau_steps':ref['wall_s'],'step_refinement':refinement_wall},
        'python':platform.python_version(),'platform':platform.platform(),'numpy':np.__version__,
        'protocol_sha256':sha256(PROTOCOL.read_bytes()).hexdigest(),
        'code_sha256':{str(f.relative_to(ROOT)):sha256(f.read_bytes()).hexdigest()
                       for f in [Path(__file__),ROOT/'src/dmm.py',ROOT/'verification/tableau_reference.py',ROOT/'verification/region_probes.py']},
        'passed':all(checks.values()),'checks':checks,
        'published_mode_and_W_match':all(v['modes']==p['published_targets']['feasible_combinations'] and
                                         v['W_shape']==p['published_targets']['W_shape'] for v in modes.values()),
        'outputs':{f.name:sha256(f.read_bytes()).hexdigest() for f in outputs}
    }
    dump(args.out/'run_manifest.json',manifest)
    print(json.dumps({'checks':checks,'published_mode_and_W_match':manifest['published_mode_and_W_match'],
                      'trajectory_relative_l2':cmp_l2,'visited_masks':metrics['visited_diode_masks'],
                      'wall_seconds':manifest['wall_seconds']},indent=2))
    return 0 if manifest['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
