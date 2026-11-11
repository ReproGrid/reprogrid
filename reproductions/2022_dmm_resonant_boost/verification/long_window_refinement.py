"""Extend the predeclared refinement steps to the thesis plot time window."""
from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from reproduce import DMM, run_dmm


def main():
    ap = ArgumentParser(description=__doc__)
    ap.add_argument('--results',type=Path,default=ROOT/'results')
    args = ap.parse_args()
    out = args.results
    p = json.loads((ROOT/'configs/protocol.json').read_text())
    dots = np.loadtxt(out/'fig314_comparison.csv',delimiter=',',skiprows=1)
    start = datetime.now(timezone.utc).isoformat()
    tick = perf_counter()
    records = {}
    records['100'] = {'published_dmm_marker_relative_l2':float(np.linalg.norm(dots[:,2]-dots[:,1])/np.linalg.norm(dots[:,1])),
                      'origin':'frozen primary run'}
    for step in p['refinement']['steps_ns'][1:]:
        dmm = DMM(p,step)
        tr = run_dmm(p,dmm,None,step,p['primary']['stop_ns'])
        t = tr['t_ns']/1000
        i = tr['x'][:,1]
        sampled = np.interp(dots[:,0],t,i)
        selected = (t>=dots[0,0]) & (t<=dots[-1,0])
        records[str(step)] = {'edges_ns':tr['edges'],'dmm_steps_wall_seconds':tr['wall_s'],
                              'published_dmm_marker_relative_l2':float(np.linalg.norm(sampled-dots[:,1])/np.linalg.norm(dots[:,1])),
                              'min_max_current_a':[float(i[selected].min()),float(i[selected].max())]}
        np.savetxt(out/f'fig314_replay_{step}ns.csv',np.column_stack([dots[:,0],sampled]),delimiter=',',
                   header='time_us,replay_ir_A',comments='',fmt='%.9g')
        print(step,records[str(step)],flush=True)
    result = {'started_utc':start,'finished_utc':datetime.now(timezone.utc).isoformat(),'wall_seconds':perf_counter()-tick,
              'scope':'Same gate rule, parameters, and initial state for all steps; extends the frozen refinement list to 38 ms for comparison with the published plot. No fitted shifts/gains. Different grids move gate times and change integration error.',
              'per_step_ns':records}
    (out/'fig314_step_diagnostic.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main()
