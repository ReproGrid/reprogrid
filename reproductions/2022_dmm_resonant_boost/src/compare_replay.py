"""Verify a clean-environment replay, excluding elapsed times and runtime labels."""
from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
import json

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_text())


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def main():
    ap = ArgumentParser(description=__doc__)
    ap.add_argument('--primary', type=Path, default=ROOT/'results')
    ap.add_argument('--replay', type=Path, default=ROOT/'results/clean_replay')
    ap.add_argument('--report', type=Path, help='Optional destination for the comparison report.')
    args = ap.parse_args()
    p, r = args.primary, args.replay
    a, b = [read_json(q/'run_manifest.json') for q in (p,r)]
    checks = {f'same_{k}':a[k]==b[k] for k in ('protocol_sha256','code_sha256','checks','published_mode_and_W_match')}
    checks['numerical_checks_passed_both'] = a['passed'] and b['passed']
    for q,m,name in ((p,a,'primary'),(r,b,'replay')):
        checks[name+'_recorded_output_hashes'] = all(digest(q/f)==expected for f,expected in m['outputs'].items())
    for f in ('metrics.json','topology_summary.json','compiled_maps.json','region_probes.json','rbc_zoom.csv'):
        checks['same_'+f] = digest(p/f)==digest(r/f)
    for group in ('published_edges','fixed_edges'):
        x, y = (read_json(q/'step_refinement.json')[group] for q in (p,r))
        checks[f'same_refinement_{group}'] = x['edges_ns']==y['edges_ns'] and x['pairwise_relative_l2']==y['pairwise_relative_l2']
    arrays = {}
    for f in ('rbc_100ns.npz','refinement.npz'):
        with np.load(p/f) as x, np.load(r/f) as y:
            arrays[f] = {k:{'shape':list(x[k].shape),'exact':np.array_equal(x[k],y[k])}
                         for k in x.files if k in y.files}
            checks['same_'+f] = x.files==y.files and all(v['exact'] for v in arrays[f].values())
    out = {'passed':all(checks.values()),'checks':checks,'arrays':arrays,
           'primary_manifest_sha256':digest(p/'run_manifest.json'),'replay_manifest_sha256':digest(r/'run_manifest.json'),
           'verification_scope':'Independent Python environment. Scientific arrays, matched tables and metrics; runtime fields are excluded.',
           'replay_runtime':{k:b[k] for k in ('python','numpy','platform','started_utc','finished_utc','wall_seconds','elapsed_seconds')}}
    (args.report or p/'clean_replay_comparison.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'passed':out['passed'],'checks':checks},indent=2))
    return 0 if out['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
