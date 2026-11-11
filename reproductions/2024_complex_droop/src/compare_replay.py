"""Check numeric outputs from a fresh virtual environment."""
from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
import json

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def main():
    ap = ArgumentParser(description=__doc__)
    ap.add_argument('--primary', type=Path, default=ROOT / 'results')
    ap.add_argument('--replay', type=Path, default=ROOT / 'results/clean_replay')
    ap.add_argument('--report', type=Path)
    args = ap.parse_args()
    a, b = [json.loads((base / 'run_manifest.json').read_text()) for base in (args.primary, args.replay)]
    checks = {name: a[name] == b[name] for name in ('protocol_sha256', 'code_sha256', 'checks')}
    checks['both_runs_passed'] = a['passed'] and b['passed']
    checks['same_output_set'] = set(a['outputs']) == set(b['outputs'])
    for base, manifest, label in ((args.primary, a, 'primary'), (args.replay, b, 'replay')):
        checks[f'{label}_hashes_verified'] = all(digest(base / name) == expected
                                                 for name, expected in manifest['outputs'].items())
    for name in sorted(a['outputs']):
        if not name.endswith('.npz'):
            checks[f'{name}_identical'] = a['outputs'][name] == b['outputs'].get(name)
    filename = 'trajectories.npz'
    with np.load(args.primary / filename) as x, np.load(args.replay / filename) as y:
        exact = {k: np.array_equal(x[k], y[k]) for k in x.files if k in y.files}
        checks['all_arrays_identical'] = x.files == y.files and all(exact.values())
    result = {
        'passed': all(checks.values()), 'checks': checks, 'array_count': len(exact),
        'arrays_exact': exact,
        'primary_manifest_sha256': digest(args.primary / 'run_manifest.json'),
        'replay_manifest_sha256': digest(args.replay / 'run_manifest.json'),
        'runtime': {key: b[key] for key in ('started_utc', 'finished_utc', 'wall_seconds', 'python', 'numpy', 'scipy')},
    }
    destination = args.report or args.primary / 'clean_replay_comparison.json'
    destination.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'passed': result['passed'], 'arrays': len(exact), 'checks': checks}, indent=2))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
