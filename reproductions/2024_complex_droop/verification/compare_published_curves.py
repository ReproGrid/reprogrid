"""Read fixed vector paths from the arXiv v3 figures and compare the replay."""
from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
import csv
import json
import tarfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pymupdf

ROOT = Path(__file__).resolve().parents[1]
FIGURES = {
    4: ('fig4-simulation-case-a', 'ec50ea1405a75aac15270ad0016868f2a5fd733a0b1a244a3a50218ae7d42307'),
    5: ('fig5-simulation-case-b1', '70b5ba941bdc08010e44158064d610e07087f19de2ff8756cee3d05f5e922b88'),
    6: ('fig6-simulation-case-b2', '630e8b8ed6eb439ffe1200570045fdd45531ef2bf2fe160500d40caf99d17131'),
    7: ('fig7-simulation-case-c1', 'b0fb79f9ff0df329dabbddaa11529445fb2a894fc109cd5be0c5be88ea9efb44'),
}
ARCHIVE_HASH = 'c4ad5adb11758fc8dca268e86eb89080840c97c26eaeb31826b6c6771be22a4d'


def path_points(page, drawings, index):
    items = drawings[index]['items']
    positions = np.linspace(0., 1., 25)
    bezier = np.column_stack(((1 - positions)**3, 3 * positions * (1 - positions)**2,
                              3 * positions**2 * (1 - positions), positions**3))
    segments = []
    for kind, *points in items:
        rotated = [point * page.rotation_matrix for point in points]
        control = np.array([(point.x, point.y) for point in rotated])
        if kind == 'l':
            segments.append(control)
        elif kind == 'c':
            segments.append(bezier @ control)
        else:
            raise ValueError(f'Unexpected path operation {kind} in drawing {index}')
    xy = np.vstack(segments)
    # Exported coordinates sometimes coincide after rounding. Average those
    # ordinates to interpolate the curve on a regular physical time grid.
    unique, mapping, counts = np.unique(xy[:, 0], return_inverse=True, return_counts=True)
    return unique, np.bincount(mapping, weights=xy[:, 1]) / counts


def calibrated(points, x_pair, x_values, y_pair, y_values):
    x, y = points
    x_out = x_values[0] + (x - x_pair[0]) * (x_values[1] - x_values[0]) / (x_pair[1] - x_pair[0])
    y_out = y_values[0] + (y - y_pair[0]) * (y_values[1] - y_values[0]) / (y_pair[1] - y_pair[0])
    return x_out, y_out


def xy_tick(page, drawings, index, axis):
    rect = drawings[index]['rect'] * page.rotation_matrix
    return rect.x0 if axis == 'x' else rect.y0


def calibration(page, drawings, x_indices, x_values, y_indices, y_values):
    x_pos = [xy_tick(page, drawings, i, 'x') for i in x_indices]
    y_pos = [xy_tick(page, drawings, i, 'y') for i in y_indices]
    return x_pos, x_values, y_pos, y_values


def score(time_src, value_src, t_sim, y_sim, start, stop, interval):
    lower = max(time_src[0], t_sim[0], start)
    upper = min(time_src[-1], t_sim[-1], stop)
    common = np.arange(lower, upper, interval)
    if len(common) < 3:
        raise ValueError('Source curve has no usable time overlap')
    author = np.interp(common, time_src, value_src)
    replay = np.interp(common, t_sim, y_sim)
    difference = replay - author
    return {'sample_count': len(common), 'start_s': float(common[0]), 'end_s': float(common[-1]),
            'rmse_pu': float(np.sqrt(np.mean(difference**2))),
            'peak_abs_error_pu': float(np.abs(difference).max())}, (common, author, replay)


def compare_fig4(page, data):
    drawings = page.get_drawings()
    scenarios = [
        (.02, 0, (3, 4), (9, 11), (.6, 1.), [15, 16, 17, 18]),
        (.06, 0, (21, 22), (27, 29), (.6, 1.), [33, 34, 35, 36]),
        (.02, 1, (39, 40), (45, 46), (.05, .1), [49, 50, 51, 52]),
        (.06, 1, (73, 74), (79, 80), (0., .1), [83, 84, 85, 86]),
    ]
    rows = []
    for eta, component, xticks, yticks, volts, paths in scenarios:
        cal = calibration(page, drawings, xticks, (.5, .6), yticks, volts)
        for order, idx in zip((12, 8, 4, 2), paths):
            t, v = calibrated(path_points(page, drawings, idx), *cal)
            name = f'case_i_eta_{eta:.3f}_order_{order}'
            metric, _ = score(t, v, data[f'{name}__t'], data[f'{name}__x'][:, component], .45, .7, .0005)
            rows.append({'eta_over_omega0': eta, 'order': order, 'component': 'dq'[component], **metric})
    return rows


def compare_fig5(page, sweep):
    drawings = page.get_drawings()
    all_rows = []
    for rg, axes, curve_idx in [(0.08, (9, 6, 4, 11), (22, 97)), (0.20, (16, 13, 2, 18), (21, 98))]:
        y_tick = .05 if rg == .08 else .2
        cal = calibration(page, drawings, axes[:2], (0., 2.), axes[2:], (0., y_tick))
        branches = [r for r in sweep if float(r['rg']) == rg and r['angle_branch'] == 'impedance_matched']
        a = np.array([float(r['alpha']) for r in branches])
        summary = {'rg': rg}
        for name, path_idx, field in [
            ('simulation', curve_idx[0], 'critical_eta_over_omega0'),
            ('sufficient', curve_idx[1], 'certificate_supremum'),
        ]:
            source_alpha, source_eta = calibrated(path_points(page, drawings, path_idx), *cal)
            keep = (source_alpha >= -1e-8) & (source_alpha <= a[-1] + 1e-8)
            source_alpha, source_eta = source_alpha[keep], source_eta[keep]
            numeric = np.array([float(r[field] or 'nan') for r in branches])
            evaluated = np.interp(source_alpha, a, numeric)
            valid = np.isfinite(evaluated)
            d = evaluated[valid] - source_eta[valid]
            summary[name] = {
                'alpha_1_source': float(np.interp(1., source_alpha, source_eta)),
                'alpha_1_replay': float(np.interp(1., a, numeric)),
                'sample_count': len(d),
                'rmse_eta_over_omega0': float(np.sqrt(np.mean(d**2))),
                'peak_abs_difference_eta_over_omega0': float(np.abs(d).max()),
            }
        all_rows.append(summary)
    return all_rows


def compare_fig6(page, data):
    drawings = page.get_drawings()
    rows = []
    scenarios = [(.099, (2, 3, 8, 9), (12, 13)), (.101, (16, 17, 22, 23), (26, 64))]
    for eta, ticks, paths in scenarios:
        cal = calibration(page, drawings, ticks[:2], (2., 2.5), ticks[2:], (.5, 1.))
        for order, path in zip((4, 2), paths):
            t, v = calibrated(path_points(page, drawings, path), *cal)
            prefix = f'case_ii_eta_{eta:.3f}_order_{order}'
            numeric = np.linalg.norm(data[f'{prefix}__x'][:, :2], axis=1)
            metric, _ = score(t, v, data[f'{prefix}__t'], numeric, 2., 2.2, .001)
            rows.append({'eta_over_omega0': eta, 'order': order, **metric})
    return rows


def compare_fig7(page, data):
    drawings = page.get_drawings()
    cal = calibration(page, drawings, (30, 31), (2., 2.5), (28, 35), (.5, 1.))
    traces, rows = {}, []
    for order, path in [(12, 27), (2, 38)]:
        t, v = calibrated(path_points(page, drawings, path), *cal)
        prefix = f'case_iii_alpha_3.0_order_{order}'
        numeric = np.linalg.norm(data[f'{prefix}__x'][:, :2], axis=1)
        metric, trace = score(t, v, data[f'{prefix}__t'], numeric, 1.5, 4., .001)
        traces[order] = trace
        rows.append({'order': order, **metric})
    return rows, traces


def main():
    ap = ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, default=ROOT / 'sources/arxiv_2310.09933v3_source.tar.gz')
    ap.add_argument('--results', type=Path, default=ROOT / 'results')
    ap.add_argument('--figures', type=Path, default=ROOT / 'figures')
    args = ap.parse_args()
    if sha256(args.source.read_bytes()).hexdigest() != ARCHIVE_HASH:
        raise ValueError('Calibration requires the pinned arXiv v3 source archive')
    args.figures.mkdir(parents=True, exist_ok=True)
    results = {}
    with (args.results / 'stability_sweep.csv').open() as file:
        sweep = list(csv.DictReader(file))
    with tarfile.open(args.source) as source, np.load(args.results / 'trajectories.npz') as data:
        for n, (name, expected) in FIGURES.items():
            content = source.extractfile(f'Figures/{name}.pdf').read()
            if sha256(content).hexdigest() != expected:
                raise ValueError(f'Unexpected Fig. {n} contents')
            with pymupdf.open(stream=content, filetype='pdf') as doc:
                page = doc[0]
                if n == 4:
                    results['fig4'] = compare_fig4(page, data)
                elif n == 5:
                    results['fig5'] = compare_fig5(page, sweep)
                elif n == 6:
                    results['fig6'] = compare_fig6(page, data)
                else:
                    results['fig7'], traces = compare_fig7(page, data)
    results['scope'] = 'Vector plot readback from arXiv v3. Independent time and voltage calibration from axis ticks, without rescaling or aligning waveforms. Fig. 6 measures 2.00–2.20 s to avoid the later printed clipping. Fig. 5 compares the local eigenvalue crossing with the simulation-traversal boundary.'
    results['arxiv_source_sha256'] = ARCHIVE_HASH
    (args.results / 'published_curve_comparison.json').write_text(json.dumps(results, indent=2) + '\n')
    fig, axes = plt.subplots(2, 1, figsize=(9.1, 5.5), sharex=True, layout='constrained')
    for ax, order in zip(axes, (12, 2)):
        t, author, replay = traces[order]
        ax.plot(t, author, color='#c16737', lw=1.5, label='Author Fig. 7, vector readback')
        ax.plot(t, replay, color='#16678e', lw=1.15, ls='--', label='Independent replay')
        ax.set(ylabel=r'$|\hat{v}|$ (pu)', title=f'{order} state averaged model')
        ax.grid(alpha=.25)
    axes[0].legend(loc='upper right', fontsize=8.5)
    axes[1].set_xlabel('Time (s)')
    fig.savefig(args.figures / 'fig7_published_curve_comparison.png', dpi=185)
    print(json.dumps({name: results[name] for name in ('fig5', 'fig6', 'fig7')}, indent=2))


if __name__ == '__main__':
    main()
