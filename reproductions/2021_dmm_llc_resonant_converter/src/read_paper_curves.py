"""Read source vector paths in a hash-pinned author manuscript; do not align outputs."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA = 'dc7413ba7d1903889a44681acff066dd13a0562317759e1abb7ea756da8f57e4'

# Each axis tuple contains two (PDF coordinate, physical coordinate) pairs.
# Anchors are the printed ticks, never simulation peaks or fitted alignments.
PANELS = {
    'fig10a': (7, {'vo': 20}, ((349.56003, 0.0), (549.82703, 0.6)), ((116.50397, 0.0), (66.021, 20.0))),
    'fig10b': (7, {'fs_khz': 46}, ((349.56003, 0.0), (549.82703, 0.6)), ((177.33197, 300), (131.646, 500))),
    'fig11a': (8, {'vo': 24}, ((88.5523, 0.049995), (183.759, 0.050005)), ((151.65802, 18.43), (93.106, 18.46))),
    'fig11b': (8, {'ir': 56, 'im': 57}, ((89.0516, 0.049996), (181.673, 0.050004)), ((278.45001, -20), (200.38196, 20))),
    'fig11c': (8, {'vo': 83}, ((204.73601, 0.09999), (302.38998, 0.10001)), ((161.41699, 0), (83.34503, 20))),
    'fig11d': (8, {'ir': 107, 'im': 108}, ((206.11201, 0.099995), (303.08002, 0.100005)), ((275.06201, -60), (207.177, 40))),
    'fig11e': (8, {'vo': 141}, ((331.04401, 0.2596), (422.05899, 0.2604)), ((161.41699, 17.4), (83.34503, 18.6))),
    'fig11f': (8, {'ir': 5539, 'im': 5540}, ((332.08102, 0.25999), (421.186, 0.26001)), ((278.45001, -40), (200.38196, 60))),
    'fig11g': (8, {'vo': 5572}, ((454.55698, 0.549996), (544.18402, 0.550004)), ((161.41699, 11.866), (83.34503, 11.878))),
    'fig11h': (8, {'ir': 5594, 'im': 5595}, ((469.84302, 0.549998), (528.20001, 0.550002)), ((278.45001, -10), (200.38196, 10))),
    'fig3a': (3, {'fe_ir': 19}, ((98.309, 0.047894), (281.009, 0.047904)), ((93.893, 0), (78.268, 20))),
    'fig4a': (3, {'fe_ir': 71}, ((364.666, 0.047901), (533.432, 0.047906)), ((118.138, -20), (80.208, 10))),
}


def affine(values, points):
    (p0, x0), (p1, x1) = points
    return x0 + (values - p0) * (x1 - x0) / (p1 - p0)


def read_stroke(item, horizontal, vertical):
    if item['type'] != 's' or not all(line[0] == 'l' for line in item['items']):
        raise RuntimeError('Expected one solid vector trace made of line segments')
    xy = np.asarray([item['items'][0][1], *(line[2] for line in item['items'])])
    time = affine(xy[:, 0], horizontal)
    value = affine(xy[:, 1], vertical)
    # The PDF discretizes coordinates; repeated abscissae mark vertical strokes or pixel rounding.
    # For waveform comparisons use the last vertex at each repeated x, preserving the original path in CSVs.
    return np.column_stack((time, value))


def plateaus(path, timing, gate):
    expected_lengths = list(range(gate['initial_half_period_ticks'], gate['final_half_period_ticks'] - 1, -1))
    shelves = path[1:][(np.abs(np.diff(path[:, 0])) > 0.005) & (np.abs(np.diff(path[:, 1])) < 1e-6)]
    starts = path[:-1][(np.abs(np.diff(path[:, 0])) > 0.005) & (np.abs(np.diff(path[:, 1])) < 1e-6), 0]
    if len(shelves) != len(expected_lengths):
        raise RuntimeError(f'Found {len(shelves)} plateaus for {len(expected_lengths)} expected levels')
    values = shelves[:, 1]
    frequencies_khz = 1e9 / (2 * timing['clock'] * np.array(expected_lengths))
    transitions = (timing['ramp_start'] + timing['frequency_hold'] * np.arange(len(starts) - 1)) * 1e-12
    return {
        'readback_count': len(values), 'pdf_levels_khz': values.tolist(),
        'pdf_segment_start_s': starts.tolist(), 'integer_half_period_ticks': expected_lengths,
        'schedule_khz': frequencies_khz.tolist(), 'max_level_mismatch_khz': float(np.max(np.abs(values - frequencies_khz))),
        'max_transition_time_error_s': float(np.max(np.abs(starts[1:] - transitions))),
    }


def periods(path):
    t, current = path.T
    roots = []
    # Use negative-to-positive zero crossings so nearby commutation fluctuations do not bias the period.
    for i in np.flatnonzero((current[:-1] < 0) & (current[1:] >= 0) & (np.diff(t) > 0)):
        roots.append(float(t[i] - current[i] * (t[i + 1] - t[i]) / (current[i + 1] - current[i])))
    return {'positive_crossings_s': roots,
            'period_readback_hz': float(1 / np.median(np.diff(roots))) if len(roots) >= 2 else None}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--pdf', type=Path, default=ROOT / 'sources/arxiv_author_manuscript.pdf')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    import pymupdf

    if sha256(args.pdf.read_bytes()).hexdigest() != SOURCE_SHA:
        raise SystemExit('PDF hash differs from the author version used for source-axis calibration.')
    args.out.mkdir(parents=True, exist_ok=True)
    with pymupdf.open(args.pdf) as pdf:
        drawings = {page: pdf[page - 1].get_drawings() for page in (3, 7, 8)}
    p = json.loads((ROOT / 'configs/deep_protocol.json').read_text())
    signals = {}
    manifest = {'source': 'https://arxiv.org/pdf/2006.04155v1', 'sha256': SOURCE_SHA, 'extractor': pymupdf.VersionBind,
                'panels': {}, 'policy': 'PDF stroke readback only, not original samples. No simulation-based axis or phase adjustment.'}
    for panel, (page, layers, h, v) in PANELS.items():
        counts = {}
        for field, idx in layers.items():
            name = f'{panel}_{field}'
            data = read_stroke(drawings[page][idx], h, v)
            counts[field] = len(data)
            signals[name] = data
            np.savetxt(args.out / f'{name}.csv', data, delimiter=',', fmt='%.12g', header=f'time_s,{field}', comments='')
        manifest['panels'][panel] = {'pdf_page': page, 'path_indices': layers, 'time_anchors': h,
                                     'value_anchors': v, 'vertices': counts}
    manifest['plateau_check'] = plateaus(signals['fig10b_fs_khz'], p['timing_ps'], p['gate_schedule'])
    manifest['fig3_4_positive_crossing_frequency'] = {f: periods(signals[f'{f}_fe_ir']) for f in ('fig3a', 'fig4a')}
    (args.out / 'readback_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(manifest['plateau_check']['readback_count'], 'plateaus', manifest['fig3_4_positive_crossing_frequency'])


if __name__ == '__main__':
    main()
