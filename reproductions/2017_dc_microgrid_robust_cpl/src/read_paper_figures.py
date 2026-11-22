"""Read vector-plot timing and voltage plateaus from the fixed arXiv PDF."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pymupdf


ROOT = Path(__file__).resolve().parents[1]
PDF_HASH = "eda8457b926ca300205df242ecd2a944a1e02f63481fa8827e1da3a8f9fa9f17"
# Two horizontal and two vertical anchors taken from labelled grid lines.
# The truncated right edge of Fig. 7 has no label. Its time is read via the 0/6 ticks.
AXES = {
    "4": {"page": 8, "power_path": 45, "voltage_path": 99, "x": (98.38861, 257.31616),
          "time": (0, 25.672), "power_y": (358.88391, 304.05884), "kw": (5, 20),
          "voltage_y": (443.52722, 385.04736), "volt": (360, 440)},
    "5": {"page": 8, "power_path": 179, "voltage_path": 225, "x": (98.82117, 259.88879),
          "time": (0, 32), "power_y": (572.36029, 525.65417), "kw": (5, 20),
          "voltage_y": (651.44458, 607.37354), "volt": (385, 400)},
    "7": {"page": 9, "power_path": 363, "voltage_path": 429, "x": (362.95786, 500.93253),
          "time": (0, 6), "power_y": (296.54169, 257.61340), "kw": (10, 20),
          "voltage_y": (384.33179, 329.83398), "volt": (350, 400)},
    "8": {"page": 9, "power_path": 478, "voltage_path": 524, "x": (362.27313, 526.60840),
          "time": (2, 24), "power_y": (520.54114, 473.66156), "kw": (5, 20),
          "voltage_y": (598.75549, 545.62427), "volt": (410, 440)},
}


def affine(values, points, units):
    return units[0] + (values - points[0]) * (units[1] - units[0]) / (points[1] - points[0])


def trace_segments(paths, seqno, axis, y_key, units):
    path = next(p for p in paths if p["seqno"] == seqno)
    segments = np.array([[tuple(left), tuple(right)] for command, left, right in path["items"] if command == "l"])
    segments[:, :, 0] = affine(segments[:, :, 0], axis["x"], axis["time"])
    segments[:, :, 1] = affine(segments[:, :, 1], axis[y_key], axis[units])
    return segments


def plateau(segments, interval):
    times = np.linspace(*interval, 31)
    left, right = segments[:, 0, 0], segments[:, 1, 0]
    endpoint_t, endpoint_v = segments.reshape(-1, 2).T
    visible = endpoint_v[(endpoint_t >= interval[0]) & (endpoint_t <= interval[1])]
    values = []
    for time in times:
        crossing = (left <= time) & (time < right)
        if np.any(crossing):
            weight = (time - left[crossing]) / (right[crossing] - left[crossing])
            values.extend(segments[crossing, 0, 1] + weight * (segments[crossing, 1, 1] - segments[crossing, 0, 1]))
    covered = len(values) == len(times)
    return {"coverage_samples": len(values), "pdf_median_v": float(np.median(values)) if covered else None,
            "pdf_span_v": float(np.ptp(np.r_[visible, values])) if covered else None}


def extract(pdf, protocol):
    if sha256(pdf.read_bytes()).hexdigest() != PDF_HASH:
        raise ValueError("The plot anchors correspond to the exact PDF in sources/source_manifest.json")
    report = {"source_pdf_sha256": PDF_HASH,
              "method": "Affine conversion of PDF vector paths. Use 31 uniform samples for the median and visible vector vertices for the voltage span.",
              "figures": {}}
    with pymupdf.open(pdf) as document:
        for key, axis in AXES.items():
            paths = document[axis["page"] - 1].get_drawings()
            power = trace_segments(paths, axis["power_path"], axis, "power_y", "kw")
            voltage = trace_segments(paths, axis["voltage_path"], axis, "voltage_y", "volt")
            rises = power[(power[:, 1, 1] - power[:, 0, 1]) > 0.5]
            profile = protocol["trajectories"][key]
            changes = np.r_[profile["start_s"], profile["first_step_s"] + profile["step_period_s"] * np.arange(profile["steps"]),
                            profile["stop_s"]]
            if len(rises) != profile["steps"]:
                raise ValueError(f"Check the Fig. {key} transcription against the PDF")
            windows = []
            for level, end in enumerate(changes[1:]):
                interval = (end - np.array(protocol["numerics"]["plateau_reference_window_before_step_s"])).tolist()
                windows.append({"power_kw": profile["first_kw"] + level * profile["increment_kw"], "window_s": interval,
                                **plateau(voltage, interval)})
            report["figures"][key] = {"power_end_s_from_pdf": float(power[-1, -1, 0]),
                                     "step_pdf_midpoint_s": np.mean(rises[:, :, 0], axis=1).tolist(),
                                     "power_level_kw_pdf": [float(power[0, 0, 1]), *rises[:, 1, 1].tolist()],
                                     "plateau_windows": windows}
    return report


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, default=ROOT / "sources/arxiv_1702_07580.pdf")
    parser.add_argument("--out", type=Path, default=ROOT / "results/pdf_vector_readback.json")
    args = parser.parse_args()
    protocol = json.loads((ROOT / "configs/protocol.json").read_text())
    args.out.write_text(json.dumps(extract(args.pdf, protocol), indent=2) + "\n")


if __name__ == "__main__":
    main()
