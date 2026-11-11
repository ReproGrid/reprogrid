"""Compare analytic Nyquist loci with the vector traces in manuscript Fig. 5."""

import argparse
from collections import defaultdict
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
import pymupdf
from scipy.optimize import brentq
from scipy.spatial import cKDTree


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PAPER = ROOT.parents[1] / "research/feasibility_batch1/sources/robust_psc2019.pdf"
PAPER_SHA256 = "e45aa9f7dc1688da6f125144619fe742220c0766ed8487f4cac47ff8a12b02e5"
DOI = "10.1109/TIE.2018.2874584"
PAGE = 6
BLUE = (0.0, 0.0, 1.0)
SCR_BY_DASH_COUNT = {0: 10, 2: 3, 4: 1}
OPERATING_POINTS = {"high_active": (0.95, 0.2), "high_reactive": (0.2, -0.95)}
MODELS = {
    "figure_labels": (False, False),
    "flip_upper_iq": (True, False),
    "round_scr3_inductance": (False, True),
    "flip_upper_iq_and_round_scr3_inductance": (True, True),
}
# Labelled grid lines in the PDF. Use the left-neighbouring −0.5 tick and
# the below-neighbouring −0.2 tick rather than the unlabelled plot edges.
X_TICKS_PT = {
    "active_power": (157.00509643554688, 129.6678924560547),
    "dc_link": (293.5360412597656, 266.1988525390625),
}
Y_TICKS_PT = {
    "high_active": (106.45367431640625, 117.389404296875),
    "high_reactive": (220.83538818359375, 231.77105712890625),
}


def published_segments(pdf):
    """Read the blue paths from the pinned vector page, in calibrated units."""
    segments = defaultdict(list)
    with pymupdf.open(pdf) as doc:
        for path in doc[PAGE - 1].get_drawings():
            if path["color"] != BLUE or path["rect"].y1 > 280:
                continue
            condition = "high_active" if path["rect"].y0 < 160 else "high_reactive"
            loop = "active_power" if path["rect"].x1 < 160 else "dc_link"
            dash_count = len(path["dashes"].split("]", 1)[0][1:].split())
            scr = SCR_BY_DASH_COUNT[dash_count]
            x_zero, x_tick = X_TICKS_PT[loop]
            y_zero, y_tick = Y_TICKS_PT[condition]
            x_scale = 0.5 / (x_zero - x_tick)
            y_scale = 0.2 / (y_tick - y_zero)

            def calibrate(point):
                return complex((point.x - x_zero) * x_scale, (y_zero - point.y) * y_scale)

            for item in path["items"]:
                if item[0] != "l":
                    raise ValueError("Expected a vector line segment in Fig. 5")
                segments[condition, loop, scr].append((calibrate(item[1]), calibrate(item[2])))
    if len(segments) != 12:
        raise ValueError(f"Expected 12 curve series, found {len(segments)}")
    return segments


def computed_locus(frequency, loop, current_d, current_q, inductance):
    s = 1j * frequency
    a = inductance * current_q
    active_resistance = 0.2 * s / (s + 0.1)
    b = -(current_q / inductance + current_d**2 + current_q**2) * active_resistance**2
    denominator = s**2 + 2 * active_resistance * s / inductance + 1 + (active_resistance / inductance)**2
    gp = 0.2 * (a * s**2 + 1 + a + b) / (inductance * s * denominator)
    if loop == "active_power":
        return gp
    return gp / (1 + gp) / (4 * np.sqrt(2) * s)


def visible_margins(segments):
    """Estimate margins only where the required crossing appears in the panel."""
    gain_margins = []
    phase_margins = []
    distance_to_minus_one = np.inf
    for start, end in segments:
        delta = end - start
        if delta:
            projection = -((start + 1) * delta.conjugate()).real / abs(delta)**2
            closest = start + delta * np.clip(projection, 0, 1)
            distance_to_minus_one = min(distance_to_minus_one, abs(closest + 1))
        if start.imag * end.imag < 0:
            point = start - delta * start.imag / delta.imag
            # Avoid high-frequency paths nearly coincident with the origin.
            if point.real < -0.03:
                gain_margins.append(float(-1 / point.real))
        if (abs(start) - 1) * (abs(end) - 1) < 0:
            t = brentq(lambda fraction: abs(start + fraction * delta) - 1, 0, 1)
            phase = np.angle(start + t * delta, deg=True)
            phase_margins.append(float((360 + phase) % 360 - 180))
    return {
        "visible_gain_margins_factor": gain_margins,
        "visible_phase_margins_deg": phase_margins,
        "visible_peak_sensitivity_estimate": float(1 / distance_to_minus_one),
    }


def compare(segments):
    frequency = np.geomspace(1e-6, 1e3, 300_001)
    metrics = []
    for (condition, loop, scr), path in sorted(segments.items()):
        points = np.unique(np.asarray(path).ravel())
        # Exclude edges where clipped path endpoints can be synthetic.
        points = points[(points.real > -1.499) & (points.real < 0.0009)
                        & (points.imag > -0.749) & (points.imag < 0.749)]
        xy = np.column_stack((points.real, points.imag))
        current_d, printed_iq = OPERATING_POINTS[condition]
        candidates = {}
        for name, (flip_iq, round_inductance) in MODELS.items():
            current_q = -printed_iq if flip_iq and condition == "high_active" else printed_iq
            inductance = 0.33 if round_inductance and scr == 3 else 1 / scr
            curve = computed_locus(frequency, loop, current_d, current_q, inductance)
            distances, _ = cKDTree(np.column_stack((curve.real, curve.imag))).query(xy)
            candidates[name] = {
                "i_d0_pu": current_d,
                "i_q0_pu": current_q,
                "L_pu": inductance,
                "rms_nearest_locus_distance_pu": float(np.sqrt(np.mean(distances**2))),
                "max_nearest_locus_distance_pu": float(np.max(distances)),
            }
        metrics.append({
            "condition": condition,
            "loop": loop,
            "scr_label": scr,
            "compared_unique_vertices": len(points),
            **visible_margins(path),
            "candidates": candidates,
        })
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, default=PAPER)
    parser.add_argument("--out", type=Path, default=HERE / "fig5_vector_comparison.json")
    args = parser.parse_args()
    digest = sha256(args.pdf.read_bytes()).hexdigest()
    if digest != PAPER_SHA256:
        raise ValueError("The PDF does not match the pinned Aalto manuscript")
    report = {
        "doi": DOI,
        "source": "https://aaltodoc.aalto.fi/server/api/core/bitstreams/c8da5953-833c-4f32-b38b-49ea7add52b9/content",
        "pdf_sha256": digest,
        "pdf_page_one_based": PAGE,
        "figure": "Fig. 5",
        "method": "Nearest Euclidean distance of each non-edge PDF vector vertex to a dense positive-frequency locus from Eq. (9), Eq. (12), and Section III-C. The computed samples use 300001 log-spaced frequencies from 1e-6 to 1e3 p.u. Distances include plot precision and finite frequency-sampling error.",
        "margin_method": "Interpolate visible negative-real-axis crossings at Re < -0.03 and visible unit-circle crossings on the PDF vector segments. An empty list indicates that no qualifying crossing was visible within the plotted axes.",
        "sensitivity_method": "Inverse minimum distance from -1 to the visible PDF line segments. Plot resolution and clipping limit this estimate.",
        "published_plot_calibration": {
            "real_axis_0_and_minus_half_tick_pt": X_TICKS_PT,
            "imaginary_axis_0_and_minus_point_two_tick_pt": Y_TICKS_PT,
        },
        "environment": {name: version(name) for name in ("numpy", "scipy", "pymupdf")},
        "curves": compare(published_segments(args.pdf)),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
