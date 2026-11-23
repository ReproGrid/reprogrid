"""Read the Fig. 9 and Fig. 13 curves from the hash-pinned IEEE PDF."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from math import pi
from pathlib import Path

import numpy as np
import pymupdf


ROOT = Path(__file__).resolve().parents[1]
PDF_SHA256 = "e2711a74f5ce25801dbcd9d11e6493c2eab0ce3f4aef7feed9fa0204e90ae533"


def coordinates(path):
    return np.array([xy for item in path["items"] if item[0] in ("l", "c") for xy in (tuple(item[1]), tuple(item[-1]))])


def read_curves(pdf):
    if sha256(pdf.read_bytes()).hexdigest() != PDF_SHA256:
        raise ValueError("The axis anchors correspond to the pinned IEEE PDF")
    with pymupdf.open(pdf) as doc:
        paths = {p["seqno"]: p for p in doc[5].get_drawings()}
        curve = coordinates(paths[412])
        # Grid ticks x=(0.6, 1), y=(0, 15). Keep the first unbroken y segment.
        ratio = .6 + .4*(curve[:,0]-363.5409)/(490.0209-363.5409)
        damping = 15*(377.8849-curve[:,1])/(377.8849-295.5649)
        valid = damping <= 15
        fig9 = np.unique(np.column_stack((ratio, damping))[valid], axis=0).tolist()
        paths = {p["seqno"]: p for p in doc[8].get_drawings()}

        def scale(points, y_top, y_bottom):
            return np.column_stack((-pi + (3*pi/2)*(points[:,0]-98.2258)/(254.9448-98.2258),
                                    10 - 30*(points[:,1]-y_top)/(y_bottom-y_top)))

        results = {}
        for name, y_top, y_bottom, curve_keys, initial_marker, final_marker in (
            ("0.14", 323.9342, 414.5342, (565, 566, 567), 621, 615),
            ("0.10", 448.1354, 538.7344, (820, 821, 822), 768, 762),
        ):
            curves = [scale(coordinates(paths[key]), y_top, y_bottom) for key in curve_keys]
            red = curves[2]
            initial = paths[initial_marker]["rect"]
            final = paths[final_marker]["rect"]
            markers = [[(m.x0+m.x1)/2, (m.y0+m.y1)/2] for m in (initial, final)]
            xy = scale(np.array(markers), y_top, y_bottom)
            order = np.argsort(red[:,0])
            results[name] = {"pre_event_marker_rad_hz": xy[0].tolist(), "final_marker_rad_hz": xy[1].tolist(),
                             "adaptive_frequency_at_zero_angle_hz": float(np.interp(0., red[order,0], red[order,1])),
                             "paths": {key: a.tolist() for key,a in zip(("case_i", "case_ii", "case_iii"), curves, strict=True)}}
    return {"pdf_sha256": PDF_SHA256, "units": {"fig9_ratio": "abs(vzq)/V_fault", "fig13": "phase in rad, frequency deviation in Hz"},
            "fig9": fig9, "fig13": results}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, default=ROOT / "sources/08820019.pdf")
    parser.add_argument("--output", type=Path, default=ROOT / "results/published_curves.json")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(read_curves(args.pdf), indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
