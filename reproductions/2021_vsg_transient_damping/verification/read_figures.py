"""Read vector coordinates from Figs. 3 and 4 of the frozen Aalborg PDF."""

import argparse
import csv
from hashlib import sha256
from pathlib import Path

import pymupdf


SOURCE_SHA256 = "8f4fd9d4ec04d3f59cf438867cf06aa7a9d52e5ac59b08949e56ad121c724b59"

# Bounds are calibrated at the labelled ticks in PDF points, with y downward.
PANELS = (
    (3, "3a", (351.113, 553.507, 267.531, 127.752), (0.5, 2.5, -0.5, 2.0),
     [("0", 406, 62), ("10", 407, 88), ("20", 388, 206), ("50", 408, 251), ("60", 409, 126)]),
    (3, "3b", (352.095, 555.033, 449.607, 310.409), (-0.5, 2.0, -400., 200.),
     [("20", 483, 202), ("50", 482, 256), ("60", 484, 131)]),
    (4, "4", (88.541, 274.350, 400.760, 256.373), (0., 3.5, 0., 250.),
     [("stable_fill_1", 113, 266), ("stable_fill_2", 115, 1179)]),
)


def readback(document):
    for page, panel, box, bounds, curves in PANELS:
        x0, x1, y0, y1 = box
        xmin, xmax, ymin, ymax = bounds
        drawing = {p["seqno"]: p for p in document[page].get_drawings()}
        for name, path_id, count in curves:
            segments = drawing[path_id]["items"]
            if len(segments) != count or any(segment[0] != "l" for segment in segments):
                raise ValueError(f"Unexpected vector geometry for {panel}/{name}")
            points = [segments[0][1], *(segment[2] for segment in segments)]
            for i, (x, y) in enumerate(points):
                yield panel, name, i, xmin + (x - x0) * (xmax - xmin) / (x1 - x0), ymin + (y - y0) * (ymax - ymin) / (y1 - y0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("csv", type=Path)
    args = parser.parse_args()
    if sha256(args.pdf.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("Expected the Aalborg PDF documented in sources/source_manifest.json")
    with pymupdf.open(args.pdf) as document, args.csv.open("w", newline="") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(("panel", "series", "index", "x", "y"))
        writer.writerows(readback(document))


if __name__ == "__main__":
    main()
