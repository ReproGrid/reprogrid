"""Calibrate and read vector vertices from Figs. 8 and 9 of the Aalborg PDF."""

from argparse import ArgumentParser
import csv
from pathlib import Path

import numpy as np
import pymupdf


FIG8_X = (346.879, 545.201)
FIG8_AXES = {
    "gain_db": ([535.762, 555.666, 575.911, 596.158, 616.402], [20, 0, -20, -40, -60]),
    "phase_deg": ([621.123, 644.398, 667.343, 690.623], [45, 0, -45, -90]),
}
FIG8_COLORS = {(0., 0., 1.): "PI", (0., .647, 0.): "ESO_C1", (1., 0., 0.): "ESO_C2"}
FIG9_X = (389.614, 503.309)
FIG9_Y = ([53.464, 64.180, 74.889, 85.595, 96.526, 107.235, 117.951, 128.658, 139.580],
          [360, 315, 270, 225, 180, 135, 90, 45, 0])
FIG9_COLORS = {(0., 0., 1.): "true_rotor", (1., 0., 0.): "estimated_rotor"}


def points(path):
    previous = None
    for operator, *vertices in path["items"]:
        if operator != "l":
            raise ValueError("Unexpected curve drawing operator.")
        for point in vertices:
            current = (point.x, point.y)
            if current != previous:
                yield current
            previous = current


def color(path):
    return tuple(round(x, 3) for x in (path.get("color") or ()))


def readback(pdf):
    rows = []
    with pymupdf.open(pdf) as doc:
        for path in doc[5].get_drawings():
            rect = path["rect"]
            name = FIG8_COLORS.get(color(path))
            if name is None or len(path["items"]) < 20 or abs(rect.x0 - FIG8_X[0]) > 0.7:
                continue
            for axis, anchors in FIG8_AXES.items():
                if rect.y0 < anchors[0][0] - 0.7 or rect.y1 > anchors[0][-1] + 0.7:
                    continue
                for x, y in points(path):
                    decade = -2 + 6 * (x - FIG8_X[0]) / (FIG8_X[1] - FIG8_X[0])
                    rows.append((8, axis, name, 10**decade, np.interp(y, *anchors)))
        for path in doc[6].get_drawings():
            rect = path["rect"]
            name = FIG9_COLORS.get(color(path))
            if (name is None or len(path["items"]) < 20 or rect.x0 < FIG9_X[0] - 0.2
                    or rect.x1 > FIG9_X[1] + 0.2 or rect.y0 < 53.4 or rect.y1 > 139.7):
                continue
            for x, y in points(path):
                rows.append((9, "angle_deg", name, .05 * (x - FIG9_X[0]) / (FIG9_X[1] - FIG9_X[0]),
                             np.interp(y, *FIG9_Y)))
    if len({row[:3] for row in rows}) != 8:
        raise ValueError("Expected all six Bode and two angle curves.")
    return rows


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("csv", type=Path)
    args = parser.parse_args()
    rows = readback(args.pdf)
    with args.csv.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("figure", "axis", "series", "abscissa", "value"))
        writer.writerows(rows)
    print(f"Read {len(rows)} points across 8 curves.")


if __name__ == "__main__":
    main()
