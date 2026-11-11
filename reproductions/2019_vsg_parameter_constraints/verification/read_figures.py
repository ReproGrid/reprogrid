"""Read the vector axes and numeric curves of Figs. 2–5 from the UCD PDF."""

from argparse import ArgumentParser
import csv
from pathlib import Path

import numpy as np
import pymupdf


BODE_AXES = {
    2: ((104.502, 253.005, 0., 5.),
        ([258.286, 269.971, 281.916, 293.601, 305.550, 317.494], [200, 100, 0, -100, -200, -300]),
        ([320.872, 338.014, 355.414, 372.555], [0, -180, -360, -540])),
    3: ((364.537, 511.976, 1., 5.),
        ([112.517, 132.076, 151.905, 171.723], [200, 0, -200, -400]),
        ([175.075, 192.320, 209.302, 226.549], [0, -180, -360, -540])),
    4: ((364.429, 510.950, 0., 5.),
        ([283.577, 303.136, 322.965, 342.783], [200, 0, -200, -400]),
        ([346.135, 363.380, 380.362, 397.609], [0, -180, -360, -540])),
}
BODE_COLORS = {
    (0., 0., 1.): "line",
    (0., .647, 0.): "swing",
    (0., 1., 1.): "converter",
    (1., 0., 0.): "full",
    (0., 0., 0.): "reduced",
}
STEP_COLORS = {(0., 0., 1.): "case1", (0., .498, 0.): "case2", (1., 0., 0.): "case3"}
STEP_X = (375.353, 512.259)
STEP_Y = ([446.112, 462.104, 478.099, 494.296, 510.282, 526.479], [6, 4, 2, 0, -2, -4])


def vertices(path):
    ordered = []
    for operation, *points in path["items"]:
        if operation != "l":
            raise ValueError("Expected straight segments for the selected curve.")
        for point in points:
            xy = (point.x, point.y)
            if not ordered or xy != ordered[-1]:
                ordered.append(xy)
    return ordered


def readback(pdf):
    with pymupdf.open(pdf) as doc:
        paths = doc[3].get_drawings()
    rows = []
    for path in paths:
        color = path.get("color")
        if color is None or len(path["items"]) < 20:
            continue
        rgb = tuple(round(c, 3) for c in color)
        rect = path["rect"]
        if rgb in STEP_COLORS and rect.x0 >= STEP_X[0] - 1 and rect.x1 <= STEP_X[1] + 1 and 445 < rect.y0 < rect.y1 < 528:
            for x, y in vertices(path):
                t = (x - STEP_X[0]) / (STEP_X[1] - STEP_X[0])
                rows.append((5, "power_step_mw", STEP_COLORS[rgb], t, np.interp(y, *STEP_Y)))
        for fig, (x_axis, magnitude, phase) in BODE_AXES.items():
            x0, x1, decade0, decade1 = x_axis
            if rgb not in BODE_COLORS or rect.x0 < x0 - .6 or rect.x1 > x1 + .6 or rect.width < .8 * (x1 - x0):
                continue
            for axis, anchors in (("magnitude_db", magnitude), ("phase_deg", phase)):
                y0, y1 = anchors[0][0], anchors[0][-1]
                if rect.y0 < y0 - .6 or rect.y1 > y1 + .6:
                    continue
                for x, y in vertices(path):
                    decade = decade0 + (x - x0) / (x1 - x0) * (decade1 - decade0)
                    rows.append((fig, axis, BODE_COLORS[rgb], 10**decade, np.interp(y, *anchors)))
    groups = {(fig, axis, series) for fig, axis, series, *_ in rows}
    if len(groups) != 33:
        raise ValueError(f"Expected all 30 Bode and 3 step curves; got {len(groups)}.")
    return rows


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path, help="The four-page UCD copy including its repository cover.")
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    rows = readback(args.pdf)
    with args.out.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("figure", "axis", "series", "abscissa", "value"))
        writer.writerows(rows)
    print(f"{len(rows)} vertices across 33 curves -> {args.out}")


if __name__ == "__main__":
    main()
