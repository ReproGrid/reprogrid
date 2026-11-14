"""Read plotted polylines and pole-marker centers from the frozen author manuscript."""

import argparse
import csv
from hashlib import sha256
from pathlib import Path

import pymupdf


EXPECTED_READBACKS = {
    "fig5_6_bode.csv": "698a366313bc195c4506954b37810cfaca3526dfdd61f4c96a1925db6e67fdb9",
    "fig8_9_poles.csv": "fe5789fd475840b60e574ab459105458798f2fc9248e2347d296ab247c8b2942",
}
CONTROLLERS = ("traditional", "power_only", "acceleration_only", "combined")
INDUCTANCES = ("0.5", "1.0", "1.5")

# In each row use centers of labelled major ticks to calibrate the axes.
BODE = (
    ("5", "magnitude_db", 78.44511, 293.15762, 69.21436, -10 / (89.48132 - 69.21436),
     299, CONTROLLERS, (81, 76, 80, 63)),
    ("5", "phase_deg", 78.07427, 293.15762, 135.20059, -45 / (148.19176 - 135.20059),
     157, CONTROLLERS, (81, 76, 80, 63)),
    ("6", "magnitude_db", 78.47706, 293.13144, 250.72067, -2 / (264.61578 - 250.72067),
     620, INDUCTANCES, (63, 63, 63)),
    ("6", "phase_deg", 78.10374, 292.75815, 315.79266, -10 / (338.57111 - 315.79266),
     475, INDUCTANCES, (63, 63, 63)),
)
# Read each vector group in PDF paint order. For k12=50 one branch marker
# falls outside the x-axis at the initial gain, leaving two at k11=1000.
LOCI = (
    ("8a", "traditional", (76.90688, 296.06342, -12., -2.), 225.62834, -20 / (245.95131 - 225.62834),
     162, (3,) * 20, range(1, 21)),
    ("8b", "combined", (77.81958, 295.62137, -25., 0.), 344.96112, -.5 / (365.18798 - 344.96112),
     64, (3,) * 20, range(1, 21)),
    ("9", "30.0", (342.07675, 557.93774, -45., 0.), 225.88825, -10 / (242.00014 - 225.88825),
     328, (3,) * 21, range(1000, 5001, 200)),
    ("9", "50.0", (342.07675, 557.93774, -45., 0.), 225.88825, -10 / (242.00014 - 225.88825),
     391, (2,) + (3,) * 20, range(1000, 5001, 200)),
    ("9", "70.0", (342.07675, 557.93774, -45., 0.), 225.88825, -10 / (242.00014 - 225.88825),
     453, (4,) * 21, range(1000, 5001, 200)),
)


def read_bode(drawings):
    for panel, quantity, first_x, last_x, zero_y, scale_y, start, series, lengths in BODE:
        for index, (name, length) in enumerate(zip(series, lengths, strict=True)):
            segments = drawings[start + index]["items"]
            if len(segments) != length or any(item[0] != "l" for item in segments):
                raise ValueError(f"Unexpected path geometry in Fig. {panel}/{name}/{quantity}")
            points = [segments[0][1], *(item[2] for item in segments)]
            offset = -4 if panel == "6" and quantity == "magnitude_db" else 0
            for x, y in points:
                log_frequency = -2 + 4 * (x - first_x) / (last_x - first_x)
                yield panel, name, quantity, 10**log_frequency, offset + (y - zero_y) * scale_y


def read_loci(drawings):
    for panel, name, axis, zero_y, scale_y, seq, counts, parameters in LOCI:
        xa, xb, xmin, xmax = axis
        for count, value in zip(counts, parameters, strict=True):
            for _ in range(count):
                marker = drawings[seq]
                seq += 1
                if len(marker["items"]) != 4 or any(item[0] != "l" for item in marker["items"]):
                    raise ValueError(f"Unexpected marker geometry in Fig. {panel}/{name}")
                x0, y0, x1, y1 = marker["rect"]
                x, y = (x0 + x1) / 2, (y0 + y1) / 2
                yield panel, name, value, xmin + (x - xa) * (xmax - xmin) / (xb - xa), (y - zero_y) * scale_y


def write(path, header, rows):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with pymupdf.open(args.pdf) as pdf:
        bode = {x["seqno"]: x for x in pdf[5].get_drawings()}
        loci = {x["seqno"]: x for x in pdf[6].get_drawings()}
        write(args.out / "fig5_6_bode.csv", ("figure", "series", "quantity", "frequency_hz", "value"), read_bode(bode))
        write(args.out / "fig8_9_poles.csv", ("figure", "series", "sweep_value", "real_per_s", "imag_per_s"), read_loci(loci))
    # The repository adds a date to its cover. Verify the extracted evidence directly.
    for name, expected in EXPECTED_READBACKS.items():
        if sha256((args.out / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"{name} differs from the figure readback in the source manifest")


if __name__ == "__main__":
    main()
