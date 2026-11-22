"""Calibrate Fig. 3 paths and the observable signals in the Fig. 5–6 scope images."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pymupdf


ROOT = Path(__file__).resolve().parents[1]
PDF_SHA256 = "ffcc20bbf011ad0350524bc8a5bede18070494d038fcac9fa6cdd3d2042abaec"
AXIS = {"x0": 229.479553, "x_pi_over_4": 259.448151, "y_zero": 120.078979, "y_one": 79.5241547}
FIGURES = {
    "5": {"xref": 260, "time_div_s": 0.2, "angle_turns_div": 0.05, "power_roi": [565, 678],
          "angle_roi": [380, 466], "voltage_roi": [186, 264], "edge_search_px": [[195, 260], [418, 485]],
          "nominal_duration_s": 0.45},
    "6": {"xref": 282, "time_div_s": 0.4, "angle_turns_div": 1.0, "power_roi": [465, 667],
          "angle_roi": [346, 453], "voltage_roi": [186, 280], "edge_search_px": [[195, 260], [355, 422]],
          "nominal_duration_s": 0.65},
}
X_PX_DIV = 100
Y_PX_DIV = (698.7 - 46.4) / 10


def vector_data(page):
    paths = {p["seqno"]: p for p in page.get_drawings()}
    angle = np.linspace(0, 2*np.pi, 721)
    normal_sag = []
    for path_id in (154, 155):
        segments = paths[path_id]["items"]
        if any(s[0] != "l" for s in segments):
            raise ValueError("Unexpected structure of the published Fig. 3 paths")
        x, y = np.array([tuple(segments[0][1]), *(tuple(s[2]) for s in segments)]).T
        coordinates = (x - AXIS["x0"]) * (np.pi/4) / (AXIS["x_pi_over_4"] - AXIS["x0"])
        watts = (AXIS["y_zero"] - y) / (AXIS["y_zero"] - AXIS["y_one"])
        if not np.all(np.diff(coordinates) >= 0):
            raise ValueError("Nonmonotone path; inspect the figure before interpolating")
        normal_sag.append(np.interp(angle, coordinates, watts))
    # The left hand panel magnifies the two equilibria. It has the same y scale.
    fine_angle = np.linspace(0., np.pi/8, 641)
    fine = []
    for path_id in (62, 63):
        segments = paths[path_id]["items"]
        x, y = np.array([tuple(segments[0][1]), *(tuple(s[2]) for s in segments)]).T
        coordinates = (x - 126.08786) * (np.pi/8) / (215.99367 - 126.08786)
        watts = (AXIS["y_zero"] - y) / (AXIS["y_zero"] - AXIS["y_one"])
        fine.append(np.interp(fine_angle, coordinates, watts))
    return np.column_stack((angle, *normal_sag)), np.column_stack((fine_angle, *fine))


def raster_center(image, channel, bounds):
    top, bottom = bounds
    red, green, blue = image[top:bottom].astype(np.int16).transpose(2, 0, 1)
    if channel == "power":
        selected = (blue > 65) & (blue - red > 48) & (green < 72)
    elif channel == "angle":
        selected = (green > 65) & (blue > 65) & (np.minimum(green, blue) - red > 55) & (abs(green - blue) < 95)
    else:
        selected = (red > 58) & (blue > 58) & (np.minimum(red, blue) - green > 55)
    centers = np.full(selected.shape[1], np.nan)
    for x in range(21, 1021):
        ys = np.flatnonzero(selected[:, x])
        if ys.size >= 2:
            centers[x] = top + np.median(ys)
    return centers


def scope_data(document, figure, cfg):
    # The raw image XObject is vertically reflected relative to the displayed PDF.
    pix = pymupdf.Pixmap(document, cfg["xref"])
    im = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if im.shape != (715, 1024, 3):
        raise ValueError(f"Unexpected raster shape for Fig. {figure}")
    blue = raster_center(im, "power", cfg["power_roi"])
    cyan = raster_center(im, "angle", cfg["angle_roi"])
    magenta = raster_center(im, "voltage", cfg["voltage_roi"])
    # Each step is localized by the mid-crossing of its two voltage plateaus.
    edges = []
    for left, right in cfg["edge_search_px"]:
        before = np.nanmedian(magenta[left-18:left-4])
        after = np.nanmedian(magenta[right+4:right+18])
        candidates = np.arange(left, right)
        direction = np.sign(after - before)
        crosses = candidates[direction * (magenta[candidates] - (before + after) / 2) >= 0]
        if len(crosses) == 0 or abs(after - before) < 10:
            raise ValueError(f"Missing visible voltage transition for Fig. {figure}")
        edges.append(int(crosses[0]))
    start, end = edges
    xs = np.arange(start - 55, 1020)
    ratios = np.isfinite(np.c_[blue[xs], cyan[xs], magenta[xs]]).mean(axis=0)
    if np.min(ratios) < .97:
        raise ValueError(f"Colored trace coverage below 97% for Fig. {figure}: {ratios}")
    t = (xs - start) * cfg["time_div_s"] / X_PX_DIV

    def interpolate(pixel_y):
        finite = np.isfinite(pixel_y)
        return np.interp(xs, np.flatnonzero(finite), pixel_y[finite])

    pre = slice(start - 50, start - 9)
    power = 0.8 + (interpolate(blue) - np.nanmedian(blue[pre])) / Y_PX_DIV
    phase = (interpolate(cyan) - np.nanmedian(cyan[pre])) * (2*np.pi * cfg["angle_turns_div"] / Y_PX_DIV)
    voltage_disp = (interpolate(magenta) - np.nanmedian(magenta[pre])) / Y_PX_DIV
    return np.column_stack((t, power, phase, voltage_disp)), {
        "first_voltage_edge_px": start, "second_voltage_edge_px": end,
        "sag_duration_from_image_s": float((end-start)*cfg["time_div_s"]/X_PX_DIV),
        "nominal_duration_s": cfg["nominal_duration_s"],
        "angle_rad_per_px": 2*np.pi*cfg["angle_turns_div"] / Y_PX_DIV,
        "power_pu_per_px": 1 / Y_PX_DIV, "coverage": ratios.tolist(),
    }


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    if sha256(args.pdf.read_bytes()).hexdigest() != PDF_SHA256:
        raise ValueError("The calibration corresponds to the exact author manuscript hash")
    args.output.mkdir(exist_ok=True, parents=True)
    with pymupdf.open(args.pdf) as doc:
        main_curve, inset = vector_data(doc[3])
        arrays = {"fig3": main_curve, "fig3_fine": inset}
        meta = {"pdf_sha256": PDF_SHA256,
                "fig3_columns": ["angle_rad", "power_at_1pu", "power_at_0p5pu"],
                "scope_columns": ["time_s", "terminal_power_pu", "phase_change_vs_prefault_rad", "voltage_change_vs_prefault_pu"],
                "notes": "Digitized from the visible PDF graphics. Scope phase uses 1 pu per turn and power baseline 0.8 pu. Scope traces are relative, with independently calibrated time/div and V/div.",
                "scope": {}}
        for figure, cfg in FIGURES.items():
            arrays[f"fig{figure}"], meta["scope"][figure] = scope_data(doc, figure, cfg)
    np.savez_compressed(args.output / "published_readback.npz", **arrays)
    (args.output / "published_readback.json").write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    main()
