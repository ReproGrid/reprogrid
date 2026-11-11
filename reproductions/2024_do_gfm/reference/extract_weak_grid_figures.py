"""Extract the published Fig. 5(b) and 6(b) stair traces from the vector PDF."""

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pymupdf


PAPER_DOI = "10.1109/TPEL.2024.3433503"
PAPER_URL = "https://acris.aalto.fi/ws/portalfiles/portal/162149547/Multifunctional_Grid-Forming_Converter_Control_Based_on_a_Disturbance_Observer.pdf"
PAPER_SHA256 = "a17aee546525bd418ced2ea980c18c82ca2f32bd552daf774415ddbce64cb51e"
PAGE = 7
SPAN_S = 1.5
SAMPLE_PERIOD_S = 0.0004
LEVELS_PU = (0.0, 0.33, 0.66, 1.0, 0.0)
FIELDS = (
    "time_s", "time_aligned_s", "p_g_ref_pu", "p_g_pu", "q_g_pu", "u_c_ref_pu",
    "u_g_pu", "u_c_hat_pu",
)
BLACK = (0.0, 0.0, 0.0)
BLUE = (0.0, 0.0, 1.0)
RED = (1.0, 0.0, 0.0)


@dataclass(frozen=True)
class Axis:
    left_pt: float
    right_pt: float
    top_pt: float
    bottom_pt: float
    low_pu: float
    high_pu: float

    def time(self, x):
        return SPAN_S * (x - self.left_pt) / (self.right_pt - self.left_pt)

    def value(self, y):
        return self.low_pu + (self.high_pu - self.low_pu) * (
            (self.bottom_pt - y) / (self.bottom_pt - self.top_pt)
        )


@dataclass(frozen=True)
class Stair:
    starts: np.ndarray
    levels: np.ndarray
    end: float

    def at(self, time):
        index = np.searchsorted(self.starts, time, side="right") - 1
        return self.levels[np.clip(index, 0, len(self.levels) - 1)]


def wide_paths(drawings, left, y_min, y_max):
    for path in drawings:
        rect = path["rect"]
        if (path["type"] == "s" and abs(rect.x0 - left) < 0.02
                and 196 < rect.width < 198 and y_min <= rect.y0 <= rect.y1 <= y_max):
            yield path


def read_axis(paths, low, high):
    frames = [path["rect"] for path in paths if len(path["items"]) == 1
              and path["rect"].height == 0 and 0.14 < path["color"][0] < 0.16]
    top, bottom = sorted(frames, key=lambda rect: rect.y0)
    return Axis(bottom.x0, bottom.x1, top.y0, bottom.y0, low, high)


def read_trace(paths, axis, color, *, command=False):
    path, = [path for path in paths if path["color"] == color
             and (len(path["items"]) == 9 if command else len(path["items"]) > 7000)]
    horizontal = []
    for line_type, start, end in path["items"]:
        if line_type != "l" or (start.x != end.x and start.y != end.y) or start.x > end.x:
            raise ValueError("Unexpected vector segment in a stair trace")
        if start.x < end.x:
            horizontal.append((start.x, end.x, start.y))
    starts, ends, levels = np.asarray(horizontal).T
    np.testing.assert_allclose(ends[:-1], starts[1:], atol=0.002, rtol=0)
    return Stair(axis.time(starts), axis.value(levels), axis.time(ends[-1]))


def step_metrics(traces, time):
    reference = traces["p_g_ref_pu"]
    p_g = traces["p_g_pu"]
    metrics = []
    for index, (start, stop) in enumerate(zip(reference.starts[1:], [*reference.starts[2:], SPAN_S])):
        before, after = LEVELS_PU[index:index + 2]
        use = (p_g.starts >= start) & (p_g.starts < stop)
        normalized = (p_g.levels[use] - before) / (after - before)
        times = p_g.starts[use]
        first_crossings = [times[normalized >= fraction][0] for fraction in (0.1, 0.9)]
        plateau = time[(time >= stop - 0.05) & (time < stop)]
        metrics.append({
            "time_s": float(start),
            "nominal_p_g_ref_before_pu": before,
            "nominal_p_g_ref_after_pu": after,
            "t10_t90_ms": float(1000 * (first_crossings[1] - first_crossings[0])),
            "directional_overshoot_percent_of_command_step": float(100 * max(0, normalized.max() - 1)),
            "last_50ms_means_pu": {name: float(trace.at(plateau).mean())
                                  for name, trace in traces.items() if name != "p_g_ref_pu"},
            "min_u_c_ref_pu_after_step": float(traces["u_c_ref_pu"].at(time[(time >= start) & (time < stop)]).min()),
        })
    return metrics


def extract_panel(drawings, left, fig, name, output):
    power = list(wide_paths(drawings, left, 229, 283))
    voltage = list(wide_paths(drawings, left, 289, 343))
    power_axis = read_axis(power, -1.0, 1.5)
    voltage_axis = read_axis(voltage, 0.8, 1.2)
    traces = {
        "p_g_ref_pu": read_trace(power, power_axis, BLACK, command=True),
        "p_g_pu": read_trace(power, power_axis, BLUE),
        "q_g_pu": read_trace(power, power_axis, RED),
        "u_c_ref_pu": read_trace(voltage, voltage_axis, BLUE),
        "u_g_pu": read_trace(voltage, voltage_axis, BLACK),
    }
    if name == "do_gfm":
        traces["u_c_hat_pu"] = read_trace(voltage, voltage_axis, RED)
    reference = traces["p_g_ref_pu"]
    np.testing.assert_allclose(reference.levels, LEVELS_PU, atol=5e-5, rtol=0)
    np.testing.assert_allclose(np.diff(reference.starts[1:]), 0.3, atol=2e-5, rtol=0)
    shift = float(0.2 - reference.starts[1])
    time = np.arange(round(SPAN_S / SAMPLE_PERIOD_S) + 1) * SAMPLE_PERIOD_S
    traces_sampled = {key: trace.at(time) for key, trace in traces.items()}
    csv_name = f"fig{fig}b_{name}.csv"
    with (output / csv_name).open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(FIELDS)
        for index, timestamp in enumerate(time):
            values = [timestamp, timestamp + shift]
            values.extend(traces_sampled[key][index] if key in traces_sampled else None for key in FIELDS[2:])
            writer.writerow([f"{value:.8f}" if value is not None else "" for value in values])
    return {
        "figure": f"{fig}(b)", "csv": csv_name, "row_count": len(time),
        "axis_calibration": {"power": vars(power_axis), "voltage": vars(voltage_axis)},
        "extracted_command_levels_pu": reference.levels.tolist(),
        "command_edges_plot_s": reference.starts[1:].tolist(),
        "aligned_time_offset_s": shift,
        "steps": step_metrics(traces, time),
    }


def main():
    project = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path,
                        default=project / "research/feasibility_batch1/sources/do_gfm2024.pdf")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    actual_sha = hashlib.sha256(args.pdf.read_bytes()).hexdigest()
    if actual_sha != PAPER_SHA256:
        raise ValueError("Use the PDF with the expected SHA-256 for this coordinate calibration")
    args.out.mkdir(parents=True, exist_ok=True)
    with pymupdf.open(args.pdf) as document:
        drawings = document[PAGE - 1].get_drawings()
    report = {
        "doi": PAPER_DOI, "paper_url": PAPER_URL, "paper_pdf_sha256": actual_sha,
        "pdf_page_one_based": PAGE, "printed_page": 13028, "weak_grid_total_inductance_pu": 0.89,
        "method": "Extract the colored stair polylines from the PDF and calibrate them with the rendered axes. The CSV samples the right-continuous hold on a 0.4 ms grid.",
        "scope": "Published plot digitization. Original laboratory telemetry and sample clocks were unavailable. Horizontal segments and repeated values can be merged in the PDF path.",
        "csv_step_s": SAMPLE_PERIOD_S,
        "clock_alignment": "The first command edge for each published trace is translated to 0.2 s. CSV time_s retains the paper clock.",
        "timing_metric": "First 10% and 90% crossings on the published vector stairs relative to the nominal command levels. Approximate 0.4 ms plot sampling. Tail means use the last 50 ms before the next command or 1.5 s.",
        "voltage_signal": "u_c_hat_pu represents the magnitude of the red estimated disturbance voltage trace shown in the proposed controller figure.",
        "panels": {
            "rfpsc": extract_panel(drawings, 78.639, 5, "rfpsc", args.out),
            "do_gfm": extract_panel(drawings, 341.655, 6, "do_gfm", args.out),
        },
    }
    (args.out / "paper_metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
