"""Extract the vector phase and gain traces in Figs. 10 and 11."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path

import pymupdf


ROOT = Path(__file__).resolve().parents[1]
SHA = "7401ea5bd619f4bdb67f8f191283264bad7acb6aaae585d4adb73d21adae0b13"


def figure_points(stroke, x_axis, y_axis):
    def scale(value, axis):
        a,b,u,v = axis
        return u+(v-u)*(value-a)/(b-a)

    lines = stroke["items"]
    if any(segment[0] != "l" for segment in lines):
        raise ValueError("Unexpected PDF drawing command for the selected trace.")
    points = [lines[0][1], *(segment[2] for segment in lines)]
    return [[scale(point.x,x_axis),scale(point.y,y_axis)] for point in points]


def extract(pdf):
    if sha256(pdf.read_bytes()).hexdigest()!=SHA:
        raise ValueError("Source PDF differs from the figure-axis calibration.")
    with pymupdf.open(pdf) as doc:
        paths = {d["seqno"]:d for d in doc[7].get_drawings()}
        phase = {str(j):figure_points(paths[seq],(390.428,488.125,1,2),(466.332,383.346,0,2.5))
                 for j,seq in enumerate((3008,3064,3065,3014,3018),1)}
        critical_gain = {rg:figure_points(paths[seq],(353.7825,544.7354,0,.1),(675.7372,543.0272,0,20))
                         for rg,seq in (("0.000",3245),("0.003",3246))}
    return {"pdf_sha256":SHA,"fig10_phase":phase,"fig11_gain":critical_gain,
            "fig11_text_points":{"0.000":{"0.02":2.6,"0.04":6.2},"0.003":{"0.02":1.4,"0.04":4.8}}}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--pdf",type=Path,default=ROOT/"sources/accepted.pdf")
    parser.add_argument("--out",type=Path,default=ROOT/"results/published_curves.json")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(extract(args.pdf),indent=2)+"\n")


if __name__=="__main__":
    main()
