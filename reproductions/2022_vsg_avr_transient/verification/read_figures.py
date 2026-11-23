"""Read the two voltage sag and four gain trajectories in the pinned PDF."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pymupdf


ROOT = Path(__file__).resolve().parents[1]
SHA = "4092d437b58178dab341d417cdd4e4649589b120da2e52fa60599f85964459c7"


def curve(path, x_axis, y_axis, x_span, y_span, count=480):
    commands = path["items"]
    if any(line[0] != "l" for line in commands):
        raise ValueError("The selected stroke unexpectedly contains a non-line segment")
    points = np.array([tuple(commands[0][1]), *(tuple(line[2]) for line in commands)])
    indices = np.unique(np.r_[np.linspace(0,len(points)-1,count,dtype=int), np.argmin(points,axis=0), np.argmax(points,axis=0)])
    x,y = points[indices].T
    return np.column_stack((x_span[0]+(x-x_axis[0])*(x_span[1]-x_span[0])/(x_axis[1]-x_axis[0]),
                            y_span[0]+(y-y_axis[0])*(y_span[1]-y_span[0])/(y_axis[1]-y_axis[0]))).tolist()


def extract(pdf):
    if sha256(pdf.read_bytes()).hexdigest()!=SHA:
        raise ValueError("Source PDF differs from the figure-axis calibration")
    with pymupdf.open(pdf) as doc:
        f6 = {r["seqno"]:r for r in doc[6].get_drawings()}
        f9 = {r["seqno"]:r for r in doc[8].get_drawings()}
        return {"pdf_sha256":SHA,
                "fig6_phase":{key:curve(f6[j],(93.82738,286.68512),(187.12527,63.68687),(.5,2.5),(.995,1.015)) for key,j in (("0.80",83),("0.60",85))},
                "fig6_voltage":{key:curve(f6[j],(94.17111,286.64313),(316.8056,232.42122),(9,15),(.88,1.03)) for key,j in (("0.80",45),("0.60",46))},
                "fig9_phase":{key:curve(f9[j],(98.37565,279.88843),(179.53197,63.35490),(.5,2.5),(.997,1.009)) for key,j in (("0.00",36),("0.30",37),("0.60",38),("0.90",39))}}


def main():
    parser=ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, default=ROOT/"sources/accepted.pdf")
    parser.add_argument("--out", type=Path, default=ROOT/"results/published_curves.json")
    args=parser.parse_args()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(extract(args.pdf),indent=2)+"\n")


if __name__=="__main__":
    main()
