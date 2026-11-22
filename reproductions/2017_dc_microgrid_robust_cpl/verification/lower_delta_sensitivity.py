"""Check the Table II Eq. (7) conclusion with the Section IV.A delta floor."""

from argparse import ArgumentParser
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from certificates import CertificateBox
from network import Components, Network, graph_symmetries


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "verification/lower_delta_sensitivity.json")
    args = parser.parse_args()
    config = ROOT / "configs/protocol.json"
    protocol = json.loads(config.read_text())
    components = Components(**protocol["components"])
    nine, eight = protocol["nine_bus"], protocol["eight_bus"]
    lower = nine["cpl_limits_w"][0] / (components.load_c_f * nine["equilibrium_load_voltage_bounds_v"][1]**2)
    results = []
    for resistance in eight["line_r_ohm"]:
        net = Network(eight, replace(components, line_r_ohm=resistance), eight["droop_ohm"])
        upper = eight["paper_delta_upper_per_s"][str(resistance)]["7"]
        box = CertificateBox(net, lower, upper, protocol["numerics"])
        result, P = box.eq7(permutations=graph_symmetries(net, eight))
        results.append({"line_r_ohm": resistance, "published_upper_per_s": upper, "result": result,
                        "P": P.tolist() if P is not None else None})
    args.out.write_text(json.dumps({"protocol_sha256": sha256(config.read_bytes()).hexdigest(),
                                    "alternate_lower_per_s": lower, "cases": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
