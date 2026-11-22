"""Fetch the twelve hash-locked inputs used by the six-sweep replay."""
from hashlib import sha256
import json
from pathlib import Path
from urllib.request import urlopen

sources = Path(__file__).resolve().parents[1] / "sources"
manifest = json.loads((sources / "source_manifest.json").read_text())
origin = "https://raw.githubusercontent.com/Future-Power-Networks/Simplus-Grid-Tool/"
case_directory = "SingleInverterCase_Simulation/Case2_SingleInverterInfiniteBus/"
for name, digest in manifest["local_source_sha256"].items():
    if not name.startswith(("Case2_GFM_", "Case2_GFL_")) or not name.endswith((".mat", ".png")):
        continue
    target = sources / name
    directory = case_directory if name.endswith(".mat") else "SingleInverterCase_Script/"
    payload = target.read_bytes() if target.exists() else urlopen(
        f"{origin}{manifest['commit']}/{directory}{name}", timeout=25).read()
    if sha256(payload).hexdigest() != digest:
        raise ValueError(f"Source digest mismatch: {name}")
    if not target.exists():
        target.write_bytes(payload)
