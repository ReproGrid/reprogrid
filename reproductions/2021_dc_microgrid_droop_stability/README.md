# System-Level Large-Signal Stability Analysis of Droop-Controlled DC Microgrids

[Publisher DOI](https://doi.org/10.1109/TPEL.2020.3019311) · [Source and parameter provenance](sources/source_manifest.json) · [Reproduction report (中文)](reports/REPORT.md)

## Replay

Use Python 3.12 and run these commands from this directory.

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

```sh
OPENBLAS_NUM_THREADS=1 python src/reproduce.py --out replay
```

Recorded checksums are in the run manifests. Compare on the same numerical stack for byte equality. The report states the model scope, assumptions and which published comparisons remain open.

The [MIT license](LICENSE) applies to the ReproGrid implementation. The published paper and original waveform data are linked through the source manifest.
