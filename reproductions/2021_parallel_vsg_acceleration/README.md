# Active Power Oscillation Damping Based on Acceleration Control in Paralleled Virtual Synchronous Generators System

[Publisher DOI](https://doi.org/10.1109/TPEL.2021.3051272) · [Source and parameter provenance](sources/source_manifest.json) · [Reproduction report (中文)](reports/REPORT.md)

## Replay

Use Python 3.12 and run these commands from this directory.

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

To run the figure comparisons, obtain the [Aalborg manuscript](https://vbn.aau.dk/ws/files/452682594/FINAL_VERSION.pdf) individually from the official repository. The extractor uses PyMuPDF and validates the readback checksums, which are independent of the date on the PDF cover.

```sh
python -m pip install pymupdf
python verification/read_figures.py /path/to/manuscript.pdf --out verification
```

```sh
OPENBLAS_NUM_THREADS=1 python src/reproduce.py --out replay
```

Recorded checksums are in the run manifests. Compare on the same numerical stack for byte equality. The report states the model scope, assumptions and which published comparisons remain open.

The [MIT license](LICENSE) applies to the ReproGrid implementation. The published paper and original waveform data are linked through the source manifest.

The public gallery redrew selected plots using only the computed trajectories. Run `python src/plot_site_figures.py` after placing the generated trajectory arrays under `results/` (including `results/deep/` for the LLC sequence).
