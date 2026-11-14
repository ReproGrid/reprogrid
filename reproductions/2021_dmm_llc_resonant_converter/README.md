# A Direct Mapped Method for Accurate Modeling and Real-Time Simulation of High Switching Frequency Resonant Converters

[Publisher DOI](https://doi.org/10.1109/TIE.2020.2998746) · [Source and parameter provenance](sources/source_manifest.json) · [Reproduction report (中文)](reports/REPORT.md)

## Replay

Use Python 3.12 and run these commands from this directory. LLC also needs a C++17 compiler.

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

```sh
OPENBLAS_NUM_THREADS=1 python src/reproduce.py --out replay
```

Recorded checksums are in the run manifests. Compare on the same numerical stack for byte equality. The report states the model scope, assumptions and which published comparisons remain open.

### Extended sequence

Obtain the [arXiv author manuscript](https://arxiv.org/pdf/2006.04155v1) individually and check its hash in [the source manifest](sources/source_manifest.json). The extractor uses PyMuPDF.

```sh
python -m pip install pymupdf
python src/read_paper_curves.py --pdf /path/to/manuscript.pdf --out source-curves
OPENBLAS_NUM_THREADS=1 python src/deep_reproduce.py --out replay-deep --source-curves source-curves
```

See the [extended report](reports/DETAILED_REPRODUCTION.md). The website ZIP provides metric summaries and generated figures. The commands above regenerate the full-size trajectory arrays.

The [MIT license](LICENSE) applies to the ReproGrid implementation. The published paper and original waveform data are linked through the source manifest.

The public gallery redrew selected plots using only the computed trajectories. Run `python src/plot_site_figures.py` after placing the generated trajectory arrays under `results/` (including `results/deep/` for the LLC sequence).
