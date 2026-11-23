# 单机 VSG 加速度电压补偿

[Paper DOI](https://doi.org/10.1109/TIE.2021.3114723) · [Technical report](reports/REPORT.md) · [Parameters](configs/protocol.json) · [Source and checksum](sources/source_manifest.json)

## Replay with Python 3.12

Download the version of the [paper](https://vbn.aau.dk/ws/files/452682649/FINAL_VERSION.pdf) listed in the source manifest and save it as `paper.pdf` in this directory. The digitizer checks its SHA-256 before extracting the figure coordinates for local comparison.

```sh
python3.12 -m venv .venv-replay
.venv-replay/bin/python -m pip install -r requirements.txt PyMuPDF==2.1
mkdir -p replay
.venv-replay/bin/python verification/read_figures.py --pdf paper.pdf --out replay/published_curves.json
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg \
  .venv-replay/bin/python src/reproduce.py --reference replay/published_curves.json --out replay
```

Replay and compare [numerical checks](results/metrics.json) and [simulated arrays](results/numerical_traces.npz). The public figures show the independent model. The paper and its extracted vector paths are regenerated locally for comparison. See the report for cases that differ from the paper and the outstanding hardware inputs.
