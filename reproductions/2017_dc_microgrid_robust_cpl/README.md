# DC microgrid with constant power loads / 恒功率负荷直流微网

Independent nine-bus and eight-bus circuit models, common Lyapunov certificates, numerical results and figures. The nine-bus crossing is about 16.057 kW per load. The public 440 V / 20 kW case gives a 399.91 V minimum CPL terminal voltage and a positive rightmost pole. Verifying the original engineering condition still requires the original simulation project and load commands.

独立重建九节点与八节点。九节点临界功率约 16.057 kW/路。公开 440 V、20 kW/路工况下，最低负荷端电压为 399.91 V，最右极点实部为正。原 Simulink 工程和逐路负荷指令仍待核对。

Use Python 3.12. Install requirements in an isolated environment and run from this directory.

```sh
python -m pip install -r requirements.txt
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg   python src/reproduce.py --out replay-output
python verification/common_mode.py --out replay-output/common_mode.json
python verification/lower_delta_sensitivity.py --out replay-output/lower_delta_sensitivity.json
```

The replay uses the bundled aggregate plot readback. To repeat the extraction, obtain the PDF at the pinned link in the source manifest and run `python src/read_paper_figures.py --pdf author-paper.pdf --out replay-output/pdf_vector_readback.json`.

[Report](reports/REPORT.md) · [Source manifest](sources/source_manifest.json) · [Metrics](results/metrics.json) · [Model arrays](results/numerical_evidence.npz)
