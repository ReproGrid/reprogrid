# Circular current limiter / 圆形限流暂态稳定性

Independent reduced-order and sampled circuit models of Fan and Wang's single converter. The 650 ms case agrees with the published slip classification. At 450 ms, the tested models slip once and the published experiment returns without a slip. The numerical reconstruction retains this difference.

复算功角曲线、圆形限流、450/650 ms 跌落和 10 kHz 采样电路。650 ms 的滑相分类与原图一致。450 ms 时，独立模型滑相一周，原实验返回原工作点。原始同步数据、固件增益和执行延迟仍待核对。

Use Python 3.12. Run these commands in this directory.

```sh
python -m pip install -r requirements.txt
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg python src/reproduce.py
python src/render_public.py
```

To regenerate the separate PDF and 10 kHz comparisons, obtain the hash-pinned accepted manuscript linked in the source manifest. Keep it local as `author-paper.pdf`.

```sh
python src/audit_figures.py --pdf author-paper.pdf --output local-figure-audit
python verification/read_published.py --pdf author-paper.pdf --output local-readback
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg   python src/run_extended.py --reference local-readback --output local-sampled-replay
```

The release includes aggregate comparisons and independent model arrays. Public figures show independent outputs. Source PDFs and digitized traces are acquired or regenerated locally.

[Report](reports/REPORT.md) · [Source manifest](sources/source_manifest.json) · [Sampled metrics](results/extended_metrics.json) · [Model arrays](results/extended_model_traces.npz)
