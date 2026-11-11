# PMSM 高频电压注入初始位置估计

IEEE TPEL 10.1109/TPEL.2017.2773094。重建 PI、ESO C1 和 ESO C2 频响、锁定转子脉冲检测、q 轴误锁、非零初值以及开关模型对照。

在本目录用 Python 3.12 重跑。数值读图文件已随复现冻结。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
OPENBLAS_NUM_THREADS=1 .venv/bin/python src/reproduce.py --out results/replay_new
.venv/bin/python src/compare_replay.py \
  --primary results --primary-figures figures \
  --replay results/replay_new --out results/replay_new/comparison.json
OPENBLAS_NUM_THREADS=1 .venv/bin/python verification/review.py --out results/replay_new/review_metrics.json
```

独立复核重算 `results` 中的八组脉冲轨迹、三种观测器频响及判定时刻。如需复核 PDF 坐标，安装 PyMuPDF 后运行 `verification/read_figures.py`，传入含仓储封面的 Aalborg 作者稿和新 CSV 路径。精确重跑采用相同软件版本的 macOS arm64。

- [报告](reports/REPORT.md)、[冻结配置](configs/protocol.json)、[来源核对](sources/source_audit.md)
- [指标](results/metrics.json)、[逐角度结果](results/angle_sweep.csv)、[脉冲轨迹](results/pulse_traces.npz)
- [观测器频响](figures/observer_bode.png)、[归一化对照](figures/detector_normalization.png)、[误锁与收敛](figures/pulse_lock_and_convergence.png)
- [干净环境重跑](results/clean_replay_comparison.json)、[独立复核](verification/review_metrics.json)
