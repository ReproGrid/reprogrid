# 单机 VSG 暂态阻尼整定

IEEE TPEL 10.1109/TPEL.2021.3074027。由式 (1)–(7) 重建静态 Q–V 闭合、相平面和增益稳定区。计算采用四组阻抗和频率条件，按原文的越过不稳定平衡角判据核对图 3–4。

在本目录使用 Python 3.12。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
OPENBLAS_NUM_THREADS=1 .venv/bin/python src/reproduce.py --out results/replay_new
.venv/bin/python src/compare_replay.py \
  --primary results --primary-figures figures \
  --replay results/replay_new --out results/replay_new/comparison.json
```

第一条计算命令可单独运行。包内比较器检查相同 Python 和库版本的输出文件及数组。冻结的图 3–4 坐标随工程提供。若重取坐标，安装 PyMuPDF 后执行 `verification/read_figures.py`，参数依次为 Aalborg PDF 和新 CSV。

- [报告](reports/REPORT.md) · [冻结条件](configs/protocol.json) · [来源核对](sources/source_audit.md)
- [计算指标](results/metrics.json) · [时序轨迹](results/phase_traces.npz) · [稳定区网格](results/stability_maps.npz)
- [功角曲线](figures/power_angle.png) · [相平面](figures/phase_portraits.png) · [稳定区对照](figures/stability_maps.png)
- [独立环境重跑](results/clean_replay_comparison.json)

`phase_traces.npz` 的分支数组按时间、状态、增益排列，三个状态依次为 δ、Δω 和低通状态 h。 `stability_maps.npz` 的矩阵按 α 和 Kₕ 排列。1 表示能量判据确认收敛，−1 表示越过不稳定平衡角，0 表示在观察时限内尚未分类。
