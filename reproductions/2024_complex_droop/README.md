# 第七篇 · 复杂下垂控制的稳定条件

单台并网变流器的 2、4、8、12 阶平均模型，覆盖作者稿算例 I–III。结果包括电压跌落、四阶临界增益、充分条件、极限环和作者图对照。主实现与独立两轴求解器的结果都保存在轨迹文件中。

[复现报告](reports/REPORT.md) · [协议](configs/protocol.json) · [来源核对](sources/source_audit.md)

使用 Python 3.12，在本目录中运行以下命令。输出位于独立的重跑目录。

```sh
python3.12 -m venv --copies .venv-replay
.venv-replay/bin/python -m pip install -r requirements.txt
OPENBLAS_NUM_THREADS=1 .venv-replay/bin/python src/reproduce.py --out results/replay
.venv-replay/bin/python src/compare_replay.py --replay results/replay --report results/replay/comparison.json
.venv-replay/bin/python src/plot_results.py --results results/replay --figures results/replay/figures
```

如需复算作者图对照，另安装 `verification/requirements.txt`，按来源清单取得 arXiv v3 排版文件，再运行 `verification/compare_published_curves.py --results results/replay --figures results/replay/figures`。已归档的数值结果在 `results/`，图在 `figures/`。
