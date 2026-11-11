# 单机 VSG 稳定性参数约束

复现 IEEE TPWRS 10.1109/TPWRS.2019.2896853 的完整和二阶传递函数、独立 dq 状态模型、三个印刷工况与图 2–5 对照。算例 2 同时保留印刷值和两个推断的残余电感。算例 3 的模型判据和论文 EMT 波形分别保存在报告。

## 重跑

在本目录用 Python 3.12。数值读图文件已随复现冻结。下列命令使用新的输出目录。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
OPENBLAS_NUM_THREADS=1 .venv/bin/python src/reproduce.py --out results/replay_new
OPENBLAS_NUM_THREADS=1 .venv/bin/python src/compare_replay.py \
  --primary results --primary-figures figures \
  --replay results/replay_new --out results/replay_new/comparison.json
```

若需核对读图坐标，可在已安装 PyMuPDF 的环境中，用 `verification/read_figures.py» 读取含仓储封面的四页 UCD 稿，输出新的 CSV。默认精确重跑对照使用相同版本的 macOS arm64 环境。

## 文件

- [主报告](reports/REPORT.md)、[来源核对](sources/source_audit.md)、[冻结配置](configs/protocol.json)
- [数值结果](results/metrics.json)、[内环时间常数扫描](results/time_constant_sweep.csv)、[频响数组](results/frequency_responses.npz)、[功率阶跃数组](results/step_responses.npz)
- [Bode 对照](figures/bode_validation.png)、[功率阶跃对照](figures/step_comparison.png)
- [独立 dq 实现](verification/dq_reference.py)、[干净环境重跑](results/clean_replay_comparison.json)
