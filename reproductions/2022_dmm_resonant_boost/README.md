# 第六篇：DMM 谐振升压变流器

[复现报告](reports/REPORT.md) · [冻结协议](configs/protocol.json) · [来源审计](sources/source_audit.md)

复现目标是 TIE 论文 §VI-C 的谐振升压算例及其直接映射算法。代码实现了梯形电阻开关模型、精确可行域筛选和软件映射查询，并用另一套迭代 tableau 对照。结果含原图读回、步长精化和干净环境重跑。硬件计时和多变流器 EVBC 不在复现范围。

Python 3.12；从本目录执行：

```sh
python3.12 -m venv --copies .venv-dmm
.venv-dmm/bin/python -m pip install -r requirements.txt
OPENBLAS_NUM_THREADS=1 .venv-dmm/bin/python src/reproduce.py --out results/replay --figures results/replay/figures
.venv-dmm/bin/python src/compare_replay.py --replay results/replay --report results/replay/comparison.json
```

已审核主结果在 `results/`，图在 `figures/`。上述命令在独立目录写结果。对学位论文 Fig. 3.14 的可选比较另需安装 `verification/requirements.txt` 和来源清单指定的 PDF，运行 `verification/compare_thesis_fig314.py --results results/replay --figures results/replay/figures`；然后可运行 `verification/long_window_refinement.py --results results/replay`。

原始 PDF 仅供本地核验。手工录入的是元件数值与网表；代码自行实现。

## Publication note / 发布说明

Published from the independently verified research handoff. Local paths and internal execution logs were removed. Third-party manuscripts are linked in the source manifest and are not included. The published result data include computational evidence only.
