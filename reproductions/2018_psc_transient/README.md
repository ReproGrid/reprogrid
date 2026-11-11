# PSC 暂态稳定计算复现

Heng Wu 与 Xiongfei Wang 的 *Design-Oriented Transient Stability Analysis of Grid-Connected Converters With Power Synchronization Control*，IEEE Transactions on Industrial Electronics。 [DOI](https://doi.org/10.1109/TIE.2018.2875669) · [作者接收稿](https://arxiv.org/abs/1804.09394v2)

[计算报告](reports/REPORT.md) · [冻结协议](configs/protocol.json) · [来源审计](sources/source_audit.md)

从本目录运行。依赖在 Python 3.12 上锁定并复跑。

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python src/reproduce.py
```

程序输出 8 个 NPZ/CSV 数据集、相图和时域图，以及包含版本、源文件哈希、工况和数值检查的 `results/run_manifest.json`。NPZ/CSV 的相角是未折叠的弧度。拓扑改变时相角连续，功率和电流按右连续的代数解取值。图中的电流是准稳态相量幅值。

可在一个新环境中用同一份依赖重跑到另一个目录，再比较数据。

```sh
python3.12 -m venv .venv-replay
.venv-replay/bin/python -m pip install -r requirements.txt
.venv-replay/bin/python src/reproduce.py --output results/clean_replay --figures results/clean_replay/figures
.venv-replay/bin/python src/compare_replay.py
```

独立验证见 [verification](verification)。主模型通过节点导纳建立各拓扑并用 DOP853/Radau 积分。独立验证从戴维南等效和标量闭式流重新推导。生成报告时对照两套结果。

```sh
.venv/bin/python verification/verifier.py
.venv/bin/python src/build_report.py
```

原论文 PDF 仅用于本地来源审计，主复现可直接从冻结协议运行。[来源清单](sources/source_manifest.json)

## Publication note / 发布说明

Published from the independently verified research handoff. Local paths and internal execution logs were removed. Third-party manuscripts are linked in the source manifest and are not included. The published result data include computational evidence only.
