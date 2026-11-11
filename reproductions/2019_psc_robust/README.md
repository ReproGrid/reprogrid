# PSC 鲁棒解析整定复现

Harnefors、Hinkkanen、Riaz、Rahman 与 Zhang，*Robust Analytic Design of Power-Synchronization Control*，IEEE Transactions on Industrial Electronics 66(8)，5810–5819，2019。[DOI](https://doi.org/10.1109/TIE.2018.2874584) · [Aalto 作者稿](https://research.aalto.fi/en/publications/robust-analytic-design-of-power-synchronization-control/)

[复现报告](reports/REPORT.md) · [冻结协议](configs/protocol.json) · [图 5 参数对照协议](configs/fig5_variants.json) · [来源审计](sources/source_audit.md)

图 5 上排的图注工况与曲线反推工况分别保存。只改变无功电流符号时，直流侧灵敏度峰值最低的工况由 SCR=1 变为 SCR=3。两套工况均复现了极弱网高无功注入下的低直流侧相位裕度。[数值 CSV](results/fig5_condition_comparison.csv) · [Nyquist 对比](figures/fig5_parameter_overlay.png) · [裕度与灵敏度对比](figures/fig5_robustness_comparison.png)

从本目录运行。复跑使用 Python 3.12 和锁定依赖。

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python src/reproduce.py
.venv/bin/python src/compare_fig5_conditions.py
.venv/bin/python verification/verifier.py
.venv/bin/python verification/verify_fig8.py
```

`results/` 包含复数频响、两套工况的非线性小阶跃、附录审计、无功扫参和运行清单。`figures/` 包含重新计算的 Fig. 5、7、8 曲线及扩展工况。图 5 参数对照读取已保存的 PDF 数值摘要，主程序从冻结协议取值。[来源清单](sources/source_manifest.json) · [PDF 数值摘要](verification/fig5_vector_comparison.json)

独立环境重跑及报告生成。

```sh
python3.12 -m venv .venv-replay
.venv-replay/bin/python -m pip install -r requirements.txt
.venv-replay/bin/python src/reproduce.py --output results/clean_replay --figures results/clean_replay/figures
.venv-replay/bin/python src/compare_fig5_conditions.py --output results/clean_replay --figures results/clean_replay/figures
.venv-replay/bin/python verification/verify_fig8.py --primary results/clean_replay --output results/clean_replay/fig8_independent_comparison.json
.venv-replay/bin/python src/compare_replay.py
.venv/bin/python src/build_report.py
```

如需重新从作者稿提取图 5 数值，安装 PyMuPDF 2.1，再传入来源清单所记录的 PDF。脚本会检查文件的 SHA-256。

```sh
.venv/bin/python -m pip install pymupdf==2.1
.venv/bin/python verification/compare_fig5_manuscript.py --pdf ../../research/feasibility_batch1/sources/robust_psc2019.pdf
```

## Publication note / 发布说明

Published from the independently verified research handoff. Local paths and internal execution logs were removed. Third-party manuscripts are linked in the source manifest and are not included. The published result data include computational evidence only.
