# GFL grid-strength transient / 跟网逆变器网强切换

This independent computational study tests the single grid-following inverter in Fig. 8(d) of Li, Gu and Green, DOI 10.1109/TPWRS.2022.3151851. It uses its own three-phase network and sampled controllers. With a trace-inferred effective resistance of 0.01 Ω, all three tested switch-step methods pass. The pinned SLX value of 10⁻⁹ Ω fails both transient windows.

独立三相模型复算作者 Fig. 8(d) 的单机网强切换。等效电阻 0.01 Ω 来自三份作者存档的数值核对。使用该推断值时，三种切换步方法均通过；按固定 SLX 掩码 10⁻⁹ Ω，两个切换波形均未通过。

```sh
python -m pip install -r requirements.txt
OPENBLAS_NUM_THREADS=1 python src/replay.py --output-dir replay-output
```

Python 3.12. Inputs are fetched at a fixed author commit and checked against SHA-256. For offline execution supply the four author files with `--source-dir`. This archive contains software, an independently generated figure and aggregate comparisons. Source models and traces can be obtained directly from the author repository.

[Six-sweep report](reports/REPORT.md) · [Fig. 8(d) follow-up](reports/GFL_TRANSIENT.md) · [Results](results/results.json) · [Figure](figures/gfl_closed_loop.png)

To replay the six analytical sweeps and the separate recorded-input loop checks:

```sh
python src/fetch_baseline_inputs.py
OPENBLAS_NUM_THREADS=1 python src/reproduce.py --output six-case-output
```

The fetch script acquires hash-pinned MAT and PNG files from the authors. Locally generated overlays and raw result arrays that include author data remain local. The published archive contains aggregate errors, independent plots, and selected mathematical comparison plots.
