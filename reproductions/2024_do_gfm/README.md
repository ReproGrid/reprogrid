# DO-GFM and RFPSC reproduction

[计算复现报告](reports/REPORT.md) · [来源与参数审计](sources/source_audit.md)

## Independent equations and clean replay

The averaged implementation uses Eqs. (1), (9)–(20) and Appendix C. It compares DO-GFM and RFPSC in strong and weak grids, tests the linearized dynamics with inductance mismatch, and reconstructs the ideal Fig. 9 frequency ramp and Fig. 10 mode changes. The second implementation uses grid-frequency coordinates and the original observer with the current derivative. The two numerical paths share the [frozen protocol](configs/protocol.json).

From this paper directory, use Python 3.12. Keep temporary environments on the host filesystem; the result directories are relative to this T7 paper directory.

```sh
replay_dir=$(mktemp -d)
python3.12 -m venv "$replay_dir/primary"
"$replay_dir/primary/bin/python" -m pip install -r requirements.txt
OPENBLAS_NUM_THREADS=1 "$replay_dir/primary/bin/python" src/reproduce.py
OPENBLAS_NUM_THREADS=1 "$replay_dir/primary/bin/python" verification/verifier.py
python3.12 -m venv "$replay_dir/clean"
"$replay_dir/clean/bin/python" -m pip install -r requirements.txt
OPENBLAS_NUM_THREADS=1 "$replay_dir/clean/bin/python" src/reproduce.py --output results/clean_replay --figures results/clean_replay/figures
"$replay_dir/primary/bin/python" src/compare_replay.py
"$replay_dir/primary/bin/python" src/build_report.py
```

The report generator combines current runs with the recorded framework replay. Initial discrepancies are retained in the first run manifest, [initial coordinate comparison](results/cross_implementation_initial.json), and [solver diagnosis](verification/solver_diagnostic.json).

## Weak-grid motulator comparison

The experiment compares the two controllers in weak-grid Figures 5(b) and 6(b) of Nurminen et al., IEEE Transactions on Power Electronics, 2024. Both motulator runs use the same 12.5 kVA converter, 0.89 p.u. total inductance and active-power command sequence. The overlays use vector curves digitized from the CC BY 4.0 publication. [Paper DOI](https://doi.org/10.1109/TPEL.2024.3433503) · [Aalto open PDF](https://acris.aalto.fi/ws/portalfiles/portal/162149547/Multifunctional_Grid-Forming_Converter_Control_Based_on_a_Disturbance_Observer.pdf)

### Framework replay

From the workspace root, create a clean environment with Python 3.12. The replay checks the Git commit and tracked files before importing motulator.

```sh
run_dir=$(mktemp -d)
.venv/bin/python -m venv "$run_dir/env"
"$run_dir/env/bin/python" -m pip install -r reproductions/2024_do_gfm/requirements.txt
git clone --depth 1 --branch v0.23.1 https://github.com/Aalto-Electric-Drives/motulator.git "$run_dir/motulator"
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR="$run_dir/mpl" \
  "$run_dir/env/bin/python" reproductions/2024_do_gfm/reproduce.py \
  --checkout "$run_dir/motulator" --output "$run_dir/results"
```

Use `python3.12` in place of `.venv/bin/python` when running from a workspace without the project environment. The published trace CSVs are included. To regenerate them, use PyMuPDF 2.1 and NumPy 2.5.4 with the [digitization script](reference/extract_weak_grid_figures.py) and the Aalto PDF with the SHA-256 in the protocol. The script's `--help` documents its arguments.

### Framework results

- [Overlay of the published and simulated P, Q and voltage curves](results/fig5b_fig6b_comparison.png) · [SVG](results/fig5b_fig6b_comparison.svg)
- [Metrics, derived SI values and solver checks](results/metrics.json)
- [RFPSC switched CSV](results/fig5b_rfpsc_switched.csv) · [DO-GFM switched CSV](results/fig6b_do_gfm_switched.csv)
- [Frozen protocol](protocol.json) · [Paper digitization and clock offsets](reference/paper_metrics.json)
- [Clean-environment replay verification](results/replay_verification.json)

| Metric | RFPSC simulation / Fig. 5(b) | DO-GFM simulation / Fig. 6(b) |
| --- | ---: | ---: |
| 0.66 to 1 p.u. active-power rise time, ms | 61.0 / 89.7 | 67.0 / 109.6 |
| 1 to 0 p.u. active-power fall time, ms | 38.8 / 44.0 | 34.8 / 35.6 |
| Rated-command plateau active power, p.u. | 1.000 / 0.969 | 1.000 / 0.961 |
| Rated-command plateau reactive power, p.u. | −0.611 / −0.795 | −0.611 / −0.775 |

Times use the first crossings of 10% and 90% of the command step. Plateau values average the last 50 ms. The two simulated active-power curves differ by 0.00652 p.u. RMS over 0.05–1.45 s. Both the simulation and paper show a faster rated-power rise for RFPSC and a faster return to zero for DO-GFM. The quantitative gap is largest in the rated-power rise time and the reactive-power plateau. Solver refinement changed any plotted signal by at most 0.000055 p.u., and switching versus averaged modeling changed the Q trace by less than 0.00015 p.u. RMS. The assumed current limit never activated.

The simulation uses Table I's 6.3 mH filter, 650 V DC bus, 10 kHz control sampling and 5 kHz PWM. Table II gives the control gains. The observer inductance estimate is 0.15 p.u. The power command levels are 0, 0.33, 0.66, 1 and 0 p.u., with transitions at 0.2, 0.5, 0.8 and 1.1 s. Both controllers use the same grid-emulator terminal measurement of P and Q. The plotted and reported powers use a centered one-carrier-period average. The simulation CSVs also include power sampled at controller instants. Paper curves share the simulation's first command edge after a constant time translation, with no amplitude rescaling.

The model uses a balanced ideal grid source and fixed DC voltage. The plant series resistance is set to the reported 0.05 p.u. estimate. The assumed 1.3 p.u. current limit comes from Figure 8. The stock DO controller implements the Eq. (13) gains with a fixed-frame discrete observer, while the laboratory implementation uses Eq. (19). Stock RFPSC uses realized PWM voltage and estimated resistive losses in power feedback. The protocol records the resistance-compensation convention and the PWM assumptions. [motulator source at the pinned commit](https://github.com/Aalto-Electric-Drives/motulator/tree/ca3fb1f225be30ff98d70ebca6fc49cc24e2bb2e)

## Publication note / 发布说明

Published from the independently verified research handoff. Local paths and internal execution logs were removed. Third-party manuscripts are linked in the source manifest and are not included. The published result data include computational evidence only.
