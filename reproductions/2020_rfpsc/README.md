# Reference-Feedforward Power-Synchronization Control (2020)

Independent analytical and continuous average-value reproduction of [Harnefors et al., IEEE TPEL, DOI 10.1109/TPEL.2020.2970991](https://doi.org/10.1109/TPEL.2020.2970991).

**Evidence scope:** Eq. (19), zero-current pole–zero cancellation, Fig. 3 operating points, and PSC/RFPSC averaged step-response comparisons. The hardware waveforms in Figs. 5/6 have not been reproduced quantitatively. Source ambiguity and unavailable experimental inputs are retained in the report.

- [Computational report](reports/REPORT.md)
- [Frozen primary protocol](configs/protocol.json)
- [Primary run manifest](results/run_manifest.json)
- [Independent derivation](verification/independent_derivation.md)
- [Cross-implementation comparison](verification/cross_implementation.md)
- [Fresh-environment replay comparison](results/clean_replay_comparison.json)

## Run

Python 3.12 was used. From the workspace root:

```sh
python3.12 -m venv .venv-rfpsc
.venv-rfpsc/bin/python -m pip install -r reproductions/2020_rfpsc/requirements.txt
.venv-rfpsc/bin/python reproductions/2020_rfpsc/src/reproduce.py
```

The source PDF is not needed to run the equations. It is retained locally as research input and excluded from redistribution. The source manifest records its repository URL, version and hash.

To regenerate into a separate directory and compare numeric arrays:

```sh
.venv-rfpsc/bin/python reproductions/2020_rfpsc/src/reproduce.py --output reproductions/2020_rfpsc/results/clean_replay
.venv-rfpsc/bin/python reproductions/2020_rfpsc/src/compare_replay.py
```

The completed independent verifier uses its own pinned environment (`verification/environment_freeze.txt`), symbolic derivation and grid-synchronous coordinates. To repeat the comparison against the supplied independent outputs:

```sh
.venv-rfpsc/bin/python reproductions/2020_rfpsc/verification/compare_implementations.py
.venv-rfpsc/bin/python reproductions/2020_rfpsc/src/build_report.py
```

To regenerate the independent outputs themselves, create another environment from `verification/environment_freeze.txt`, retrieve the author manuscript from the `source_url` in `verification/protocol.json`, save it locally as `verification/rfpsc_2020_author_manuscript.pdf`, and run `verification/verifier.py`. The verifier hashes that local research input. The two repository copies used in this study have different PDF container/cover hashes but identical research pages; see `verification/source_comparison.md`. The separate verification environment is necessary because it includes SymPy and a separately pinned NumPy version.

The primary program uses converter-synchronous coordinates, including the changing-frame angular-speed term. Neither implementation imports author controller code. The contemporary motulator example was inspected as a separate reference; it is not the basis of the independent run.

## Artifacts and interpretation

`results/*.npz` retain numeric arrays; the corresponding CSV files are convenient exports. `figures/` contains independently generated plots. `run_manifest.json` records protocol, source and implementation hashes, dependency versions, checks and per-step metrics.

The two active-resistance branches are intentionally separate: the paper's reported **0.2 pu**, and **4.4 Ω / 12.6 Ω = 0.349206 pu**. Neither branch is silently selected as the true experimental value. All comparisons use identical plant parameters and tuning within a branch.

Metrics use the previous and next **reference** levels in the continuous four-step sequence; each step does not start from a separately reinitialized equilibrium. A missing settling-time value means the signal did not stay within the 2% band before the next step/end of the recorded window. Sampling resolution is 0.1 ms in the primary outputs.

## Execution issues retained

The first environment attempt encountered AppleDouble metadata inside matplotlib's style directory on the exFAT T7 volume. Only metadata files created inside the newly installed environments were removed after verifying their AppleDouble signature. No scientific parameter changed. See `results/execution_events.json` and `results/clean_environment_setup.json`.

One independent solver comparison initially failed its predeclared numerical tolerance. Its original code/data and the successful tolerance-refinement diagnostic remain in `verification/`. The acceptance threshold was not relaxed.

The original primary source and manifest are retained as `src/reproduce_initial.py` and `results/initial_run_manifest.json`. Review subsequently connected protocol parameters explicitly and enforced the declared equilibrium-residual threshold; current default values and scientific conclusions did not change.

## Publication note / 发布说明

Published from the independently verified research handoff. Local paths and internal execution logs were removed. Third-party manuscripts are linked in the source manifest and are not included. The published result data include computational evidence only.
