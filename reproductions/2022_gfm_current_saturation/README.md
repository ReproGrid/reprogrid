# Grid-forming converter current saturation and transient stability

Independent computational reproduction of the single-converter model in Rokrok et al., *IEEE Transactions on Power Systems*, 37(2), 1519–1531 (2022). [Publisher DOI](https://doi.org/10.1109/TPWRS.2021.3107959) · [HAL author manuscript](https://hal.science/hal-03966325v1)

The code checks the equal-area critical clearing times (CCTs) in Tables II–IV for a balanced 100% voltage sag, then includes the paper’s IP damping term in a lossless quasi-static power–angle model. It also evaluates the Fig. 9 saturated current-angle bound. The [protocol](configs/protocol.json) records the source, parameters and tolerances. The model and all results run without the paper PDF.

## Replay

Run from this directory with Python 3.12.

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
replay_dir=$(mktemp -d)
MPLBACKEND=Agg MPLCONFIGDIR="$replay_dir/mpl" .venv/bin/python src/reproduce.py --output "$replay_dir/results"
.venv/bin/python src/build_report.py --replay "$replay_dir/results"
```

The report script checks byte equality for four artifacts and regenerates the report and status on success. Reference hashes were generated with the pinned dependencies on macOS ARM64. Floating-point and image output can vary with platform libraries.

## Results and scope

- [Full report (中文)](reports/REPORT.md) · [All 18 CCT comparisons](results/cct_results.json)
- [CCT comparison figure](results/cct_comparison.png) · [Rated first-swing traces](results/rated_first_swing.png) · [Trajectory arrays](results/rated_trajectories.npz)
- [Replay check](results/replay_check.json) · [Evidence status](reports/status.json)

At rated power the reduced model gives 36.52 ms with a saturated current angle of 0 rad and 95.54 ms at 0.8 rad. The corresponding detailed simulations in the paper give 38 ms and 94 ms. At 38 ms the reduced model crosses the first-swing saddle angle even though the paper reports recovery. The first-swing model stops at angular turnaround or saddle-angle crossing. Reproducing voltage-mode reentry requires the inner voltage/current loops and anti-windup states. Partial sags, phase jumps, switching dynamics and laboratory waveforms remain outside the verified scope.

The [MIT license](LICENSE) covers the ReproGrid implementation. The paper is linked at its source.
