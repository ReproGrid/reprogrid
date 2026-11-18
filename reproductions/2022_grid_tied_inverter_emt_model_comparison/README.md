# Grid-tied inverter EMT voltage interpolation

An independent implementation of Eq. (1) from Sano, Horiuchi and Noda. The three-phase reconstruction uses the circuit of their cited CIRED 2019 study. The original Table II and XTAP closed-loop cases remain pending. [TPEL paper](https://doi.org/10.1109/TPEL.2021.3117633) · [Source audit](sources/source_audit.md) · [Report (中文)](reports/REPORT.md)

## Replay

Use Python 3.12 and run the following commands from this directory. All inputs needed to run the model are provided in the ZIP.

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r verification/environment_freeze.txt
MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 python src/reproduce.py --output replay/results --figures replay/figures
```

The [run manifest](results/run_manifest.json) lists the SHA-256 values of the reference outputs. The [replay comparison](results/clean_replay_comparison.json) records the same-stack check. Results are portable as numeric data. Byte equality can depend on the numerical libraries and plotting environment.

- [Protocol](configs/protocol.json) · [Derivation](verification/derivation.md)
- [Metrics](results/metrics.json) · [Waveforms](results/waveforms.npz)
- [Carrier comparison](figures/carrier_comparison.png) · [Solver and phase sweep](figures/step_phase_sensitivity.png) · [Network and grid fifth harmonic](figures/network_and_grid_fifth.png)

Waveform archive keys identify each case, model and signal. Units are seconds, volts and amperes, and the phase order is A, B, C. Event arrays represent the exact switching reference. The [MIT license](LICENSE) applies to ReproGrid's implementation.
