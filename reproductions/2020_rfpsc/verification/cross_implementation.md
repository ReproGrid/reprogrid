# Cross-implementation verification

Two independently written models were compared: primary converter-dq frame and verifier fixed-grid frame. The independent derivation and outputs were completed before reading the primary source.

The primary grid is 100 microseconds (10,001 points), and the verifier grid is 25 microseconds (40,001 points). Every fourth verifier sample is selected. Every timestamp is checked within 1e-12 seconds; no signal interpolation is used.

Both outputs use post-step references at 0.2, 0.4, 0.6, and 0.8 seconds while the physical states remain continuous. Full reference vectors and all four event samples are checked. Algebraic power may jump for RFPSC at nonzero current; those event samples are included, not discarded.

The power threshold of 2e-7 pu was recorded in cross_implementation_protocol.json before computing paired differences; it matches the independent solver gate.

| Case | Max power difference (pu) | Time of maximum (s) | Event max (pu) | Passed |
|---|---:|---:|---:|---|
| paper_pu_L1_psc | 3.68435e-10 | 0.661400 | 1.48317e-10 | True |
| paper_pu_L1_rfpsc | 2.54226e-09 | 0.435100 | 9.93248e-11 | True |
| paper_pu_L0.1_psc | 1.36443e-08 | 0.280100 | 1.07592e-12 | True |
| paper_pu_L0.1_rfpsc | 5.03409e-09 | 0.724400 | 6.29163e-13 | True |
| table_ohm_L1_psc | 7.59824e-10 | 0.818200 | 2.68359e-11 | True |
| table_ohm_L1_rfpsc | 1.42694e-09 | 0.818000 | 5.47385e-11 | True |
| table_ohm_L0.1_psc | 3.20524e-08 | 0.236100 | 3.08422e-11 | True |
| table_ohm_L0.1_rfpsc | 3.94117e-09 | 0.648100 | 3.31946e-12 | True |

All eight cases passed: **True**. Maximum power difference: **3.20524e-08 pu**.

Raw file hashes and numeric-array hashes are recorded in cross_implementation.json. The generating code hash from the primary run manifest is retained separately from the current inspected source hash, because source edits may occur after the outputs were generated. Re-run this comparison after a clean replay to attach the new output/code version.

This establishes implementation agreement for the stated model and protocol. It does not resolve Table I ambiguity, supply missing experimental data, or establish hardware validation.
