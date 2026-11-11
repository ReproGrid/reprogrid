# Independent mathematical and numerical verification

The independently constructed physical current/filter/angle state model reproduces Eqs9–12 to a maximum scaled complex error of 6.73e−15 across all six Fig5 operating conditions and 501 frequencies. The physical Jacobian also matches finite differences to 1.74e−10. All six inner and outer closed loops are asymptotically stable in this continuous averaged model. Fig7's claim that zero-inertia droop requires about 30% of the DC-controller gain to recover the proposed design's phase margin is corroborated.

These checks do not establish reproduction of the hardware experiments. No experimental raw traces, current-limit behavior, PWM, sampling delay, or actual DC-voltage transient were reproduced in this independent scope. Finite-inertia Fig8 was also outside the frozen verification scope.

## Independently verified numerical margins

Margins use exact Fig5 label currents and ωb=.1 pu. Frequencies are normalized to ω1; multiply by 2π50 for rad/s at 50 Hz. Negative-reactive current means reactive-current injection.

| Case | Inner GM factor | Inner PM ° | DC GM factor | DC PM ° |
|---|---:|---:|---:|---:|
| Active, SCR10 | 11.4081 | 45.3765 | 2.9940 | 50.3318 |
| Active, SCR3 | 2.7338 | 51.8947 | 3.2110 | 69.8464 |
| Active, SCR1 | 2.1015 | 81.9483 | 6.6311 | 55.9980 |
| Reactive, SCR10 | 5.8171 | 55.2767 | 4.6566 | 66.2610 |
| Reactive, SCR3 | 2.2507 | 61.5782 | 3.6090 | 62.9587 |
| Reactive, SCR1 | 2.0188 | 89.9939 | 12.2401 | 13.5118 |

Thus the inner gain-margin bound exceeds 2 in the six finite-filter examples. The finite-filter DC gain margins need not reach 4, as the paper already qualifies. The very weak grid with high reactive-current injection has poor DC-loop phase margin despite a large gain margin. Stability and usable transient performance must be reported separately.

For Fig7, taking ωb=.1 pu as an explicitly recorded inherited assumption, outer phase margins are 55.7717° for robust PSC, 28.6984° for zero-inertia droop, and 55.9502° after reducing the droop DC gain to 30%. This corroborates the phase-margin comparison; the consequent 3.3× physical capacitance assertion was not independently tested.

## Source algebra inconsistencies and their effect

All equations below were checked against the rendered original PDF page10, not only extracted text. The complete independent derivation is in independent_derivation.md; machine-readable counterexamples are in results.json.

1. **Appendix II, Eq39 second equality and Eq40:** the reactive-current term does not cancel as printed. Eq39's first expression matches the independent physical model. In the ωb=0, SCR1 reactive example, the printed final expression gives GM=2.16148810, whereas the physical model and corrected expression give 2.00327458. The bound GM≥2 nevertheless survives: the difference needed for that inequality equals a nonnegative quantity proportional to |V−jω1Li0|². This is an intermediate derivation error, not evidence that the principal robust bound is false.
2. **Appendix IV crossover frequency:** from Eq14, the actual nonzero crossover is ω1√[(1+a+b)/(2+a)], rather than the printed ω1√[(1−b)/(2+a)]. The reactive SCR1 example gives 0.218871565 pu instead of 0.975753677 pu. At the printed frequency the independent Gd has imaginary part 0.15132038, so it is not a real-axis intersection.
3. **Appendix IV Eq42:** the first term needs an additional factor 2 in its denominator. At a=b=0, Ra=.2, L=√2Ra and Kd=1/(4√2), independent numerical GM=4 and the corrected expression gives 4, but printed Eq42 gives 6.
4. **Appendix IV minimization:** for fixed a,b, equality requires L=(Ra/ω1)√[(2+a)/(1−b)]. The printed √2Ra/ω1 is the a=b=0 special case. Holding physical currents fixed makes a,b depend on L, so the equality point cannot be read as the general minimizer of that different sweep.
5. **Final Eq43 and Eq24:** the corrected Eq42 yields exactly the stated Eq43 lower bound by the positive-term AM–GM inequality. The Eq24 sufficient gain bound therefore remains supported within its applicable domain. Positive quantities, a valid negative-real crossing and closed-loop poles are checked separately rather than inferred from a margin formula alone.

The initial run exposed the Eq39 discrepancy. Its output is preserved in results_initial_before_eq39_audit.json, and protocol_amendment_eq39.json records the added diagnostic without relaxing any acceptance threshold. The amended run verifies corrected crossover/gain-margin expressions against numerical crossings below 1e−8 relative tolerance.

## Operating limits and source completeness

A 3,150-point frozen-filter scan covers SCR1–10, current magnitude 0–.999 pu, and 25 current angles. All scanned inner loops and DC loops at the pointwise Eq24 gain bound are stable. This finite scan is corroboration, not a proof for all grids, depressed voltages, current-limit modes or arbitrary operating points.

At zero current, the frozen-filter factorization (s+r)(s²+rs+1) is exact, with r=Ra/L and ζ=.1SCR for Ra=.2. The pair becomes critically damped at SCR10 and real beyond it; the strong-grid complex-pole interpretation has a domain limit. The large-Ra/high-active-current and high-reactive-current sweeps both show a pole approaching the origin, supporting the stated bandwidth-degradation mechanism. The approximate pole expression can lose accuracy near these limits, as the paper itself notes.

The Fig5 current labels (.95,.2) and (.2,−.95) have magnitude 0.97082439, despite prose describing full current of 1 pu. This verification uses the labels verbatim. Table I lists Ra=4.4 Ω=.2 pu with Zbase=12.6 Ω; 4.4/12.6=.34920635, so these entries are inconsistent. The analytic baseline uses the explicit .2 pu design recommendation. Neither source issue justifies an allegation of fabrication.

## Reproduction and files

Run from the repository root:

```sh
.venv/bin/python reproductions/2019_psc_robust/verification/verifier.py
```

The script regenerates results.json, frequency_responses.npz and fig5_independent.png. frequency_responses.npz contains w_pu=np.geomspace(1e−3,1e2,501), the 18 keys fig5_{active|reactive}_scr{1|3|10}_{gp|gc|gd}, and nine Fig7 keys. Frequency response is computed from the physical-state resolvent; the equations in the paper serve as comparison targets, not the primary implementation. Root implementation and other agent derivations were not read during this independent phase.

Stage timestamps, measured wall time and inherited model-setting visibility are recorded in stage_record.json. The numerical script separately records its own wall time in results.json.
