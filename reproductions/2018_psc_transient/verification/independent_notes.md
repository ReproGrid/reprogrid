# Independent verification: PSC transient stability

Paper: **Design-Oriented Transient Stability Analysis of Grid-Connected
Converters with Power Synchronization Control**, DOI
`10.1109/TIE.2018.2875669`. The source is the locally supplied eleven-page
accepted manuscript. No primary implementation or other agent's derivation was
read while deriving, freezing, or executing this verifier.

## Source and model audit

The cover is PDF page 1. Equations (2)-(7) and Figures 1, 3, 4, 5 define the
physical model. PDF pages 3-10 were rendered and the relevant equations,
topology, tables, plots and appendix were visually checked.

The regulated PCC is **after** the converter output filter Xf and **before**
the transformer XT. Thus Xf is excluded from the reduced PCC-to-grid transfer
reactance. Under the paper's assumptions, the inner voltage loop is replaced
by instantaneous fixed-magnitude PCC voltage tracking. Constant DC voltage,
pure inductive lines and continued PSC operation are assumed. Current-limited
vector control/PLL dynamics are explicitly outside this scalar model.

For phase peak voltage magnitudes, the physical equations are

    P = 3*Vpcc_peak*Vg_peak/(2*Xg) * sin(delta)
    delta_dot = Ki*(Pref-P) = a-b*sin(delta)
    a = Ki*Pref; b = Ki*Pmax.

Time is in **seconds**, not electrical angle time. Table I gives
Ki=9.3e-9 and Pref=1e9, hence a=9.3 rad/s; Table III gives .0093*1000=9.3
as well. The manuscript's additional Ki=.01 pu annotation does not specify
its normalization sufficiently to replace a by .01*2*pi*50=3.14159. The
dimensional gain-product is used without tuning.

The simulations are single-converter/infinite-bus reduced-model studies;
their large power rating does not introduce a large-network model. No relay,
breaker timing design, protection logic or hardware implementation is made.

## Topology and fault reduction

The source's Figure 1 places the ground branch on line 2 after S3 and before
Xg2, at the line sending end. For the stated drawing, both transmission lines
remain in service during the disturbance, and their common node after XT is
connected to ground through C=Xgnd. Define B=Xg1*Xg2/(Xg1+Xg2).

    Before: E=1, X=XT+B
    During: E=C/(B+C), X=XT+B*C/(B+C)
    After line-2 removal: E=1, X=XT+Xg1
    Pmax_pu=E/X.

Equivalently the during-disturbance transfer reactance is
XT+B+XT*B/C. A complex nodal KCL calculation is independently checked against
this Thevenin reduction for 101 angles. Treating the ground point as a midpoint
of Xg2 would be a different network and is not silently substituted here.

For Type I, Table I's primary pu entries are XT=.02, Xg1=Xg2=.85. Before
line removal X=.445; afterwards X=.87. The initial angle is asin(.445) and
the new stable angle is asin(.87).

For Type II, the primary pu entries are XT=.8, Xg1=.15, Xg2=.8, Xgnd=.5.
The pre-disturbance X=.9263157895. During the disturbance,
Pmax=.8861940299<Pref=1; hence no equilibrium exists. After line-2 removal,
X=.95 and the first unstable equilibrium is pi-asin(.95).

The disturbance starts at t=3 s in Figures 9 and 10 (read from the plotted
axes). Durations .5 and .7 s are explicitly stated in the Figure 10 caption
and text, so clearing times are 3.5 and 3.7 s. Figure 9 is Type I; Figure 10
is Type II. Figures 12/13 are experimental plots, whose trigger origins and
raw samples are not reconstructed. Topology/power/current are right-continuous
at each event; angle remains continuous.

## Independent mathematical path

The primary verification trajectory is **not obtained by an ODE integrator**.
Using z=tan(delta/2), the scalar equation becomes the Riccati equation

    z_dot = (a/2)*z^2-b*z+a/2.

Set z=u/v and M=[[-b/2,a/2],[-a/2,b/2]]. Then [u,v]'=M[u,v] and
M^2=(b^2-a^2)*I/4. Its exponential is evaluated by a real hyperbolic form
when b>a, a real trigonometric form when a>b, and I+tM at equality. Recover
delta from twice the continuously unwrapped atan2(u,v). This avoids principal
arctangent jumps and handles both Type I and Type II in real arithmetic.

The paper's real arctangent expression in (9)/(10)/(12) requires a^2>b^2
as written and branch care. It is appropriate for the tested no-equilibrium
Type-II CCT interval, whose endpoints are below pi. It is not a universal
real-valued expression for Type I, where b>a. The projective form supplies
the needed alternative representation; this is an implementation/domain
qualification rather than a failed Type-I phase-portrait claim.

The critical time is independently evaluated three ways:

1. Direct quadrature of 1/(a-b*sin(delta)) from the initial angle to the
   post-disturbance UEP.
2. The paper's arctangent primitive on its valid branch.
3. Brent root finding on the exact projective trajectory.

The trajectory is separately checked against DOP853 scalar ODE integration.
This ODE run is a checker, not the source of the independent trajectory.

## Verified results and interpretation

With the stated pu parameters and physical gain-product a=9.3:

| Quantity | Independent value | Paper's stated value |
|---|---:|---:|
| Type-I initial stable angle | 26.4233387659 deg | 26.4 deg |
| Type-I final stable angle | 60.4586394999 deg | 60.5 deg |
| Type-II critical angle | 108.1948723388 deg | 108 deg |
| Type-II critical time | 0.580269816566 s | 0.58 s |
| No-clearance phase-slip period | 1.45821383450 s | Not numerically tabulated |

Type I converges monotonically without overshoot in this first-order model.
For Type II, .5 s clearance gives delta=102.0150831642 deg, remaining in
basin 0; .7 s gives delta=120.3957000335 deg, entering basin 1. The latter
approaches the next stable equilibrium shifted by 2*pi. The uncleared case
has monotonically unbounded **unwrapped** angle. Its plotted wrapped angle is
a sawtooth; it is not a back-and-forth oscillation of the scalar state.

The UEP is the boundary of same-well synchronization, not a boundary beyond
which this ideal periodic first-order equation necessarily diverges forever.
At exact UEP equality the state is an unstable equilibrium; arbitrarily small
perturbations select opposite basins. Initial points must be classified by
their basin, not by Euclidean distance to the geometrically nearest stable
point. The CCT +/-1 ms tests select the predicted two basins.

CCA is determined by the post-disturbance circuit. CCT also depends on the
initial state and during-disturbance dynamics. Keeping the post-circuit fixed,
Xgnd=.3,.5,.8 gives CCT=.3378357463,.5802698166,1.1502592297 s, while CCA is
unchanged. This qualifies any reading of “fixed CCT” as independent of the
disturbance severity; equation (12) itself contains those dependencies.

## Current-limit applicability finding

The paper states a 1.8 pu current limit and confines the PSC argument to
disturbances that do not trigger a switch to vector current control. Under
consistent standard power/phase-voltage normalization, the reduced network
current is Ipu=abs(exp(j*delta)-E)/X. Auditing the ideal scalar trajectories:

| Primary-pu case | First I >= 1.8 pu after disturbance | Scalar trajectory max I |
|---|---:|---:|
| Clear after .5 s | No crossing in the modeled interval | 1.63627 pu |
| Clear after .7 s | .7 s, at the topology/output jump | 2.10526 pu |
| Never clear | .756149373367 s | 1.99627 pu |

The late-clearance recovery therefore remains a valid **formal scalar-model
result**, but the continued no-limiter PSC model is outside its stated
applicability region under this current normalization. This verifier does not
claim that the actual device switched at those times or failed to recover.
The paper does not separately tabulate the current-base/limiter signal
definition or supply original firmware and raw data. Those missing details,
along with inner-loop dynamics, must be resolved before treating this as a
contradiction of the complete simulation or hardware experiment. No limiter
has been invented or inserted to hide the scope exit.

## SI/table ambiguity and sensitivity branches

The 3/2 power expression uses **phase peaks**. Interpreting the tables' voltage
entries as phase RMS and converting by sqrt(2) approximately reconciles the
listed inductance bases. This is explicitly an inference: the tables do not
state the RMS/peak and phase/line voltage convention. With that interpretation,
Zbase=3*Vphase_rms^2/Pref and Lbase=Zbase/(2*pi*50).

Using the actual SI entries rather than their rounded pu annotations gives:

| Branch | Type-I initial/final angle | Type-II CCA | Type-II CCT |
|---|---|---:|---:|
| Stated pu (primary) | 26.4233 / 60.4586 deg | 108.1949 deg | .580270 s |
| Table I SI, inferred phase RMS | 26.3302 / 59.9344 deg | 107.8247 deg | .558672 s |
| Table III SI, inferred phase RMS | 27.4366 / 64.2357 deg | 112.8486 deg | .974583 s |

The third branch would recapture without a slip at .7 s. This is **not a
replacement measurement or definitive corrected experimental result**. It
shows that the tabulated physical entries, pu annotations and unspecified
voltage/current normalization do not uniquely specify one reproducible SI
experiment, particularly near a stability boundary. The reported main
results agree with the explicitly stated pu parameter branch. No parameters
were tuned to force agreement, and the SI branches remain separately labeled.

## Files, checks and execution record

`protocol.json` was frozen before numeric execution. `verifier.py` is
self-contained and runnable using the existing workspace Python environment:

    .venv/bin/python reproductions/2018_psc_transient/verification/verifier.py

`results.json` contains stable scalar keys under `branches.stated_pu`,
`branches.table_SI_simulation`, and `branches.table_SI_experiment`, plus
provenance hashes and numeric checks. All internal checks and primary reported
numeric targets passed. Sensitivity-branch classifications are checked against
their own CCT, not forced to match primary classifications.

`trajectories.npz` provides the two principal branches on exactly
round(linspace(2,7,5001),12), with time key `time_s`, and keys
`{branch}_{scenario}_{field}`. Scenarios are `type_i_line_open`,
`type_ii_clear_0p5`, `type_ii_clear_0p7`, `type_ii_uncleared`; fields are
`angle_rad`, `power_pu`, `current_pu`. The grid is selected directly from this
verifier's analytic samples, with no interpolation. It is ready for an
independently implemented solver comparison.

The initial execution stopped on a floating-point t_eval endpoint
(.7000000000000001 exceeding .7), before any scientific verdict. Clipping
t_eval to the exact segment limits repaired the checker. Gates and physical
event times were unchanged. This is recorded in `stage_record.json`.

`stage_record.json` records UTC start/end and measured UTC wall-clock
elapsed time for this agent's whole stage. `results.json` separately records
numeric-script runtime. Parent settings were inherited; reasoning strength
and the precise model identifier were not exposed and are not estimated.
