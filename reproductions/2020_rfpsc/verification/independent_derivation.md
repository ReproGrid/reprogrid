# Independent RFPSC verification record

This record was prepared from the Aalto accepted author manuscript of DOI
10.1109/TPEL.2020.2970991, before inspecting either the main reproduction or any
author implementation. The implementation in this directory imports neither.
The verification scope is the paper's mathematical claims and an explicitly
idealized continuous average model, not replication of hardware measurements.

## Source observations and parameter provenance

The seven-page PDF includes two cover pages. Printed article page 1 is PDF page
3. Local PNGs `page-4.png`, `page-5.png`, and `page-6.png` were visually inspected
to check equations (9)-(20), Figure 3 panel labels, Table I, and the reference
steps in Figures 5 and 6. Text extraction was not the sole formula evidence.

- Equation (2): theta_dot = omega1 + Kp*(Pref-P).
- Equation (3): P = kappa*Re(v*conj(i)), with current/power positive OUT of the
  converter. For the paper's per-unit variables, kappa=1. Do not replace the
  instantaneous feedback with kappa*V*id; that expression is a steady-state
  simplification and removes relevant voltage/current perturbation terms.
- Equation (4): v = V + Ra*(iref-i), with constant real V.
- Conventional PSC: iref = H(s)*i, H(s)=omega_b/(s+omega_b).
- Equation (5), RFPSC: iref = Pref/(kappa*V) + j*H(s)*iq.
- Equation (18): Kp = omega1*Ra/(kappa*V^2).
- No independent symbol Rf is defined in the manuscript. If an implementation
  rewrites the d-axis reference feedforward as Rf*Pref/(kappa*V), it must set
  Rf=Ra to represent the paper. Rf is a local implementation parameter, not a
  separately reported paper parameter.
- Analysis: a purely inductive grid, ideal infinite bus, linear PWM range,
  switching harmonics omitted, and controller latency neglected. Claims about
  arbitrary grid R/X do not follow from this model.
- The analytic model freezes perturbations of H(s), justified in the paper by
  its low bandwidth. This means Delta z=0 about z0=i0, NOT z=0 at nonzero load.
- Figure 2 is a linearized feedback/feedforward block diagram. Figure 4 is an
  experimental setup schematic. Neither is a simulated response figure.
- Figure 3: V=omega1=1 pu, L=.5 pu, Ra=.2 pu, and (id0,iq0) equal to
  (0,0), (.7,.7), (-.7,.7), and (-.7,-.7).
- Figure 5: weak grid L=1 pu. Figure 6: strong grid L=.1 pu. These are
  EXPERIMENTAL measurements, not author-published simulation data.
- Figure 5/6 reference levels visually read from the dashed traces are
  0 -> .4 -> .8 -> 1 -> 0 pu at .2, .4, .6, .8 seconds. These values are
  plot-inferred, not an explicitly tabulated experiment protocol.
- Table I: 12.7 kVA, sqrt(2/3)*400 V voltage base, sqrt(2)*18.3 A current
  base, 12.6 ohm impedance base, 50 Hz fundamental, 8 kHz sampling, 4 kHz
  switching, and omega_b=31 rad/s (approximately .1 pu).
- Table I visually gives Ra=4.4 ohm AND Ra=.2 pu. 4.4/12.6=.34920635 pu,
  whereas .2*12.6=2.52 ohm. The inspected source is internally inconsistent
  on this conversion. Its cause and the actual hardware setting are unknown.
  Primary calculations use .2 pu, also stated in the body and Figure 3;
  the 4.4/12.6 branch is a separately labeled sensitivity calculation.

## Independent fixed-grid-frame model

Use delta=theta-omega1*t and let I=i*exp(j*delta) be current in the grid's
synchronous frame. The grid voltage Vg is real in this frame. Then

    u = [V + Ra*(iref-I*exp(-j*delta))]*exp(j*delta)
    dI/dt = (u-Vg)/L - j*omega1*I
    delta_dot = Kp*(Pref-P)
    P = kappa*Re(u*conj(I)).

Full-filter states satisfy zd_dot=omega_b*(id-zd) and
zq_dot=omega_b*(iq-zq), with id+j*iq=I*exp(-j*delta). Both states are retained
in the software for consistent dimensions. RFPSC does not use zd; its
unobservable filter mode must not be counted as an observable plant pole.

This frame keeps the plant's rotation rate fixed. An implementation using the
controller frame instead needs the term -j*(omega1+delta_dot)*i. Omitting the
delta_dot contribution gives a different nonlinear model.

Per-unit simulation time is tau=omega_base*t_seconds, omega_base=2*pi*50.
The code sets normalized omega1=1 and omega_b=.1. When converting inductance
to SI, L_SI=L_pu*Zbase/omega_base; L_pu is not the number of henries.
The normalized physical reactance is omega1*L, which equals L only because
normalized omega1=1.

## Equilibrium and initialization

At equilibrium both controller variants have v0=V and Pref0=kappa*V*id0.
The circuit gives

    Vg*exp(-j*delta0) = V-j*omega1*L*i0.

Therefore derive Vg=abs(V-j*omega1*L*i0) and
delta0=-arg(V-j*omega1*L*i0) when reproducing arbitrary Figure 3 operating
points. Setting Vg=V=1 for all those points generally violates the equilibrium
equation. The grid-frame initial current is I0=i0*exp(j*delta0), not i0.
Initialize full filter states to id0,iq0; initialize frozen states to the same
equilibrium values. The verification checks the actual RHS residual.

For nominal V=Vg=omega1=1 and the normal low-angle equilibrium,
id0=Pref/(kappa*V), sin(delta0)=L*id0, and
iq0=(cos(delta0)-V)/L. A real low-angle equilibrium requires |L*id0|<=1.
The weak-grid Pref=1 point is consequently at the ideal pure-L transfer
limit when Vg=V=1. Real experiment voltages and resistances may differ.
That edge case does not justify extrapolating small-perturbation conclusions
or the approximate first-order reduction to arbitrary large steps.

## Independent linearization

For frozen filter perturbations, define r=Pref, x=[delta, Re(I), Im(I)],
rho=exp(j*delta0), and xi=0 for PSC / 1 for RFPSC. Direct differentiation of
the physical grid-frame model gives

    Cdelta = kappa*V*iq0
    CI = kappa*(V-Ra*i0)*rho
    C = [Cdelta, Re(CI), Im(CI)]
    D = xi*Ra*id0/V
    u_delta = j*(V+Ra*i0)*rho
    u_r = xi*Ra/(kappa*V)*rho

    A[0,:] = -Kp*C
    A[1:,0] = [Re(u_delta), Im(u_delta)]/L
    A[1:,1:] = [[-Ra/L, omega1],[-omega1,-Ra/L]]
    B = [Kp*(1-D), Re(u_r)/L, Im(u_r)/L].

The resulting transfer is C*(sI-A)^(-1)*B+D. It is checked against the
separately transcribed paper polynomial, not built from that polynomial.
Centered finite differences of the nonlinear grid-frame model independently
check every A/B/C/D entry.

Let alpha=Ra/L, a=omega1*L*iq0/V,
b=-(Ra^2/V)*(iq0/(omega1*L)+|i0|^2/V), c=Ra*id0/V,
d=Ra^2*iq0/(omega1*L*V), and k0=(1+a+b)*alpha*omega1^2. Then the
paper's equation (19) is

    Gc(s) = [xi*c*s^3 + alpha*(a+xi*(1+c))*s^2
             + xi*((c+d)*omega1^2+alpha^2)*s + k0]
            /[s^3+(2+a)*alpha*s^2+(alpha^2+omega1^2)*s+k0].

It is not generally Gc(s)=alpha/(s+alpha). That expression is EXACT in the
linearized, frozen-filter model at i0=0. There the numerator and denominator
factor as

    numerator = alpha*(s^2+alpha*s+omega1^2)
    denominator = (s+alpha)*(s^2+alpha*s+omega1^2).

At nonzero current the paper gives only the approximation
(c*s+(1+a+b)*alpha)/(s+(1+a+b)*alpha). The feedthrough c is nonzero and
can be negative at reverse power, so a universal monotonic-step claim should
not be inferred. For L=.1, Ra=.2, omega1=1, the canceled pair is critically
damped (double real root at -1), despite generic wording about a complex pair.

The frozen-filter state matrix A does not depend on xi, confirming the
paper's pole-invariance statement WITHIN that approximation. With finite
filters, PSC retains the d-current filter in the loop while RFPSC replaces
that reference with feedforward. Full-filter state matrices need not share
the same poles; this is characterized rather than required to be equal.

There is a further independently derived zero-current invariant. At i0=0,
write the full-filter RFPSC linear grid-frame current as x+j*y and start from
zero perturbation. The subspace y=zq=0 and delta=(omega1*L/V)*x is invariant
under a power-reference input, because Kp=omega1*Ra/(kappa*V^2). On that
subspace, x_dot=(Ra/L)*(r/(kappa*V)-x). Thus the reference-to-power transfer
is exactly first order even with a finite q-filter in this special linear
case. Other internal modes remain; they are not excited/seen in this channel.
The numerical full-filter frequency checks corroborate this. Neither this
argument nor input-output pole cancellation guarantees equivalent responses
to grid disturbances or arbitrary internal initial-condition perturbations.

## Execution and interpretation

`protocol.json` was written before running `verifier.py`. The script checks:

1. Four Figure 3 equilibrium points, each controller: equilibrium residual,
   physical Jacobian versus finite differences, physical transfer versus
   equation (19), and xi-invariance of the frozen-filter matrix.
2. Exact rational symbolic factorization at i0=0 and numerical first-order
   response at L=.1,.5,1.
3. A 1e-5 pu nonlinear input perturbation against the independently obtained
   state-space response.
4. Full-filter piecewise sequences at L=.1,1 and Ra=.2,4.4/12.6, using
   DOP853 and Radau independently, with no current or state clipping.
5. Full-filter versus frozen-filter poles/frequency responses as an
   approximation-boundary diagnostic, not as a pass/fail assertion.

Raw samples, numeric metrics, environment versions, source/code/protocol
hashes, and plots are produced locally. The sequence is a qualitative trend
reconstruction. Missing measured grid traces, physical Ra, delay, sensing,
DC/PWM limits, and experimental raw data prevent quantitative reproduction
of Figures 5 and 6 as hardware experiments.

The first execution encountered an environment failure before calculations:
AppleDouble `._*` files on the external drive were interpreted as Matplotlib
style files. Only resource-fork metadata within this newly created verifier
venv was removed, then the same verifier was retried. This is recorded as an
environment repair, not evidence about the paper.

The initial numeric run passed all analytic/Jacobian checks but one of eight
full-filter solver comparisons exceeded its PREDECLARED 2e-7 pu maximum
power-error gate: L=.1, Ra=.2, RFPSC gave 4.2552e-7 pu. The failure is preserved
in `verification_results_initial.json`, `verifier_initial.py`, and
`sequence_initial_failed_solver_gate.npz`. Tightening both solvers' rtol/atol
100-fold reduced their maximum difference to 5.0341e-9 pu. The acceptance
threshold was not changed. `solver_refinement_diagnostic.json` records the
diagnosis. The final runner applies at most one such refinement when this
gate fails and records both attempts; this is a numerical-precision recovery,
not evidence of a problem in the paper and not an identified SciPy defect.
