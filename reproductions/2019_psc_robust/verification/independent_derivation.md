# Independent derivation frozen before numerical verification

Source: Harnefors et al., DOI 10.1109/TIE.2018.2874584, accepted manuscript, PDF pages 4–6 and 10. No primary implementation or other agent derivation has been read.

Use a grid-synchronous frame fixed at the equilibrium converter angle. Let J be complex current, Y the lowpass-filter current transformed into this fixed frame, and δ the converter-angle perturbation. Set g₀=V−jω₁Li₀ and v=Vexp(jδ)−Rₐ(J−Y). Then

- J̇=(v−g₀)/L−jω₁J,
- Ẏ=ω_b(J−Y)+jδ̇Y,
- P=κ Re(v J*), δ̇=Kp e.

The equilibrium is J=Y=i₀, δ=0. A physical open-loop Jacobian A,B,C gives Gp=C(sI−A)⁻¹B. Closing e=Pref−P gives Ac=A−BC and Gc=C(sI−Ac)⁻¹B. An independent DC energy-integrator state closes the outer loop via Pref=Kd W and Ẇ=−P. This coordinate choice avoids the primary dq transfer-polynomial route.

For ω_b=0, let r=Rₐ/L, a=ω₁Li_q0/V, b=−Rₐ²[i_q0/(ω₁L)+|i₀|²/V]/V, B₀=1+a+b and C₀=2+a. Eq14 gives Gc=r(a s²+B₀ω₁²)/[s³+C₀r s²+(ω₁²+r²)s+B₀ω₁²r]. Consequently, for Gd=Kd Gc/s, the denominator at s=jw has imaginary part r w[B₀ω₁²−C₀w²]. Its nonzero negative-real crossing is therefore wπ=ω₁ sqrt(B₀/C₀), provided the ratio is positive and the evaluated transfer is negative real. The printed Appendix IV frequency uses (1−b)/C₀ instead and is generally different.

At the actual crossing,

gm = [((1−b)/C₀)ω₁²/r+r]/(2Kd)
   = (1/Kd)[(1−b)ω₁²L/(2C₀Rₐ)+Rₐ/(2L)].

The printed Eq42 omits the factor 2 in its first denominator. The positive-term AM–GM inequality nevertheless gives gm ≥ (ω₁/Kd)sqrt((1−b)/C₀), exactly Eq43, and thus Eq24 still suffices within the stated valid domain. With a,b held fixed, equality requires L=(Rₐ/ω₁)sqrt(C₀/(1−b)); the printed L=sqrt(2)Rₐ/ω₁ is the a=b=0 special case. Since a and b depend on L when currents are held fixed, this equality condition is not automatically a minimizer over a fixed-current L sweep. The pointwise inequality does not require holding them fixed.

Initial protocol included auditing the printed Theorem1 gain-margin formula. The first numerical run exposed a second algebraic inconsistency, added to the audit with unchanged numerical tolerances in protocol_amendment_eq39.json. At ωπ=sqrt(ω₁²+r²), the correct expression is the FIRST expression of Eq39: 2[1+(r/ω₁)²]/[1+b−a(r/ω₁)²]. Eq10 implies b=−a(r/ω₁)²−(Rₐ|i₀|/V)², so the denominator becomes 1−2a(r/ω₁)²−(Rₐ|i₀|/V)². The second Eq39 expression and Eq40 incorrectly drop the reactive-current term. However, subtracting that correct denominator from 1+(r/ω₁)² gives (r/ω₁)²|V−jω₁Li₀|²/V², which is nonnegative. Thus the gain-margin bound of 2 survives for a valid negative-real crossing and positive denominator. A gain-margin formula alone is not proof of asymptotic stability for arbitrary operating points. Physical equilibria, phase margin, denominator signs and closed-loop poles must be assessed separately.

All finite-highpass examples use the exact Fig5 label currents (.95,.2) and (.2,−.95), not rescaled unit magnitudes. ω_b=.1 is exact for Fig5 and an explicit inherited assumption for Fig7. Table I's 4.4 Ω/.2 pu pair is inconsistent with its 12.6 Ω base; the analytic branch uses the explicit .2 pu design recommendation.
