# Single-inverter grid-strength change

The 100 µs closed-loop model solves one four-state electrical network per phase and feeds independent capacitor voltage back to a power-invariant Park transform and sampled PLL. A forward-Euler current PI drives the network through an ABC unit delay. The input filter, PLL integral and phase integral retain their sampled state updates. Initial states come from a periodic trapezoidal phasor solution. Each phase opens at its predicted first breaker-current zero after the 2.0 s command. The 2.2 s close command changes the topology over the following interval and first affects the output at 2.2001 s.

At the pre-event operating point, capacitor d-axis voltage differs by about 8.70e-7 pu. All configurations pass the steady operating-point checks. The comparison includes 6,001 timestamps from 1.8 to 2.4 s and evaluates each event separately without shifting or rescaling simulated waveforms.

| Configuration | Opening peak frequency error | Reclosing peak frequency error | Event checks |
| --- | ---: | ---: | --- |
| Pinned SLX mask, 10⁻⁹ Ω | 0.3581 Hz | 0.4013 Hz | Both exceed tolerances |
| Inferred 0.01 Ω, trapezoid | 0.0904 Hz | 0.0572 Hz | Both pass |
| Inferred 0.01 Ω, backward Euler at switch | 0.0834 Hz | 0.0642 Hz | Both pass |
| Inferred 0.01 Ω, two half backward Euler steps at switch | 0.0943 Hz | 0.0535 Hz | Both pass |

The waveform limits are 0.01 pu RMS and 0.02 pu peak for voltage and current, 0.05 Hz RMS and 0.15 Hz peak for frequency, and 0.02 rad peak for wrapped phase. The limits are study criteria. Every inferred-resistance case passes all channels and both windows. Aggregate measurements are in the result JSON.

The model file specifies 10⁻⁹ Ω for the grid-strength branch and 0.01 Ω for both gain-change branches. The other two traces therefore corroborate the source values of their own branches. The continuous analytical script separately specifies 0.005 pu. A sampled filter-inductor KVL audit of the archived grid-strength, PLL-gain and current-loop-gain cases identifies an effective series resistance within 10⁻⁸ Ω of 0.01 Ω in each. Identification intervals and held-out check intervals are recorded for each archive. The current-controller proportional and integral gains are scaled by 0.6 and 0.6² when specified in the current-loop-gain case. The network simulation uses no archived voltage or current as an input. The three saved traces carry April 18, 2021 timestamps, while the published R2020b model records a modification on April 21, 2021. Public Git history first adds the files together. This leaves the exact model revision behind the grid-strength trace unresolved.

The public model scope is the GFL single-inverter grid-strength transient. The additional archives provide KVL corroboration; full gain-sweep trajectories, the proprietary switch-step solver, switching semiconductor behavior, and hardware are outside this validation. Fig. 8 source and use of inferred data are explicitly distinguished. The figure presents independent traces and summary errors, without redistributing the author's data.

Sources: [Paper](https://arxiv.org/abs/2105.13094v3) · [Pinned Case 2 archive](https://github.com/Future-Power-Networks/Simplus-Grid-Tool/tree/8f73752332d6c2144570ac101e556f1859c06930/SingleInverterCase_Simulation/Case2_SingleInverterInfiniteBus) · [Integrator initialization](https://www.mathworks.com/help/simulink/slref/discretetimeintegrator.html) · [Switch-step critical damping method](https://www.ipstconf.org/papers/Proc_IPST2019/19IPST078.pdf)
