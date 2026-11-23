"""First-swing critical damping from the fault saddle's stable manifold."""

from math import asin, cos, pi, sin, sqrt

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


def saddle_gap(damping, sag_v, r_drop, starting_angle, nominal_v, epsilon):
    """Compare the initial speed with the saddle manifold in normalized time."""
    ratio = r_drop/sag_v
    saddle = -pi + asin(ratio)
    scale = 2*damping*sqrt(sag_v/nominal_v)
    cos_saddle = cos(saddle)
    trace, determinant = -scale*cos_saddle, cos_saddle
    stable_pole = (trace-sqrt(trace**2-4*determinant))/2
    initial_speed = -scale*(ratio+sin(starting_angle))

    def reverse(_, state):
        angle, speed = state
        return -speed, scale*cos(angle)*speed + ratio + sin(angle)

    def reach_initial(_, state):
        return state[0]-starting_angle

    def turn(_, state):
        return state[1]

    reach_initial.terminal = True
    reach_initial.direction = 1
    turn.terminal = True
    turn.direction = 1
    sol = solve_ivp(reverse, (0, 2500), (saddle+epsilon, stable_pole*epsilon),
                    rtol=4e-10, atol=4e-13, max_step=.3, method="DOP853", events=(reach_initial, turn))
    if not sol.success or sol.status != 1:
        raise RuntimeError(f"Saddle branch could not be classified at zeta={damping:g}")
    # A branch that already turned at a smaller angle does not intersect the
    # initial line with negative velocity. It is below the critical damping.
    if not len(sol.t_events[0]):
        return initial_speed
    return initial_speed - sol.y_events[0][0, 1]


def critical_damping(sag_v, r_drop, starting_angle, nominal_v, epsilon):
    def gap(damping):
        return saddle_gap(damping, sag_v, r_drop, starting_angle, nominal_v, epsilon)

    lower = 1e-5
    if gap(lower) >= 0:
        # This fault is already below the first-swing separatrix as damping
        # tends to zero. Positive damping is still needed for convergence.
        return 0.
    upper = 1.
    while gap(upper) <= 0:
        upper *= 2
        if upper > 128:
            raise RuntimeError("No positive stable-manifold gap in the damping search range")
    return brentq(gap, lower, upper, xtol=5e-9)


def forward_outcome(damping, sag_v, drop, angle0, nominal_v):
    ratio = drop / sag_v
    equilibrium = -asin(ratio)
    saddle = -pi - equilibrium
    gain = 2*damping*sqrt(sag_v/nominal_v)
    initial = (angle0, -gain*(ratio+sin(angle0)))

    def rhs(_, state):
        angle, speed = state
        return speed, -ratio - sin(angle) - gain*cos(angle)*speed

    def leave(_, state):
        return state[0]-saddle

    leave.terminal = True
    leave.direction = -1
    sol = solve_ivp(rhs, (0, 500), initial, rtol=2e-10, atol=2e-13,
                    events=leave, method="DOP853", max_step=.4)
    if not sol.success:
        raise RuntimeError(sol.message)
    if sol.status == 1:
        return "crosses_saddle"
    return "converges" if max(abs(sol.y[0,-1]-equilibrium), abs(sol.y[1,-1])) < 1e-5 else "unresolved"
