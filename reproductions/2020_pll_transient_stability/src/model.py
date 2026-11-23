"""Current-source VSC and continuous/discrete SRF-PLL equations of Wu and Wang."""

from dataclasses import dataclass
from math import asin, cos, exp, pi, sin

import numpy as np
from scipy.integrate import solve_ivp


PERIOD = 2*pi


@dataclass(frozen=True)
class Grid:
    nominal_hz: float
    resistance_pu: float
    line_x_pu: float
    voltage_prefault: float
    current_prefault: tuple
    current_fault: tuple

    @property
    def inductance_pu_s(self):
        return self.line_x_pu / (PERIOD*self.nominal_hz)

    @property
    def initial_angle(self):
        d, q = self.current_prefault
        return asin((d*self.line_x_pu + q*self.resistance_pu) / self.voltage_prefault)

    @property
    def fault_static(self):
        d, q = self.current_fault
        return d*self.line_x_pu + q*self.resistance_pu


@dataclass(frozen=True)
class Gains:
    kp: float
    ki: float

    @classmethod
    def from_damping(cls, nominal_voltage, damping, settling_time):
        kp = 9.2 / (nominal_voltage * settling_time)
        return cls(kp, kp**2 * nominal_voltage / (4*damping**2))


def equilibria(grid_voltage, static):
    ratio = static/grid_voltage
    if abs(ratio) > 1. + 1e-13:
        return {"count": 0, "stable_rad": None, "lower_separatrix_rad": None}
    principal = asin(np.clip(ratio, -1., 1.))
    if abs(abs(ratio) - 1.) <= 1e-13:
        return {"count": 1, "stable_rad": principal, "lower_separatrix_rad": principal}
    return {"count": 2, "stable_rad": principal, "lower_separatrix_rad": -pi-principal}


def detector_parameters(grid, voltage, current, angle, current_ramp=(0., 0.)):
    d, q = current
    dd, dq = current_ramp
    r, x, ell = grid.resistance_pu, grid.line_x_pu, grid.inductance_pu_s
    offset = d*x + q*r - voltage*sin(angle) + ell*dq
    direct = d*r - q*x + voltage*cos(angle) + ell*dd
    return direct, offset, d*ell, -q*ell


def pll_derivative(grid, voltage, current, angle, bias, kp, current_ramp=(0., 0.)):
    direct, offset, bq, bd = detector_parameters(grid, voltage, current, angle, current_ramp)
    mass = 1.-kp*bq
    if mass <= 0.:
        raise ValueError(f"The ideal current-source algebraic loop is singular or inverted ({mass:g})")
    return (kp*offset+bias)/mass, (direct, offset, bq, bd)


def phase_run(grid, gains, voltage, current, initial, horizon, cfg, representation="pi"):
    """Unbounded Fig. 13 equation via the PI state or the differentiated Eq. (11)."""
    kp, ki = gains.kp, gains.ki
    y0 = [initial, 0.]
    if representation == "swing":
        y0[1] = pll_derivative(grid, voltage, current, initial, 0, kp)[0]

    def rhs(_, state):
        angle, y = state
        if representation == "pi":
            velocity, (_, static, feedback, _) = pll_derivative(grid, voltage, current, angle, y, kp)
            return velocity, ki*(static+feedback*velocity)
        _, static, feedback, _ = detector_parameters(grid, voltage, current, angle)
        return y, (ki*(static+feedback*y)-kp*voltage*cos(angle)*y)/(1.-kp*feedback)

    def leave_plot(_, state):
        return state[0] + pi

    leave_plot.terminal = True
    leave_plot.direction = -1
    result = solve_ivp(rhs, (0, horizon), y0, method="DOP853" if representation == "pi" else "RK45",
                       dense_output=True, rtol=cfg["integral_state_ode_rtol"], atol=cfg["integral_state_ode_atol"],
                       max_step=.025, events=leave_plot)
    if not result.success:
        raise RuntimeError(result.message)
    step = cfg["phase_plot_interval_s"]
    time = np.r_[np.arange(0, result.t[-1]-1e-10, step), result.t[-1]]
    angle, second = result.sol(time)
    if representation == "pi":
        speed = np.array([pll_derivative(grid, voltage, current, a, x, kp)[0] for a, x in zip(angle, second, strict=True)])
    else:
        speed = second
    return np.column_stack([time, angle, speed/PERIOD]), result.status == 1


def marginal_p_solution(time, initial_angle, r_drop, kp):
    """At V=|I_q R|, cot((δ+π/2)/2) advances linearly with time."""
    phase_offset = initial_angle + pi/2
    return -pi/2 + 2*np.arctan2(1., 1./np.tan(phase_offset/2) + kp*r_drop*np.asarray(time))


@dataclass(frozen=True)
class Implementation:
    kind: str
    rocof_filter: str = "abs_before_derivative"
    threshold_units: str = "hz"
    integral_policy: str = "reset"
    antiwindup: bool = False
    current_ramp_s: float = 0.


SAMPLE_COLUMNS = ["time_s", "angle_rad", "frequency_hz", "raw_frequency_hz", "pcc_d_pu", "pcc_q_pu",
                  "threshold_signal_in_selected_units", "p_mode", "integrator_rad_s", "current_d_pu",
                  "current_q_pu", "grid_voltage_pu"]


def input_at(grid, time, duration, fault_v, ramp):
    if time < duration-1e-12:
        v = fault_v
        level, d_level = (time/ramp, 1./ramp) if (ramp and time < ramp-1e-12) else (1., 0.)
    else:
        v = grid.voltage_prefault
        dt = time-duration
        level, d_level = (1.-dt/ramp, -1./ramp) if (ramp and dt < ramp-1e-12) else (0., 0.)
    before, after = grid.current_prefault, grid.current_fault
    delta_d, delta_q = (after[0]-before[0], after[1]-before[1])
    current = before[0]+delta_d*level, before[1]+delta_q*level
    return (v, current, (delta_d*d_level, delta_q*d_level))


def sample_replay(grid, gains, fault_v, duration, post_s, dt, rocof, implementation):
    angle, integral, signal, previous = grid.initial_angle, 0., 0., 0.
    p_mode = implementation.kind == "p"
    carry = implementation.integral_policy == "hold_omit"
    nominal = grid.nominal_hz
    lower, upper = [PERIOD*(value - nominal) for value in rocof["output_frequency_limits_hz"]]
    mem = exp(-dt/rocof["lpf_s"])
    gain = (1.-mem) / (dt*(PERIOD if implementation.threshold_units == "hz" else 1.))
    samples = np.zeros((round((duration+post_s)/dt)+1, len(SAMPLE_COLUMNS)))
    events = []

    for k in range(len(samples)):
        t = k*dt
        voltage, current, ramp = input_at(grid, t, duration, fault_v, implementation.current_ramp_s)
        velocity, (direct, error, feedback, fb_d) = pll_derivative(
            grid, voltage, current, angle, 0. if p_mode else integral, gains.kp, ramp)
        # Fig. 12 takes Δω before the frequency limiter. The first implementation
        # follows the block ordering; the other follows a magnitude RoCoF filter.
        if implementation.rocof_filter == "abs_before_derivative":
            sample_difference = abs(velocity) - abs(previous)
        else:
            sample_difference = abs(velocity-previous)
        signal = mem*signal + gain*sample_difference
        previous = velocity
        magnitude = abs(signal)
        if implementation.kind == "adaptive":
            threshold = rocof["switch_off_hz_per_s" if p_mode else "switch_on_hz_per_s"]
            after_switch = magnitude >= threshold
            if after_switch != p_mode:
                p_mode = after_switch
                if p_mode and not carry:
                    integral = 0.
                events.append({"time_from_fault_s": t, "p_mode": p_mode, "phase_rad": angle,
                               "threshold_signal": magnitude})
                velocity = (gains.kp*error+(0. if p_mode else integral))/(1.-gains.kp*feedback)
        applied = 0. if implementation.kind == "freeze" else min(upper, max(lower, velocity))
        vq = error + feedback*applied
        pd = direct + fb_d*applied
        samples[k] = (t, angle, nominal+applied/PERIOD, nominal+velocity/PERIOD, pd, vq,
                      magnitude, p_mode, integral, *current, voltage)
        angle += applied*dt
        windup = (velocity > upper and vq > 0) or (velocity < lower and vq < 0)
        if not p_mode and implementation.kind != "freeze" and not (implementation.antiwindup and windup):
            integral += dt*gains.ki*vq
    return samples, events
