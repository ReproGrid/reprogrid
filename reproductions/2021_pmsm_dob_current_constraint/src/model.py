"""Physical PMSM coordinates and equations printed in Section IV."""

from dataclasses import dataclass, replace
from math import pi, sin, ulp

import numpy as np
from scipy.integrate import solve_ivp


RPM = 30 / pi


@dataclass(frozen=True)
class Motor:
    pairs: int
    flux: float
    inertia: float
    friction: float
    resistance: float
    inductance: float
    reference: float
    ceiling: float

    @property
    def k(self):
        return self.pairs * self.flux

    @property
    def torque_at_ceiling(self):
        return self.k * self.ceiling - self.friction * self.reference


@dataclass(frozen=True)
class Controller:
    law: str
    k1: float
    k2: float
    p: float
    ki: float
    barrier_l: float


@dataclass(frozen=True)
class Segment:
    name: str
    start: float
    end: float
    torque_start: float
    slope: float = 0.0
    harmonics: tuple = ()

    def torque(self, t):
        ripple = sum(amplitude * sin(omega * t) for amplitude, omega in self.harmonics)
        return self.torque_start + self.slope * (t - self.start) + ripple


@dataclass(frozen=True)
class Case:
    name: str
    family: str
    poles: str
    controller_name: str
    motor: Motor
    controller: Controller
    segments: tuple
    voltage_limit: float | None
    speed_0: float
    iq_0: float
    auxiliary_0: float
    antiwindup: bool = False


@dataclass
class TrajectoryPiece:
    segment: Segment
    sol: object
    physical: object
    event_margins: tuple[float, float]


@dataclass
class Run:
    pieces: list[TrajectoryPiece]
    status: str
    message: str


def load_cases(protocol):
    table = protocol['table_i']
    test = protocol['numerical_test']
    initial = test['initial_conditions']
    controllers = {name: Controller(**data) for name, data in test['controllers'].items()}
    constants = tuple(Segment(name, *values) for name, values in zip(
        ('startup', 'load_step', 'second_step'), test['published_constant_segments_s_start_end_nm'], strict=True))
    sine = test['published_sinusoid']
    harmonic = Segment('sinusoid', *sine['interval_s'], sine['offset_nm'], harmonics=tuple(
        (amplitude, 2 * pi * hz) for amplitude, hz in zip(sine['amplitude_nm'], sine['frequency_hz'], strict=True)))
    ramp = test['unspecified_sawtooth']

    def load_profile(cycles):
        start, stop = ramp['interval_s']
        low, high = ramp['bounds_nm']
        corners = np.linspace(start, stop, cycles * 2 + 1)
        ramps = []
        for leg, (left, right) in enumerate(zip(corners[:-1], corners[1:], strict=True)):
            t0, t1 = (low, high) if leg % 2 == 0 else (high, low)
            ramps.append(Segment('sawtooth', float(left), float(right), t0, (t1 - t0) / (right - left)))
        return constants + tuple(ramps) + (harmonic,)

    cases = {}
    for label, pairs in table['pair_interpretations'].items():
        m = Motor(pairs, table['flux_wb'], table['inertia_kg_m2'], table['viscous_nm_s_per_rad'],
                  table['resistance_ohm'], table['inductance_h'], test['reference_rpm'] / RPM, test['iq_bound_a'])
        for name, controller in controllers.items():
            case = Case(f'{label}__{name}', 'main', label, name, m, controller, load_profile(1), test['uq_limit_v'],
                        initial['speed_rpm'] / RPM, initial['iq_a'], initial['observer_torque_estimate_nm']
                        if controller.p else initial['integral_speed_error_rad'])
            cases[case.name] = case

    def add(original, family, **changes):
        case = replace(original, name=f'{family}__{original.controller_name}', family=family, **changes)
        cases[case.name] = case

    nominal = {name: cases[f'equation_symbol_4__{name}'] for name in controllers}
    for name in ('dob_cc_eq16', 'dob_cc_eq17'):
        add(nominal[name], 'sawtooth_two_cycles', segments=load_profile(2))
        add(nominal[name], 'observer_initialization', auxiliary_0=constants[0].torque_start)
    authority = protocol['scenarios']['voltage_authority']
    for name in authority['equation_symbol_4_controllers']:
        for voltage_limit in authority['limits_v']:
            label = 'voltage_unlimited' if voltage_limit is None else f'voltage_{voltage_limit}v'
            horizon = tuple(part for part in constants if part.end <= authority['horizon_s'])
            add(nominal[name], label, voltage_limit=voltage_limit, segments=horizon)
    add(nominal['ccpid'], 'conditional_antiwindup', antiwindup=True)
    for name, case in nominal.items():
        if case.controller.p:
            m = case.motor
            iq = (harmonic.torque_start + m.friction * m.reference) / m.k
            add(case, 'sinusoid_from_equilibrium', segments=(harmonic,), speed_0=m.reference,
                iq_0=iq, auxiliary_0=harmonic.torque_start)
    return list(cases.values())


def apply_voltage(base, penalty, margin, ceiling, limit):
    """Evaluate the clipped quotient without dividing at a crossing probe."""
    if penalty == 0.0:
        if limit is None or -limit <= base <= limit:
            return base, 0
        return (limit, 1) if base > limit else (-limit, -1)
    denominator = max(0.0, margin * (2 * ceiling - margin))
    numerator = base * denominator + penalty
    if limit is not None:
        if numerator > limit * denominator:
            return limit, 1
        if numerator < -limit * denominator:
            return -limit, -1
    # Out-of-domain implicit solver probes are discarded by the terminal event.
    return numerator / (denominator or ulp(ceiling**2)), 0


def command(case, omega, margin, auxiliary):
    motor, ctrl = case.motor, case.controller
    k = motor.k
    drive = k * margin - motor.torque_at_ceiling + (auxiliary if ctrl.p else 0.0)
    base = (motor.resistance * (motor.ceiling - margin) + k * omega
            + (ctrl.k1 * (motor.reference - omega) + ctrl.k2 * drive + ctrl.ki * auxiliary) / k)
    penalty = ctrl.barrier_l * drive / k
    if ctrl.law in ('eq5', 'eq16'):
        penalty /= k**2
    return apply_voltage(base, penalty, margin, motor.ceiling, case.voltage_limit)


def primary(case, numerics):
    motor, ctrl = case.motor, case.controller
    k, c = motor.k, motor.ceiling
    wref, j, l, b, r = motor.reference, motor.inertia, motor.inductance, motor.friction, motor.resistance
    state = np.array([case.speed_0, c - case.iq_0, case.auxiliary_0])
    guard = numerics['boundary_margin_a']
    atol = [numerics['atol_other_states'], numerics['atol_margin'], numerics['atol_other_states']]
    if not ctrl.barrier_l:
        atol = numerics['plain_dobc_primary_atol']

    def physical(_times, states):
        return np.array([states[0], c - states[1], states[2]])

    def positive_edge(_t, y):
        return y[1] - guard

    def negative_edge(_t, y):
        return 2 * c - y[1] - guard

    for edge in (positive_edge, negative_edge):
        edge.terminal = True
        edge.direction = -1
    penalty_scale = ctrl.barrier_l / k
    if ctrl.law in ('eq5', 'eq16'):
        penalty_scale /= k**2

    def jac(_t, y):
        omega, margin, auxiliary = map(float, y)
        _, clipped = command(case, omega, margin, auxiliary)
        u_w = u_m = u_a = 0.0
        if clipped == 0:
            u_w, u_m = k - ctrl.k1 / k, -r + ctrl.k2
            u_a = (ctrl.k2 if ctrl.p else ctrl.ki) / k
            if ctrl.barrier_l:
                denominator = max(0.0, margin * (2 * c - margin)) or ulp(c**2)
                drive = k * margin - motor.torque_at_ceiling + (auxiliary if ctrl.p else 0.0)
                u_m += penalty_scale * (k - drive * 2 * (c - margin) / denominator) / denominator
                if ctrl.p:
                    u_a += penalty_scale / denominator
        if ctrl.p:
            auxiliary_gradient = [0.0, 0.0, -ctrl.p / j]
        else:
            auxiliary_gradient = [0.0 if case.antiwindup and clipped * (wref - omega) > 0 else -1.0, 0.0, 0.0]
        return [[-b / j, -k / j, 0.0], [(k - u_w) / l, (-r - u_m) / l, -u_a / l], auxiliary_gradient]

    pieces = []
    for segment in case.segments:
        # Local time also resolves sub-microsecond events far from t = 0.
        horizon = segment.end - segment.start

        def rhs(t, y):
            omega, margin, auxiliary = map(float, y)
            iq = c - margin
            torque = segment.torque(segment.start + t)
            voltage, clipped = command(case, omega, margin, auxiliary)
            rate = wref - omega
            if ctrl.p:
                rate = (ctrl.p / j) * (torque - auxiliary)
            elif case.antiwindup and clipped * rate > 0:
                rate = 0.0
            return [(k * iq - b * omega - torque) / j, (r * iq + k * omega - voltage) / l, rate]

        sol = solve_ivp(rhs, (0.0, horizon), state, method='Radau', jac=jac, dense_output=True, atol=atol,
                        events=(positive_edge, negative_edge) if ctrl.barrier_l else None,
                        max_step=numerics['max_step_s'], rtol=numerics['primary_rtol'])
        pieces.append(TrajectoryPiece(segment, sol, physical, (guard, 2 * c - guard)))
        if sol.status != 0:
            return Run(pieces, 'near_boundary' if sol.status == 1 else 'numerical_failure', sol.message)
        state = sol.y[:, -1]
    return Run(pieces, 'completed', '')
