"""Locked-rotor current dynamics and pulse-pair measurements."""

from dataclasses import dataclass

import numpy as np

from observer import advance


SQRT_HALF = np.sqrt(.5)
PHASE_FROM_AB = np.array([[1., 0.], [-.5, np.sqrt(3) / 2], [-.5, -np.sqrt(3) / 2]])


@dataclass(frozen=True)
class Motor:
    resistance: float
    inductances: np.ndarray
    sample_period: float

    @property
    def saliency_factor(self):
        ld, lq = self.inductances
        return 1 - ld / lq


def fixed_rotor_vectors(angle_deg):
    degrees = np.asarray(angle_deg)
    radians = np.radians(degrees)
    vectors = np.column_stack((np.cos(radians), np.sin(radians)))
    cardinal = np.mod(degrees, 90.) == 0
    axes = np.array([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]])
    vectors[cardinal] = axes[np.mod(degrees[cardinal] / 90., 4).astype(int)]
    return vectors


def direction(angle):
    return np.column_stack((np.cos(angle), np.sin(angle)))


def into_frame(ab, vector):
    x, y = ab.T
    c, s = vector.T
    return np.column_stack((c * x + s * y, -s * x + c * y))


def out_of_frame(dq, vector):
    d, q = dq.T
    c, s = vector.T
    return np.column_stack((c * d - s * q, s * d + c * q))


def held_voltage(motor, current, dq_voltage, duration):
    # Each rotor-frame axis is a scalar RL circuit at standstill.
    interval = np.asarray(duration)[..., None]
    if motor.resistance == 0.:
        return current + interval * dq_voltage / motor.inductances
    ex = -motor.resistance * interval / motor.inductances
    return np.exp(ex) * current - np.expm1(ex) * dq_voltage / motor.resistance


def switching_interval(motor, current, ab_voltage, rotor, bus, rising):
    """Integrate an ideal, event-resolved half carrier with min/max SVPWM."""
    phases = ab_voltage @ PHASE_FROM_AB.T
    neutral = -.5 * (phases.max(axis=1) + phases.min(axis=1))
    duty = .5 + (phases + neutral[:, None]) / bus
    if np.any((duty < 0.) | (duty > 1.)):
        raise ValueError("The commanded voltage exceeds the linear PWM region.")
    edge_fraction = np.sort(duty if rising else 1. - duty, axis=1)
    edges = np.column_stack((np.zeros(len(duty)), edge_fraction, np.ones(len(duty))))
    for start, end in zip(edges[:, :-1].T, edges[:, 1:].T, strict=True):
        carrier = (start + end) / 2
        switches = (duty > (carrier if rising else 1. - carrier)[:, None]).astype(float)
        ab_switch = (2 * bus / 3) * switches @ PHASE_FROM_AB
        current = held_voltage(motor, current, into_frame(ab_switch, rotor),
                               (end - start) * motor.sample_period)
    return current


def measured_detector(i0, ip, im, estimated_direction):
    # Apply the same measurement frame (estimated d-axis lagging by pi/4)
    # to the three samples, as drawn in Fig. 4(b).
    c, s = estimated_direction.T
    measure_axis = SQRT_HALF * np.column_stack((c + s, s - c))
    dm, qm = into_frame(2 * ip - i0 - im, measure_axis).T
    norm = np.hypot(dm, qm)
    if np.any(norm == 0.):
        raise ValueError("Cannot normalize a zero pulse-response vector.")
    return (dm - qm) / norm


def printed_detector(angle_error, inductances):
    """Right-hand analytic expression printed in Eq. (15)."""
    ld, lq = inductances
    l0, l1 = (ld + lq) / 2, (ld - lq) / 2
    return ((lq - ld) * np.sin(2 * angle_error)
            / np.sqrt(l0**2 + l1**2 - 2 * l0 * l1 * np.cos(2 * angle_error)))


def pulse_pair(motor, current, rotor, estimated, voltage, group, bus=None):
    dt = motor.sample_period
    before = held_voltage(motor, current, 0., dt)
    ab_voltage = voltage * estimated
    if bus is None:
        dq_voltage = into_frame(ab_voltage, rotor)
        positive = held_voltage(motor, before, dq_voltage, dt)
        negative = held_voltage(motor, positive, -dq_voltage, dt)
    else:
        # Command schedule [+U, -U, 0] is delayed by one control tick.
        # Each pulse group spans three half carriers, so its parity alternates.
        positive = switching_interval(motor, before, ab_voltage, rotor, bus, rising=bool(group % 2))
        negative = switching_interval(motor, positive, -ab_voltage, rotor, bus, rising=not bool(group % 2))
    eps = measured_detector(out_of_frame(before, rotor), out_of_frame(positive, rotor),
                            out_of_frame(negative, rotor), estimated)
    return negative, eps


def run_puvi(motor, degrees, gain, settings, speed_seed, bus=None, detector_scale=1.):
    if settings["injection_vector_amplitude_v"] <= 0.:
        raise ValueError("The pulse amplitude must be positive.")
    rotor = fixed_rotor_vectors(degrees)
    state = np.zeros((len(degrees), len(gain)))
    state[:, 1] = speed_seed
    current = np.zeros_like(rotor)
    dt = 3 * motor.sample_period
    groups = round(settings["duration_s"] / dt)
    estimated = np.empty((groups + 1, len(degrees)))
    raw_error = np.empty((groups, len(degrees)))
    estimated[0] = state[:, 0]
    for group in range(groups):
        current, eps = pulse_pair(motor, current, rotor, direction(state[:, 0]),
                                  settings["injection_vector_amplitude_v"], group, bus)
        raw_error[group] = detector_scale * eps
        observer_error = raw_error[group] / (2 * motor.saliency_factor)
        state = advance(state, observer_error, gain, dt)
        estimated[group + 1] = state[:, 0]
    return np.arange(groups + 1) * dt, estimated, raw_error


def rtvi_steady_error(real_angle, estimated_angle, motor, voltage, omega):
    """Eqs. (3)–(8), verified by the independent sinusoidal dq RL response."""
    phase = np.arange(1024) * (2 * np.pi / 1024)
    ang = phase - real_angle
    ld, lq = motor.inductances
    i_d = np.real(voltage * np.exp(1j * ang) / (motor.resistance + 1j * omega * ld))
    i_q = np.real(-1j * voltage * np.exp(1j * ang) / (motor.resistance + 1j * omega * lq))
    c, s = np.cos(real_angle), np.sin(real_angle)
    alpha, beta = c * i_d - s * i_q, s * i_d + c * i_q
    demod = 2 * estimated_angle - phase
    first = np.mean(alpha * np.cos(demod) + beta * np.sin(demod))
    second = np.mean(alpha * np.sin(demod) - beta * np.cos(demod))
    return first / np.hypot(first, second)
