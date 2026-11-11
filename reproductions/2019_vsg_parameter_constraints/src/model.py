"""Published transfer functions and margins in a normalized Laplace variable."""

from dataclasses import dataclass, replace

import numpy as np
from numpy.polynomial import polynomial as poly
from scipy.optimize import brentq


@dataclass(frozen=True)
class Model:
    omega: float
    voltage: float
    grid_resistance: float
    grid_inductance: float
    virtual_resistance: float
    virtual_inductance: float
    inertia: float
    damping: float
    capacitance: float
    voltage_kp: float
    voltage_ki: float
    current_tau: float

    @property
    def resistance(self):
        return self.grid_resistance + self.virtual_resistance

    @property
    def inductance(self):
        return self.grid_inductance + self.virtual_inductance


def published_model(common, case):
    return Model(
        2 * np.pi * common["grid_frequency_hz"], common["internal_and_grid_phase_peak_v"],
        common["grid_resistance_ohm"], common["grid_inductance_h"],
        case["virtual_r_ohm"], case["virtual_l_h"], common["inertia"], case["damping"],
        common["filter_capacitance_f"], common["voltage_kp"], common["voltage_ki"],
        common["filter_inductance_h"] / common["current_kp"],
    )


def printed_line(model, angle=0.):
    """Eq. (3), using s = omega_grid * q and E0 = Ug0."""
    e = ug = model.voltage
    r, l = model.resistance, model.inductance
    x = model.omega * l
    z2 = r**2 + x**2
    r_g, l_g = model.grid_resistance, model.grid_inductance
    r_v, l_v = model.virtual_resistance, model.virtual_inductance
    sine, cosine = np.sin(angle), np.cos(angle)
    common = r * sine + x * cosine
    difference = r * sine - x * cosine
    a1 = ug * l * l_g * common - e * l * x * l_g
    a2 = (2 * ug * l * r_g * common - 2 * e * l * x * r_g
          - ug * l_v * sine * (r**2 - x**2) + 2 * e * l_v * x * sine * difference)
    a3 = ug * z2 * common - 2 * ug * r_v * r**2 * sine + 2 * e * r_v * x * sine * difference
    return (1.5 * e * np.array([a3, a2 * model.omega, a1 * model.omega**2]) / z2,
            poly.polytrim([z2, 2 * r * x, x**2]))


def transfer_functions(model):
    w = model.omega
    swing = (np.array([1.]), np.array([0., model.damping * w, model.inertia * w**2]))
    converter = (
        np.array([model.voltage_ki, model.voltage_kp * w]),
        np.array([model.voltage_ki, model.voltage_kp * w, model.capacitance * w**2,
                  model.capacitance * model.current_tau * w**3]),
    )
    line = printed_line(model)
    full = (poly.polymul(converter[0], line[0]),
            poly.polymul(swing[1], poly.polymul(converter[1], line[1])))
    reduced = (np.array([line[0][0] / line[1][0]]), swing[1])
    return {"swing": swing, "converter": converter, "line": line, "full": full, "reduced": reduced}


def evaluate(pair, s, omega):
    numerator, denominator = pair
    return poly.polyval(s / omega, numerator) / poly.polyval(s / omega, denominator)


def close_loop(pair):
    numerator, denominator = pair
    return numerator, poly.polyadd(denominator, numerator)


def closed_poles(pair, omega):
    return poly.polyroots(close_loop(pair)[1]) * omega


def reduced_crossover(gain, model):
    if gain == 0.:
        return None
    # Rationalized Eq. (7) to avoid cancellation for large damping.
    d, m = model.damping, model.inertia
    return float(np.sqrt(2 * gain**2 / (d**2 + np.hypot(d**2, 2 * m * gain))))


def inferred_case2_models(model, reported_omega):
    target_gain = reported_omega * np.hypot(model.damping, model.inertia * reported_omega)
    gain_constant = 1.5 * model.voltage**2
    r = model.resistance
    high_x = (gain_constant + np.sqrt(gain_constant**2 - (2 * target_gain * r)**2)) / (2 * target_gain)
    low_x = r**2 / high_x
    return {name: replace(model, virtual_inductance=float(x / model.omega - model.grid_inductance))
            for name, x in (("case2_inferred_high_x", high_x), ("case2_inferred_low_x", low_x))}


def sign_crossings(frequencies, values, function):
    found = [float(f) for f in frequencies[values == 0.]]
    for index in np.flatnonzero(values[:-1] * values[1:] < 0.):
        found.append(brentq(function, frequencies[index], frequencies[index + 1], xtol=1e-11))
    return sorted(set(found))


def margins(pair, frequency, omega):
    response = evaluate(pair, 1j * frequency, omega)
    if not np.any(response):
        return {"gain_crossings": [], "negative_real_crossings": []}

    def gain_error(w):
        return abs(evaluate(pair, 1j * w, omega)) - 1.

    magnitude = abs(response)
    phase = np.unwrap(np.angle(response))
    gain_crossings = []
    for f in sign_crossings(frequency, magnitude - 1., gain_error):
        expected = np.interp(np.log(f), np.log(frequency), phase)
        angle = np.angle(evaluate(pair, 1j * f, omega))
        angle += 2 * np.pi * round((expected - angle) / (2 * np.pi))
        side = np.array([f * np.exp(-1e-4), f * np.exp(1e-4)])
        local_gain = 20 * np.log10(abs(evaluate(pair, 1j * side, omega)))
        gain_crossings.append({"omega_rad_s": float(f), "phase_margin_deg": float(180 + np.degrees(angle)),
                               "slope_db_decade": float(np.diff(local_gain)[0] / np.diff(np.log10(side))[0])})

    real_crossings = []
    for f in sign_crossings(frequency, response.imag, lambda w: evaluate(pair, 1j * w, omega).imag):
        real = evaluate(pair, 1j * f, omega).real
        if real < 0.:
            real_crossings.append({"omega_rad_s": float(f), "gain_margin_db": float(-20 * np.log10(-real))})
    return {"gain_crossings": gain_crossings, "negative_real_crossings": real_crossings}


def nyquist(pair, frequency, omega):
    # Clockwise RHP contour, with a right indentation excluding the
    # swing integrator at the origin.
    indentation = frequency[0] * np.exp(1j * np.linspace(-np.pi / 2, np.pi / 2, 1025))
    outer_arc = frequency[-1] * np.exp(1j * np.linspace(np.pi / 2, -np.pi / 2, 1025))
    s = np.r_[-1j * frequency[::-1], indentation[1:-1], 1j * frequency, outer_arc[1:]]
    samples = 1 + evaluate(pair, s, omega)
    steps = np.angle(samples[1:] / samples[:-1])
    return int(np.rint(steps.sum() / (2 * np.pi))), float(max(abs(steps)))
