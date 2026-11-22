"""Single-inverter models with the original analytic script conventions."""

from dataclasses import dataclass, replace

import numpy as np
from numpy.polynomial import Polynomial


J = np.array([[0.0, -1.0], [1.0, 0.0]])


@dataclass(frozen=True)
class Parameters:
    kind: str
    base_hz: float = 50.0
    lf: float = 0.05
    cf: float = 0.02
    rf: float = 0.01
    grid_x_pu: float = 0.3
    droop_pu: float = 0.05
    droop_lpf_hz: float = 15.0
    current_index_hz: float = 500.0
    voltage_index_hz: float = 250.0
    voltage_ki_multiplier: float = 20.0
    pll_index_hz: float = 15.0
    voltage: float = 1.0
    load_current: float = -0.5

    @property
    def base_rad_s(self):
        return 2 * np.pi * self.base_hz

    def gains(self):
        wi = self.current_index_hz / self.base_hz
        wv = self.voltage_index_hz / self.base_hz
        return wi * self.lf, wi**2 * self.lf / 4, wv * self.cf, wv**2 * self.cf * self.voltage_ki_multiplier / 4


def from_protocol(config, kind):
    p = config[kind.lower()]
    return Parameters(kind=kind, base_hz=config["base_frequency_hz"],
                      lf=config["filter_l_pu"], cf=config["filter_c_pu"],
                      rf=p["filter_r_analytic_pu"], grid_x_pu=p["grid_x_pu"],
                      current_index_hz=p["current_index_hz"],
                      voltage_index_hz=p.get("voltage_index_hz", 250),
                      droop_pu=p.get("droop_pu", 0.05),
                      droop_lpf_hz=p.get("droop_lpf_hz", 15),
                      voltage_ki_multiplier=p.get("voltage_ki_multiplier", 20),
                      pll_index_hz=p.get("pll_index_hz", 15),
                      voltage=p.get("analytic_voltage_pu", 1),
                      load_current=p.get("analytic_load_current_pu", -0.5))


def conjugate(poly):
    return Polynomial(poly.coef.conj())


def gfm_characteristic(p):
    s = Polynomial([0, 1])
    pi, ii, pv, iv = p.gains()
    n = p.lf * s * (s + 1j) + (p.rf + pi) * s + ii
    d = s**2 + p.cf * (s + 1j) * n * s + (pi * s + ii) * (pv * s + iv)
    line = p.grid_x_pu * (s + 1j + 0.2)
    total = s * n + line * d
    theta = s * (1 + s * p.base_hz / p.droop_lpf_hz)
    return theta * total * conjugate(total) + 0.5j * p.droop_pu * (conjugate(total) * d - total * conjugate(d))


def gfm_state_matrix(p):
    """Independent 12-state circuit reproducing the author zero-current model."""
    pi, ii, pv, iv = p.gains()

    def rhs(state):
        i_f, v, z_i, z_v, i_g = state[:10].reshape(5, 2)
        filtered, theta = state[10:]
        error = z_v - pv * v - i_f
        u = pi * error + z_i
        return np.r_[
            (u - v - p.rf * i_f) / p.lf - J @ i_f,
            (i_f - i_g) / p.cf - J @ v,
            ii * error, -iv * v,
            (v + np.array([0.0, theta])) / p.grid_x_pu - 0.2 * i_g - J @ i_g,
            p.droop_lpf_hz / p.base_hz * (i_g[0] - filtered),
            -p.droop_pu * filtered,
        ]
    return np.column_stack([rhs(b) for b in np.eye(12)]) * p.base_rad_s


def gfl_characteristic(p, *, literal_off_diagonal=True):
    """Eliminate the dq± nodal equations in the author's CalcSwingPLL.m.

    The released off-diagonal inner admittance lacks division by Z_inner.
    Keep that literal expression and calculate the divided version separately.
    """
    s = Polynomial([0, 1])
    pi, ii, _, _ = p.gains()
    w = p.pll_index_hz / p.base_hz
    h = w * s + w**2 / 4
    k = p.load_current * p.lf / 2
    n = p.lf * s * (s + 1j) + (p.rf + pi) * s + ii
    line = p.grid_x_pu * (s + 1j + 0.2)
    e = p.cf * (s + 1j) * line + 1
    a = s - k * h
    b = a * line + e * n
    nm, lm, bm = conjugate(n), conjugate(line), conjugate(b)
    pll = s * s + p.voltage * h
    kh = k * h
    if literal_off_diagonal:
        q = p.load_current * s * n - p.voltage * a * s + p.voltage * kh * n
        qm = conjugate(q)
        poly = s*s*pll*b*bm - kh*kh*n*nm*line*lm*pll + h/2 * (
            line*q*(s*bm+kh*nm*lm) + lm*qm*(s*b+kh*n*line))
    else:
        q = p.load_current*n - p.voltage*(s-2*kh)
        qm = conjugate(q)
        poly = pll*(b*bm-kh*kh*line*lm) + h/2*(
            line*q*(bm+kh*lm) + lm*qm*(b+kh*line))
    return poly


def roots(p, *, literal_off_diagonal=True):
    polynomial = (gfm_characteristic(p) if p.kind == "GFM" else
                  gfl_characteristic(p, literal_off_diagonal=literal_off_diagonal))
    if np.max(abs(polynomial.coef.imag)) > 1e-9:
        raise ArithmeticError("Characteristic coefficients must be real.")
    coefficients = polynomial.coef.real
    # Analytic denominator clearing contributes an exact origin factor.
    first = np.flatnonzero(coefficients != 0)[0]
    return Polynomial(coefficients[first:]).roots() * p.base_rad_s


def gfl_nodal_matrix(p, s, *, literal_off_diagonal=True):
    """Direct rational source equations used to verify polynomial roots."""
    pi, ii, _, _ = p.gains()
    w = p.pll_index_hz / p.base_hz
    h = w + w**2 / (4*s)
    inner = pi + ii / s + p.rf + p.lf * (s + 1j)
    inner_minus = pi + ii / s + p.rf + p.lf * (s - 1j)
    k = p.load_current * p.lf * h / 2
    off_plus = k if literal_off_diagonal else k / inner
    off_minus = k if literal_off_diagonal else k / inner_minus
    y = np.array([[(1-k)/inner, off_plus], [off_minus, (1-k)/inner_minus]])
    i0 = np.array([1j*p.load_current, -1j*p.load_current])
    v0 = np.array([1j*p.voltage, -1j*p.voltage])
    gw = h / (2j) * np.array([1, -1])
    y += np.outer(i0-y @ v0, gw) / (s + gw @ v0)
    y += np.diag([p.cf*(s+1j)+1/(p.grid_x_pu*(s+1j+0.2)),
                  p.cf*(s-1j)+1/(p.grid_x_pu*(s-1j+0.2))])
    return y


def dominant_mode(values):
    oscillatory = values[values.imag > 1e-5]
    return oscillatory[np.argmax(oscillatory.real)]


def mode_record(mode):
    return {"real_per_s": float(mode.real), "imag_hz": float(mode.imag / (2*np.pi))}
