"""Grid-synchronous averaged dVOC models and their exact real Jacobians."""
from dataclasses import dataclass
from math import atan2, pi

import numpy as np


@dataclass(frozen=True)
class Parameters:
    rg: float
    xg: float
    p_star: float
    q_star: float
    alpha: float
    eta_over_omega0: float
    base: dict
    angle: str = 'impedance_matched'
    phi_override: float | None = None

    @property
    def omega0(self):
        return 2 * pi * self.base['nominal_frequency_hz']

    @property
    def omega_g(self):
        return self.base['grid_frequency_pu'] * self.omega0

    @property
    def omega_delta(self):
        return self.omega0 - self.omega_g

    @property
    def eta(self):
        return self.eta_over_omega0 * self.omega0

    @property
    def lg(self):
        return self.xg / self.omega0

    @property
    def z(self):
        return self.rg + 1j * self.omega_g * self.lg

    @property
    def y(self):
        return 1 / self.z

    @property
    def phi(self):
        if self.phi_override is not None:
            return self.phi_override
        if self.angle == 'printed_literal':
            return atan2(self.lg, self.rg)
        if self.angle == 'impedance_matched':
            return atan2(self.z.imag, self.z.real)
        raise ValueError(f'Unknown rotation angle interpretation {self.angle}')

    @property
    def rotation(self):
        return np.exp(1j * self.phi)

    @property
    def v_star(self):
        return self.base['v_star']

    @property
    def rotated_setpoint(self):
        return self.rotation * (self.p_star - 1j * self.q_star) / self.v_star**2

    @property
    def cf(self):
        return self.base['filter_susceptance_pu'] / self.omega0

    @property
    def lf(self):
        return self.base['filter_x_pu'] / self.omega0

    @property
    def yf(self):
        return self.base['filter_g_pu'] + 1j * self.omega_g * self.cf


def complex_to_real(matrix):
    """Convert complex-linear coefficients to interleaved real coordinates."""
    rows, cols = matrix.shape
    result = np.empty((2 * rows, 2 * cols))
    result[::2, ::2] = matrix.real
    result[::2, 1::2] = -matrix.imag
    result[1::2, ::2] = matrix.imag
    result[1::2, 1::2] = matrix.real
    return result


def rectangular(values):
    return np.ascontiguousarray(values, dtype=np.complex128).view(np.float64)


def cubic_coefficients(p, vg):
    """Ascending radial equation times s, returned in descending order."""
    mu = p.rotated_setpoint - p.rotation * p.y + p.alpha + 1j * p.omega_delta / p.eta
    beta = p.alpha / p.v_star**2
    forcing = p.rotation * p.y * vg
    coeff = np.array([beta**2, -2 * beta * mu.real, abs(mu)**2, -abs(forcing)**2])
    return coeff, mu, beta, forcing


def equilibria(p, vg):
    if vg <= 0:
        raise ValueError('This root routine is for a positive forced grid voltage.')
    coeff, mu, beta, forcing = cubic_coefficients(p, vg)
    if beta == 0:
        if mu == 0:
            return []
        return [complex(-forcing / mu)]
    squares = sorted(s.real for s in np.roots(coeff) if abs(s.imag) < 1e-9 and s.real > 0)
    return [complex(-forcing / (mu - beta * s)) for s in squares]


def lifted_equilibrium(p, vg, v, order):
    if p.omega_delta != 0:
        raise ValueError('The equilibrium lift uses the Appendix B assumption omega_delta=0.')
    current = p.y * (v - vg)
    values = [v, current, v, 0j, p.yf * v + current, 0j]
    return rectangular(values[:order // 2])


def stability_margins(p, v):
    kappa = p.rotated_setpoint - p.rotation * p.y
    radius_term = p.alpha * abs(v)**2 / p.v_star**2
    return {
        'global_eq23': float(0.5 * radius_term - (kappa.real + p.alpha)),
        'local_sufficient': float(radius_term - (kappa.real + p.alpha)),
        'trace_instability_margin': float(kappa.real + p.alpha - 2 * radius_term),
    }


def eta_certificate_supremum(p, v):
    """Limit of Eq. (26b) as c1 increases to margin and epsilon decreases to 3."""
    margin = stability_margins(p, v)['global_eq23']
    if margin <= 0:
        return 0.
    ce = abs(p.rotated_setpoint - p.rotation * p.y + p.alpha)
    ce += 3 * p.alpha * abs(v)**2 / p.v_star**2
    return margin / ((p.lg / p.rg) * abs(p.y) * (margin + ce)) / p.omega0


def voltage_bound(p, vg):
    kappa_r = (p.rotated_setpoint - p.rotation * p.y).real
    radicand = 1 + (kappa_r + abs(p.y)) / p.alpha
    return max(vg, p.v_star * np.sqrt(max(0., radicand)))


class AveragedModel:
    """Assemble the affine and cubic terms in Eqs. (19)–(22)."""

    def __init__(self, p, vg, order):
        if order not in (2, 4, 8, 12):
            raise ValueError(f'Unsupported order {order}')
        n = order // 2
        linear = np.zeros((n, n), dtype=complex)
        forcing = np.zeros(n, dtype=complex)
        linear[0, 0] = 1j * p.omega_delta + p.eta * (p.rotated_setpoint + p.alpha)
        if order == 2:
            linear[0, 0] -= p.eta * p.rotation * p.y
            forcing[0] = p.eta * p.rotation * p.y * vg
        else:
            linear[0, 1] = -p.eta * p.rotation
            terminal = 0 if order == 4 else 2
            linear[1, terminal] = 1 / p.lg
            linear[1, 1] = -p.z / p.lg
            forcing[1] = -vg / p.lg

        if order >= 8:
            kp_v = p.base['kp_v']
            kr_v = p.base['kr_v_per_s']
            linear[3, [0, 2, 3]] = [-1, 1, 1j * p.omega_delta]
            if order == 8:
                linear[2, [0, 2, 3]] = np.array([kp_v, -kp_v, -kr_v]) / p.cf
            else:
                kp_c = p.base['kp_c']
                kr_c = p.base['kr_c_per_s']
                linear[2, [1, 2, 4]] = np.array([-1, -p.yf, 1]) / p.cf
                current_reference = np.array([kp_v, 1, p.yf - kp_v, -kr_v, 0, 0])
                current_error = np.array([0, 0, 0, 0, 1, 0]) - current_reference
                linear[4] = -kp_c * current_error / p.lf
                linear[4, 5] -= kr_c / p.lf
                linear[5] = current_error
                linear[5, 5] += 1j * p.omega_delta

        self.linear = complex_to_real(linear)
        self.forcing = rectangular(forcing)
        self.cubic_gain = p.eta * p.alpha / p.v_star**2

    def rhs(self, t, x):
        result = self.linear @ x + self.forcing
        voltage = x[:2]
        result[:2] -= self.cubic_gain * (voltage @ voltage) * voltage
        return result

    def jacobian(self, t, x):
        result = self.linear.copy()
        voltage = x[:2]
        derivative = (voltage @ voltage) * np.eye(2) + 2 * np.outer(voltage, voltage)
        result[:2, :2] -= self.cubic_gain * derivative
        return result
