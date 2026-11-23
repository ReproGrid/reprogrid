"""Virtual-PCC single-VSG model in Xiong, Wu and Blaabjerg, Eqs. (14)–(20)."""

from dataclasses import dataclass
from math import cos, hypot, pi, sin, sqrt

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, minimize_scalar


@dataclass(frozen=True)
class Vsg:
    omega: float
    x: float
    pref: float
    qref: float
    vset: float
    j: float
    dp: float
    dq: float
    r_grid: float
    r_virtual: float
    detect_voltage: float

    def impedance_norm_squared(self):
        return (self.r_grid+self.r_virtual)**2 + self.x**2

    def virtual_voltage(self, angle, grid):
        norm = self.impedance_norm_squared()
        a = self.dq * self.x/norm
        b = 1-self.dq*grid*(self.x*cos(angle)+(self.r_grid+self.r_virtual)*sin(angle))/norm
        c = self.vset+self.dq*self.qref
        return 2*c/(b+sqrt(b*b+4*a*c))

    def upper_voltage(self, grid):
        impedance = hypot(self.r_grid+self.r_virtual,self.x)
        a = self.dq * self.x/impedance**2
        b = 1-self.dq*grid/impedance
        c = self.vset+self.dq*self.qref
        return 2*c/(b+sqrt(b*b+4*a*c))

    def circuit(self, angle, grid, amplitude):
        virtual_pcc = amplitude*np.exp(1j*angle)
        current = (virtual_pcc-grid)/(self.r_grid+self.r_virtual+1j*self.x)
        physical_pcc = grid+(self.r_grid+1j*self.x)*current
        power = physical_pcc*current.conjugate()
        virtual_power = virtual_pcc*current.conjugate()
        return power,virtual_power,physical_pcc,current

    def power_flow(self, angle, grid, independent=False):
        voltage = self.virtual_voltage(angle,grid)
        if independent:
            # Solve the Q–V characteristic with a complex KVL power evaluation.
            def droop_error(amplitude):
                q = self.circuit(angle,grid,amplitude)[0].imag
                return amplitude-self.vset-self.dq*(self.qref-q)
            voltage = brentq(droop_error,0,2,xtol=1e-14)
            power = self.circuit(angle,grid,voltage)[0].real
        else:
            ev = voltage*grid
            sine,cosine = sin(angle),cos(angle)
            power = (self.r_grid*(voltage**2-ev*cosine)
                     +self.r_virtual*(ev*cosine-grid**2)+self.x*ev*sine)/self.impedance_norm_squared()
        return voltage,power

    def command(self, voltage, gain_pu):
        if voltage<=self.detect_voltage:
            return self.pref-gain_pu*(self.vset-voltage)
        return self.pref

    def margin(self, angle, grid, gain_pu, independent=False):
        voltage,power = self.power_flow(angle,grid,independent)
        return power-self.command(voltage,gain_pu)

    def peak_power(self, grid, gain_pu=0.):
        maximum = minimize_scalar(lambda a:-self.margin(a,grid,gain_pu),bounds=(0,pi),
                                  method="bounded",options={"xatol":1e-12})
        return float(maximum.x),float(-maximum.fun)

    def positive_equilibria(self, grid, gain_pu):
        angle,peak = self.peak_power(grid,gain_pu)
        gap = lambda a:self.margin(a,grid,gain_pu)
        if peak<=0 or gap(0)>=0 or gap(pi)>=0:
            return None
        return tuple(brentq(gap,left,right,xtol=1e-13) for left,right in ((0,angle),(angle,pi)))

    def fault_equilibria(self, grid, gain_pu, initial_angle):
        roots = self.positive_equilibria(grid,gain_pu)
        if roots is not None:
            return roots

        # For large gain interpretations the right saddle can lie above pi.
        # Examine a full period of the continuous fault power-angle equation.
        if self.upper_voltage(grid)>self.detect_voltage and gain_pu:
            raise ValueError("The periodic root search requires a continuously active sag detector.")
        gap = lambda angle:self.margin(angle,grid,gain_pu)
        angles = np.linspace(-pi,pi,257)
        values = np.array([gap(angle) for angle in angles])
        rising = np.flatnonzero((values[:-1]<0) & (values[1:]>=0))
        falling = np.flatnonzero((values[:-1]>0) & (values[1:]<=0))
        if not len(rising) or not len(falling):
            return None
        stable = min((brentq(gap,angles[i],angles[i+1]) for i in rising),key=lambda root:abs(root-initial_angle))
        saddles = [brentq(gap,angles[i],angles[i+1]) for i in falling]
        saddle = min(stable+(root-stable)%(2*pi) for root in saddles)
        return stable,saddle

    def initial_angle(self, nominal):
        equilibria = self.positive_equilibria(nominal,0.)
        if equilibria is None:
            raise ValueError("No nominal point under the configured R–L load flow")
        if self.virtual_voltage(equilibria[0],nominal)<=self.detect_voltage:
            raise ValueError("The configured sag detector is active at the nominal point")
        return equilibria[0]

    def simulate(self, grid, angle0, gain_pu, saddle, numerical, first_swing_only=False, independent=False):
        def rhs(_, state):
            angle,speed = state
            return speed,(-self.omega*self.margin(angle,grid,gain_pu,independent)-self.dp*speed)/self.j

        def cross_saddle(_, state):
            return state[0]-saddle

        def turn(_, state):
            return state[1]

        def forward_slip(_, state):
            return state[0]-angle0-2*pi

        def backward_slip(_, state):
            return state[0]-angle0+2*pi

        cross_saddle.direction = forward_slip.direction = 1
        turn.direction = backward_slip.direction = -1
        cross_saddle.terminal = turn.terminal = first_swing_only
        forward_slip.terminal = backward_slip.terminal = True
        events = (cross_saddle,turn,forward_slip,backward_slip) if saddle is not None else (turn,forward_slip,backward_slip)
        horizon = numerical["search_horizon_s"] if first_swing_only else numerical["trace_horizon_s"]
        sol = solve_ivp(rhs,(0,horizon),(angle0,0.),events=events,dense_output=not first_swing_only,
                        method="RK45" if independent else "DOP853",max_step=numerical["max_step_s"],
                        rtol=numerical["solver_rtol"],atol=numerical["solver_atol"])
        if not sol.success:
            raise RuntimeError(sol.message)
        return sol,events
