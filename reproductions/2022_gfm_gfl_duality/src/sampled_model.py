"""Reconstruct the sampled single-inverter circuit from the author's SLX blocks."""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm
from scipy.optimize import root


@dataclass(frozen=True)
class SampledCircuit:
    kind: str
    key: str
    step: float
    rf: float
    xmain: float
    extra_x: float = 0.0
    base_hz: float = 50.0
    lf: float = 0.05
    cf: float = 0.02
    wi_hz: float = 500.0
    wv_hz: float = 250.0
    wpll_hz: float = 15.0
    droop: float = 0.05
    grid_breaker: float = 1e6
    load_r: float = np.inf
    integrator: str = "trapezoidal"

    @property
    def omega(self):
        return 2*np.pi*self.base_hz

    @property
    def plant_states(self):
        return 4 if self.extra_x else 3

    def plant(self):
        if self.extra_x:
            r=self.grid_breaker
            a=np.array([[-self.rf/self.lf, -1/self.lf, 0, 0],
                        [1/self.cf, -1/self.cf/self.load_r, -1/self.cf, 0],
                        [0, 1/self.xmain, -0.2-r/self.xmain, r/self.xmain],
                        [0, 0, r/self.extra_x, -0.2-r/self.extra_x]])
        else:
            a=np.array([[-self.rf/self.lf, -1/self.lf, 0],
                        [1/self.cf, -1/self.cf/self.load_r, -1/self.cf],
                        [0, 1/self.xmain, -0.2]])
        b=np.zeros((len(a),2))
        b[0,0]=1/self.lf
        b[2,1]=-1/self.xmain
        return self.omega*a,self.omega*b

    def propagator(self):
        a,b=self.plant()
        count=len(a)
        rotate=np.exp(-1j*self.omega*self.step)
        if self.integrator == "trapezoidal":
            inv=np.linalg.solve(np.eye(count)-self.step*a/2,np.eye(count))
            gain=inv @ (self.step*b/2)
            return (rotate*inv @ (np.eye(count)+self.step*a/2),
                    rotate*gain[:,0],gain[:,0],(1+rotate)*gain[:,1])
        if self.integrator == "backward_euler":
            inv=np.linalg.solve(np.eye(count)-self.step*a,np.eye(count))
            return rotate*inv,np.zeros(count),self.step*inv@b[:,0],self.step*inv@b[:,1]
        # Include held bridge input and exact grid sin/cos forcing in the
        # autonomous augmented system to avoid ill-conditioned inverses.
        aug=np.zeros((count+3,count+3),dtype=complex)
        aug[:count,:count]=a
        aug[:count,count]=b[:,0]
        aug[:count,count+1]=b[:,1]
        aug[count+1,count+2]=-self.omega
        aug[count+2,count+1]=self.omega
        e=expm(aug*self.step)
        return (rotate*e[:count,:count],rotate*e[:count,count],np.zeros(count),
                rotate*(e[:count,count+1]-1j*e[:count,count+2]))

    def gains(self):
        wi=2*np.pi*self.wi_hz
        wv=2*np.pi*self.wv_hz
        return wi*self.lf/self.omega,wi*wi*self.lf/(4*self.omega),wv*self.cf/self.omega,wv*wv*self.cf*20/(4*self.omega)

    def advance(self, state, mapping, *, scale=1.0):
        """All measurements precede Euler updates. The bridge has one delay."""
        n=self.plant_states
        plant=state[:2*n].view(np.complex128)
        control=state[2*n:]
        i_f,vc,i_g=plant[:3]
        zi=complex(control[0],control[1])
        pi,ii,pv,iv=self.gains()
        if self.kind=="GFM":
            zv=complex(control[2],control[3])
            filtered,delta=control[4:6]
            held=complex(control[6],control[7])
            rotation=np.exp(-1j*delta)
            error_v=1-vc*rotation
            scale_v=scale if self.key=="GFM_w_v" else 1.0
            current_ref=scale_v*pv*error_v+zv
            error_i=current_ref-i_f*rotation
            command_dq=pi*error_i+zi
            m=self.droop*(scale if self.key=="GFM_m" else 1)
            wpu=1-m*filtered
            new_filter=filtered+self.step*(2*np.pi*15)*(np.real(i_g*rotation)-filtered)
            new_controls=np.r_[zi.real+self.step*ii*error_i.real,
                               zi.imag+self.step*ii*error_i.imag,
                               zv.real+self.step*iv*scale_v**2*error_v.real,
                               zv.imag+self.step*iv*scale_v**2*error_v.imag,
                               new_filter,delta+self.omega*self.step*(wpu-1),
                               (command_dq*np.exp(1j*delta-self.omega*self.step*1j)).real,
                               (command_dq*np.exp(1j*delta-self.omega*self.step*1j)).imag]
        else:
            filtered,zpll,delta=control[2:5]
            held=complex(control[5],control[6])
            rotation=np.exp(-1j*delta)
            scale_i=scale if self.key=="GFL_w_i" else 1.0
            error_i=0.5-i_f*rotation
            command_dq=pi*scale_i*error_i+zi
            scale_pll=scale if self.key=="GFL_w_pll" else 1.0
            kp=2*np.pi*self.wpll_hz*scale_pll
            ki=(2*np.pi*self.wpll_hz)**2*scale_pll**2/4
            wpu=1+(kp*filtered+zpll)/self.omega
            new_filter=filtered+self.step*(2*np.pi*1000)*((vc*rotation).imag-filtered)
            new_controls=np.r_[zi.real+self.step*ii*scale_i**2*error_i.real,
                               zi.imag+self.step*ii*scale_i**2*error_i.imag,
                               new_filter,zpll+self.step*ki*filtered,
                               delta+self.omega*self.step*(wpu-1),
                               (command_dq*np.exp(1j*delta-self.omega*self.step*1j)).real,
                               (command_dq*np.exp(1j*delta-self.omega*self.step*1j)).imag]
        transition,old_bridge,new_bridge,grid=mapping
        new_held=complex(new_controls[-2],new_controls[-1])
        new_plant=transition @ plant+old_bridge*held+new_bridge*new_held+grid
        return np.r_[np.ascontiguousarray(new_plant).view(float),new_controls],wpu

    def initial(self):
        # Start from the physical operating point, then solve the complete
        # sampled equilibrium using only the circuit and controller equations.
        if self.kind=="GFM":
            voltage=1+0j
            current=1j*self.cf
            grid_current=0j
        else:
            zgrid=self.xmain*(0.2+1j)
            if self.extra_x:
                ze=self.extra_x*(0.2+1j)
                zgrid+=ze*self.grid_breaker/(ze+self.grid_breaker)
            factor=1+1j*self.cf*zgrid
            a=abs(factor)**2
            b=-np.real(factor*np.conj(zgrid))
            c=abs(zgrid)**2/4-1
            voltage=(-b+np.sqrt(b*b-4*a*c))/(2*a)
            delta=-np.angle(voltage-zgrid*(0.5-1j*self.cf*voltage))
            voltage=voltage*np.exp(1j*delta)
            current=0.5*np.exp(1j*delta)
            grid_current=current-1j*self.cf*voltage
        delta=np.angle(voltage)
        u=voltage+(self.rf+1j*self.lf)*current
        values=[current,voltage,grid_current]
        if self.extra_x:
            values.append(grid_current*self.grid_breaker/(self.grid_breaker+self.extra_x*(0.2+1j)))
        plant=np.asarray(values,dtype=complex).view(float)
        zi=u*np.exp(-1j*delta)
        held=u*np.exp(-1j*self.omega*self.step)
        if self.kind=="GFM":
            ctrl=np.r_[zi.real,zi.imag,current.real,current.imag,0,delta,held.real,held.imag]
        else:
            ctrl=np.r_[zi.real,zi.imag,0,0,delta,held.real,held.imag]
        guess=np.r_[plant,ctrl]
        mapping=self.propagator()
        answer=root(lambda x:self.advance(x,mapping)[0]-x,guess,tol=1e-11)
        residual=np.max(abs(self.advance(answer.x,mapping)[0]-answer.x))
        if residual>1e-8:
            raise RuntimeError(f"Sampled equilibrium failed {self.key} residual {residual}")
        return answer.x,float(residual)


def simulate(base, changed, time, event, restore, impulse_duration, *, backward_euler_events=False):
    """Preserve every electrical branch state through parameter events."""
    state,residual=base.initial()
    load=1e4
    from dataclasses import replace
    impulse=replace(changed,load_r=load)
    maps=[c.propagator() for c in (base,changed,impulse)]
    event_maps=[replace(c,integrator="backward_euler").propagator() for c in (base,changed,impulse)]
    result=np.empty((len(time),9))
    ratio={"GFM_m":4.0,"GFL_w_pll":4.0,"GFM_w_v":0.6,"GFL_w_i":0.6}.get(base.key,1.0)
    previous_branch=0
    for k,t in enumerate(time):
        active=event-1e-12<=t<restore-1e-12
        branch=2 if active and t<event+impulse_duration-1e-12 and base.key not in ("GFM_Z","GFL_Y") else int(active)
        circuit=(base,changed,impulse)[branch]
        mapping=event_maps[branch] if backward_euler_events and branch != previous_branch else maps[branch]
        next_state,w=circuit.advance(state,mapping,scale=ratio if active else 1.0)
        plant=state[:2*base.plant_states].view(np.complex128)
        controls=state[2*base.plant_states:]
        delta=controls[5] if base.kind=="GFM" else controls[4]
        rotation=np.exp(-1j*delta)
        result[k,:6]=np.asarray(plant[:3]*rotation).view(float)
        result[k,6:]=[w,delta,abs(plant[1])]
        state=next_state
        previous_branch=branch
    return result,residual
