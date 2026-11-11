#!/usr/bin/env python3
"""Independent physical-state verification. No primary implementation imported."""
from __future__ import annotations
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
from scipy.optimize import brentq

HERE = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(HERE / '.mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def physical_model(L, id0, iq0, Ra=.2, wb=.1, Kp=.2, V=1., omega=1., kappa=1.):
    i0 = complex(id0, iq0)
    g0 = V - 1j * omega * L * i0
    x0 = np.array([id0, iq0, id0, iq0, 0.])
    def field(x, e=0.):
        J, Y, delta = complex(*x[:2]), complex(*x[2:4]), x[4]
        rate = Kp * e
        v = V * np.exp(1j * delta) - Ra * (J - Y)
        dj = (v - g0) / L - 1j * omega * J
        dy = wb * (J - Y) + 1j * rate * Y
        return np.array([dj.real, dj.imag, dy.real, dy.imag, rate])
    def power(x):
        J, Y, delta = complex(*x[:2]), complex(*x[2:4]), x[4]
        v = V * np.exp(1j * delta) - Ra * (J - Y)
        return kappa * (v * J.conjugate()).real
    A = np.array([[-Ra/L, omega, Ra/L, 0, 0],
                  [-omega, -Ra/L, 0, Ra/L, V/L],
                  [wb, 0, -wb, 0, 0], [0, wb, 0, -wb, 0], [0, 0, 0, 0, 0]])
    B = Kp * np.array([0, 0, -iq0, id0, 1.])
    C = kappa * np.array([V-Ra*id0, -Ra*iq0, Ra*id0, Ra*iq0, V*iq0])
    h = 1e-5
    Afd = np.column_stack([(field(x0+np.eye(5)[k]*h)-field(x0-np.eye(5)[k]*h))/(2*h) for k in range(5)])
    Bfd = (field(x0,h)-field(x0,-h))/(2*h)
    Cfd = np.array([(power(x0+np.eye(5)[k]*h)-power(x0-np.eye(5)[k]*h))/(2*h) for k in range(5)])
    error = max(np.max(abs(A-Afd)), np.max(abs(B-Bfd)), np.max(abs(C-Cfd)))
    return A, B, C, float(error), float(np.max(abs(field(x0)))), float(abs(g0)), float(-np.angle(g0))


def frozen_model(L, id0, iq0, Ra=.2, Kp=.2, V=1., omega=1., kappa=1.):
    # Converter dq coordinates with the highpass perturbation frozen to Ra.
    A = np.array([[-Ra/L, omega, omega*id0], [-omega, -Ra/L, V/L+omega*iq0], [0.,0.,0.]])
    B = Kp * np.array([iq0, -id0, 1.])
    C = kappa * np.array([V-Ra*id0, -Ra*iq0, 0.])
    return A, B, C


def response(A, B, C, w):
    w = np.asarray(w)
    mat = 1j*w[...,None,None]*np.eye(len(A))-A
    rhs = np.broadcast_to(B, w.shape + B.shape)
    return np.linalg.solve(mat, rhs[...,None])[...,0] @ C


def paper_eq9(L, id0, iq0, w, Ra=.2, wb=.1, Kp=.2, V=1., omega=1., kappa=1.):
    s=1j*np.asarray(w); ha=Ra*s/(s+wb) if wb else np.full_like(s,Ra)
    a=omega*L*iq0/V
    b=-ha**2/V*(iq0/(omega*L)+(id0**2+iq0**2)/V)
    plant=(kappa*V**2/(omega*L))*(a*s**2+(1+a+b)*omega**2)/(s**2+2*ha/L*s+omega**2+(ha/L)**2)
    return Kp*plant/s


def root_crossings(fun, quantity, grid):
    vals=quantity(fun(grid))
    roots=[]
    for i in np.flatnonzero(vals[:-1]*vals[1:]<0):
        root=np.exp(brentq(lambda z: float(quantity(fun(np.exp(z)))),np.log(grid[i]),np.log(grid[i+1]),xtol=1e-13))
        if not roots or abs(root/roots[-1]-1)>1e-8: roots.append(float(root))
    return roots


def margins(fun):
    grid=np.geomspace(1e-7,1e5,6001)
    f=fun(grid)
    phase=np.unwrap(np.angle(f))
    if phase[0]>0: phase-=2*np.pi
    pcs=root_crossings(fun,np.imag,grid)
    gain=[]
    for w in pcs:
        val=fun(w)
        if val.real<0:
            gain.append({'frequency_pu':w,'factor':float(-1/val.real),'imaginary_residual':float(abs(val.imag))})
    ucs=root_crossings(fun,lambda z: abs(z)-1,grid)
    phases=[]
    for w in ucs:
        theta=float(np.angle(fun(w)))
        expected=np.interp(np.log(w),np.log(grid),phase)
        theta+=2*np.pi*round((expected-theta)/(2*np.pi))
        phases.append({'frequency_pu':w,'margin_deg':float(180+np.rad2deg(theta))})
    return {'negative_real_crossings':gain,'unity_gain_crossings':phases,
            'gain_margin_factor':min((r['factor'] for r in gain),default=None),
            'phase_margin_deg':min((r['margin_deg'] for r in phases),default=None),
            'frequency_search_pu':[1e-7,1e5]}


def pole_record(A):
    p=np.linalg.eigvals(A)
    return {'values_real_imag':[[float(z.real),float(z.imag)] for z in sorted(p,key=lambda z:(z.real,z.imag))],
            'max_real':float(max(p.real)), 'asymptotically_stable':bool(max(p.real)<-1e-10)}


def dc_matrix(Ac,B,C,Kd):
    return np.block([[Ac,(B*Kd)[:,None]],[-C[None,:],np.zeros((1,1))]])


def main():
    started=dt.datetime.now(dt.timezone.utc); tick=time.perf_counter()
    protocol=json.loads((HERE/'protocol.json').read_text())
    w=np.geomspace(1e-3,1e2,501); Kd=1/(4*np.sqrt(2)); freq={'w_pu':w}
    results={'doi':protocol['doi'],'protocol_sha256':hashlib.sha256((HERE/'protocol.json').read_bytes()).hexdigest(),
             'implementation_route':'Independent physical fixed-grid-frame Jacobian/resolvent; frozen converter-frame Jacobian; analytic algebra',
             'fig5':{},'fig7':{},'appendix_iv':{},'limits':{},'source_issues':{},'gates':{}}
    max_tf=0.; max_fd=0.; max_eq=0.
    fig,axes=plt.subplots(2,2,figsize=(9,7))
    for row,(operating,ids) in enumerate({'active':(.95,.2),'reactive':(.2,-.95)}.items()):
        for scr,style in [(10,'-'),(3,'--'),(1,'-.')]:
            L=1/scr; id0,iq0=ids
            A,B,C,fd,eq,Vg,delta=physical_model(L,id0,iq0)
            Ac=A-np.outer(B,C)
            fp=lambda ww: response(A,B,C,ww)
            fc=lambda ww: response(Ac,B,C,ww)
            fg=lambda ww: Kd*fc(ww)/(1j*np.asarray(ww))
            gp,gc,gd=fp(w),fc(w),fg(w)
            reference=paper_eq9(L,id0,iq0,w)
            tf=float(np.max(abs(gp-reference)/np.maximum(1,abs(reference))))
            closure=float(np.max(abs(gc-gp/(1+gp))/np.maximum(1,abs(gc))))
            max_tf=max(max_tf,tf,closure);max_fd=max(max_fd,fd);max_eq=max(max_eq,eq)
            key=f'{operating}_scr{scr}'
            results['fig5'][key]={'L_pu':L,'id0_pu':id0,'iq0_pu':iq0,'current_magnitude_pu':float(np.hypot(id0,iq0)),
                'Vg_pu_derived':Vg,'load_angle_deg_derived':float(np.rad2deg(delta)),
                'gp':margins(fp),'gd':margins(fg),'inner_closed_loop_poles':pole_record(Ac),
                'outer_closed_loop_poles':pole_record(dc_matrix(Ac,B,C,Kd)),
                'physical_vs_eq9_scaled_error':tf,'feedback_identity_scaled_error':closure,'fd_jacobian_max_error':fd}
            for kind,val in [('gp',gp),('gc',gc),('gd',gd)]: freq[f'fig5_{key}_{kind}']=val
            for col,val in [(0,gp),(1,gd)]: axes[row,col].plot(val.real,val.imag,style,label=f'SCR {scr}')
    for row in range(2):
        for col in range(2):
            axes[row,col].plot([-1],[0],'r+'); axes[row,col].set(xlim=(-1.5,0),ylim=(-.6,.8),xlabel='Re(G)',ylabel='Im(G)')
            axes[row,col].grid(alpha=.25); axes[row,col].legend(); axes[row,col].set_title(f'{["Active", "Reactive"][row]}: {["Gp","Gd"][col]}')
    fig.suptitle('Independent physical-state reconstruction of Fig. 5');fig.tight_layout();fig.savefig(HERE/'fig5_independent.png',dpi=160);plt.close(fig)
    for name,Kp,kd in [('psc',.2,Kd),('droop',.05,Kd),('droop_reduced_dc_gain',.05,.3*Kd)]:
        A,B,C,*_=physical_model(.5,.7,-.7,Kp=Kp);Ac=A-np.outer(B,C)
        fp=lambda ww:response(A,B,C,ww)
        fc=lambda ww:response(Ac,B,C,ww)
        fg=lambda ww:kd*fc(ww)/(1j*np.asarray(ww))
        results['fig7'][name]={'omega_b_pu_assumption':.1,'Kp_pu':Kp,'Kd_pu':float(kd),'gp':margins(fp),'gd':margins(fg),'inner_poles':pole_record(Ac),'outer_poles':pole_record(dc_matrix(Ac,B,C,kd))}
        freq[f'fig7_{name}_gp']=fp(w);freq[f'fig7_{name}_gc']=fc(w);freq[f'fig7_{name}_gd']=fg(w)
    # A physically admissible, nonzero-reactive frozen-filter example discriminates the two printed crossing formulas.
    appendix_cases={}
    max_cross_error=0.;max_gm_error=0.;max_inner_gm_error=0.
    for label,L,id0,iq0 in [('zero_current',.2*np.sqrt(2),0.,0.),('reactive_scr1',1.,.2,-.95),('active_scr3',1/3,.95,.2)]:
        A,B,C=frozen_model(L,id0,iq0);Ac=A-np.outer(B,C)
        fp=lambda ww:response(A,B,C,ww)
        fc=lambda ww:response(Ac,B,C,ww)
        fg=lambda ww:Kd*fc(ww)/(1j*np.asarray(ww))
        a=L*iq0;b=-.2**2*(iq0/L+id0**2+iq0**2);r=.2/L
        wp=np.sqrt((1+a+b)/(2+a));wrong_wp=np.sqrt((1-b)/(2+a))
        gm=((1-b)/((2+a)*r)+r)/(2*Kd)
        printed_gm=((1-b)/((2+a)*r)+r/2)/Kd
        bound=np.sqrt((1-b)/(2+a))/Kd
        numeric=margins(fg);nc=numeric['negative_real_crossings'][0]
        max_cross_error=max(max_cross_error,abs(nc['frequency_pu']/wp-1));max_gm_error=max(max_gm_error,abs(nc['factor']/gm-1))
        inner_numeric=margins(fp)
        inner_gm_printed=2*(1+r*r)/(1-.2**2*(id0**2+iq0**2))
        inner_gm=2*(1+r*r)/(1+b-a*r*r)
        max_inner_gm_error=max(max_inner_gm_error,abs(inner_numeric['gain_margin_factor']/inner_gm-1))
        cubic=np.roots([1,(2+a)*r,1+r*r,(1+a+b)*r])
        pole_error=max(min(abs(p-z) for z in np.linalg.eigvals(Ac)) for p in cubic)
        appendix_cases[label]={'L_pu':L,'id0_pu':id0,'iq0_pu':iq0,'a':a,'b':b,
            'corrected_crossing_frequency_pu':float(wp),'printed_crossing_frequency_pu':float(wrong_wp),
            'numeric_crossing_frequency_pu':nc['frequency_pu'],'gd_at_printed_frequency_real_imag':[float(fg(wrong_wp).real),float(fg(wrong_wp).imag)],
            'gain_margin_corrected':float(gm),'gain_margin_printed_eq42':float(printed_gm),'gain_margin_numeric':nc['factor'],
            'eq43_lower_bound':float(bound),'eq24_Kd_upper_bound':float(np.sqrt((1-b)/(2+a))/4),
            'theorem1_gm_corrected_eq39_first_expression':float(inner_gm),
            'theorem1_gm_printed_eq39_second_expression':float(inner_gm_printed),
            'theorem1_gm_numeric':inner_numeric['gain_margin_factor'],
            'physical_vs_eq14_max_pole_error':float(pole_error),'inner_poles':pole_record(Ac),'outer_poles':pole_record(dc_matrix(Ac,B,C,Kd))}
    results['appendix_iv']={'cases':appendix_cases,'frequency_formula':'omega1*sqrt((1+a+b)/(2+a))',
        'corrected_eq42':'((1-b)*omega1^2*L/(2*(2+a)*Ra)+Ra/(2*L))/Kd',
        'eq43_and_eq24':'Valid pointwise by AM-GM for positive quantities and applicable negative-real crossing; intermediate printed expressions inconsistent',
        'minimizer_holding_a_b_fixed':'(Ra/omega1)*sqrt((2+a)/(1-b)); printed sqrt(2)*Ra/omega1 is a=b=0 special case',
        'max_numeric_crossing_relative_error':max_cross_error,'max_numeric_gm_relative_error':max_gm_error}
    results['appendix_ii']={'corrected_gm_formula':'2*(1+(Ra/(omega1*L))^2)/(1+b-a*(Ra/(omega1*L))^2)',
        'corrected_denominator':'1-2*Ra^2*iq0/(omega1*L*V)-(Ra*abs(i0)/V)^2',
        'printed_error':'Eq39 first denominator to second denominator drops a nonzero reactive-current term; Eq40 inherits it',
        'bound_survives':'Numerator(half gm)-denominator = (Ra/(omega1*L))^2*abs(V-j*omega1*L*i0)^2/V^2 >=0; gm>=2 when denominator positive and crossing negative real',
        'max_corrected_formula_vs_physical_gm_relative_error':max_inner_gm_error}
    # Independently scan the stated operational regime, explicitly excluding endpoints with zero DC gain.
    scan=[]
    for L in np.geomspace(.1,1,21):
        for amplitude in [0,.25,.5,.75,.95,.999]:
            for angle in np.linspace(-np.pi,np.pi,25,endpoint=False):
                id0=amplitude*np.cos(angle);iq0=amplitude*np.sin(angle)
                a=L*iq0;b=-.04*(iq0/L+amplitude**2);r=.2/L
                A,B,C=frozen_model(L,id0,iq0);Ac=A-np.outer(B,C)
                kd=np.sqrt((1-b)/(2+a))/4
                gm1=2*(1+r*r)/(1+b-a*r*r)
                gm2=((1-b)/((2+a)*r)+r)/(2*kd)
                scan.append((gm1,gm2,max(np.linalg.eigvals(Ac).real),max(np.linalg.eigvals(dc_matrix(Ac,B,C,kd)).real)))
    scan=np.array(scan)
    results['frozen_filter_domain_scan']={'count':len(scan),'range':'V=omega1=kappa=1; Ra=.2; SCR1..10; |i|0..0.999; 25 current angles; wb=0; Kd chosen at Eq24 bound',
        'minimum_inner_gm_formula':float(scan[:,0].min()),'minimum_outer_gm_corrected_formula':float(scan[:,1].min()),
        'maximum_inner_pole_real':float(scan[:,2].max()),'maximum_outer_pole_real':float(scan[:,3].max()),
        'inner_unstable_count':int(np.sum(scan[:,2]>=0)),'outer_unstable_count':int(np.sum(scan[:,3]>=0)),
        'limitation':'Finite scan corroborates but does not prove global claims or cover depressed converter voltage/current limiting.'}
    for name,Ra,L,id0,iq0 in [('large_Ra_id_0',1.,1.,0.,0.),('large_Ra_id_0_99',1.,1.,.99,0.),('large_Ra_id_0_9999',1.,1.,.9999,0.),('reactive_iq_0',.2,1.,0.,0.),('reactive_iq_neg_0_99',.2,1.,0.,-.99),('reactive_iq_neg_0_9999',.2,1.,0.,-.9999)]:
        A,B,C=frozen_model(L,id0,iq0,Ra=Ra,Kp=Ra)
        a=L*iq0;b=-Ra**2*(iq0/L+id0**2+iq0**2)
        poles=pole_record(A-np.outer(B,C))
        results['limits'][name]={'Ra_pu':Ra,'L_pu':L,'id0':id0,'iq0':iq0,'a':a,'b':b,'slow_pole_approx':float(-(1+a+b)*Ra/L),'poles':poles}
    results['low_current_factorization']={}
    for scr in [1,3,10,20]:
        L=1/scr;r=.2/L;A,B,C=frozen_model(L,0,0);Ac=A-np.outer(B,C)
        poles=pole_record(Ac)
        results['low_current_factorization'][str(scr)]={'SCR':scr,'zeta_exact_for_a_b_zero':r/2,
            'eq20_zeta':.1*scr,'factorization':'(s+r)*(s^2+r*s+1), exact only a=b=0 in frozen-filter model',
            'pair_type':'complex' if r<2 else 'critical' if r==2 else 'two_real', 'poles':poles}
    results['source_issues']={'table_I_Ra':'4.4 ohm / 12.6 ohm = .349206 pu, inconsistent with .2 pu; independent analytic baseline uses recommended .2 pu',
        'Fig5_current_norm':float(np.hypot(.95,.2)),'frequency_units':'All generated frequencies normalized to omega1; dimensional rad/s=pu*2*pi*50',
        'experimental_claim_scope':'No raw traces and no hardware replication; Tables/Figs 11-17 cannot be treated as numerically replicated experiments.'}
    results['gates']={'max_physical_vs_paper_scaled_complex_error':max_tf,'transfer_gate_pass':bool(max_tf<1e-8),
        'max_analytic_vs_finite_difference_jacobian_error':max_fd,'jacobian_gate_pass':bool(max_fd<1e-7),
        'max_equilibrium_residual':max_eq,'appendix_numeric_gate_pass':bool(max_cross_error<1e-8 and max_gm_error<1e-8 and max_inner_gm_error<1e-8),
        'finite_filter_inner_all_stable':all(x['inner_closed_loop_poles']['asymptotically_stable'] for x in results['fig5'].values()),
        'finite_filter_outer_all_stable':all(x['outer_closed_loop_poles']['asymptotically_stable'] for x in results['fig5'].values()),
        'frozen_filter_scan_all_stable':bool(np.all(scan[:,2:]<0))}
    results['run']={'started_utc':started.isoformat(),'finished_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'measured_wall_elapsed_s':time.perf_counter()-tick}
    np.savez_compressed(HERE/'frequency_responses.npz',**freq)
    (HERE/'results.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'gates':results['gates'],'appendix_iv':results['appendix_iv'],'run':results['run']},indent=2))


if __name__=='__main__': main()
