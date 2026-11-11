"""Independent scalar PSC transient verifier: projective exact flow + quadrature.

No primary or other-agent implementation is imported. Input comes only from
the independently transcribed protocol and the paper. Time is in seconds.
"""
from __future__ import annotations
from pathlib import Path
import hashlib
import json
import os
import platform
import sys
import time

HERE=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(HERE/'.mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.integrate import quad, solve_ivp
from scipy.optimize import brentq

PROTOCOL=json.loads((HERE/'protocol.json').read_text())
LIMIT=PROTOCOL['current_scope_audit']['reported_limit_pu']

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def exact_flow(a,b,delta0,times):
    """Möbius/projective Riccati flow, avoiding arctan branch/domain mistakes.

    With z=tan(delta/2), z'=a*z*z/2-b*z+a/2. Write z=u/v,
    [u,v]'=[[-b/2,a/2],[-a/2,b/2]]*[u,v]. M²=(b²-a²)I/4.
    Input times must start at zero and be sufficiently dense for phase unwrap.
    """
    t=np.asarray(times,float)
    if len(t)==0 or t[0]!=0 or np.any(np.diff(t)<0):raise ValueError('times start at 0, monotone')
    initial=np.array([np.sin(delta0/2),np.cos(delta0/2)])
    M=np.array([[-b/2,a/2],[-a/2,b/2]])
    q2=(b*b-a*a)/4
    if q2>1e-14:
        q=np.sqrt(q2);c=np.cosh(q*t);s=np.sinh(q*t)/q
    elif q2 < -1e-14:
        q=np.sqrt(-q2);c=np.cos(q*t);s=np.sin(q*t)/q
    else:
        c=np.ones_like(t);s=t
    z=c[:,None]*initial+s[:,None]*(M@initial)
    phase=np.unwrap(np.arctan2(z[:,0],z[:,1]))
    return delta0+2*(phase-phase[0])

def travel_time(a,b,d0,d1):
    return quad(lambda d:1/(a-b*np.sin(d)),d0,d1,epsabs=1e-12,epsrel=1e-12,limit=200)[0]

def paper_cct_primitive(a,b,d0,d1):
    if not a>b>0:raise ValueError('Real arctan form restricted to a>b>0')
    # This branch is valid for the tested CCA<pi and initial angle in (0,pi/2).
    g=np.sqrt(a*a-b*b)
    return 2/g*(np.arctan((a*np.tan(d1/2)-b)/g)-np.arctan((a*np.tan(d0/2)-b)/g))

def network(XT,X1,X2,C=None,post=False):
    if post:return {'E':1.,'X':XT+X1,'Pmax_pu':1/(XT+X1),'Xtransfer':XT+X1}
    B=X1*X2/(X1+X2)
    if C is None:E,X=1.,XT+B
    else:E,X=C/(B+C),XT+B*C/(B+C)
    return {'E':E,'X':X,'Pmax_pu':E/X,'Xtransfer':X/E}

def current(delta,net):return abs(np.exp(1j*np.asarray(delta))-net['E'])/net['X']

def nodal_check(XT,X1,X2,C):
    net=network(XT,X1,X2,C)
    errors=[];residuals=[];powers=[]
    for delta in np.linspace(-np.pi,np.pi,101):
        U=np.exp(1j*delta)
        Y=1/(1j*XT)+1/(1j*X1)+1/(1j*X2)+1/(1j*C)
        W=(U/(1j*XT)+1/(1j*X1)+1/(1j*X2))/Y
        I=(U-W)/(1j*XT)
        It=(U-net['E'])/(1j*net['X'])
        P=np.real(U*np.conj(I))
        errors.append(abs(I-It))
        residuals.append(abs((W-U)/(1j*XT)+(W-1)/(1j*X1)+(W-1)/(1j*X2)+W/(1j*C)))
        powers.append(abs(P-net['Pmax_pu']*np.sin(delta)))
    return {'max_current_complex_error_pu':float(max(errors)),
            'max_power_error_pu':float(max(powers)),
            'max_kcl_residual':float(max(residuals))}

def scalar_at(a,b,d0,t):
    # Dense path fixes winding before scalar evaluation; deterministic, no ODE.
    if t==0:return d0
    return float(exact_flow(a,b,d0,np.linspace(0,t,max(3,int(np.ceil(t*200))+1)))[-1])

def stable_basin(delta,sep,uep):
    if abs((delta-uep+np.pi)%(2*np.pi)-np.pi)<1e-10:return None
    return int(np.floor((delta-uep)/(2*np.pi))+1)

def evaluate_branch(name,one,two,a):
    pre1=network(*one);post1=network(*one,post=True)
    d01=float(np.arcsin(1/pre1['Pmax_pu']));ds1=float(np.arcsin(1/post1['Pmax_pu']))
    t1=np.linspace(0,4,4001)
    delta1=exact_flow(a,a*post1['Pmax_pu'],d01,t1)
    ode1=solve_ivp(lambda t,y:[a*(1-post1['Pmax_pu']*np.sin(y[0]))],(0,4),[d01],
                   t_eval=t1,method='DOP853',rtol=1e-11,atol=1e-13,max_step=.005)
    e1=float(np.max(abs(delta1-ode1.y[0])))
    mono=bool(np.min(np.diff(delta1))>=-1e-10 and delta1.max()<=ds1+1e-10)
    typeI={'initial_angle_deg':float(np.degrees(d01)),'post_sep_angle_deg':float(np.degrees(ds1)),
           'post_uep_angle_deg':float(np.degrees(np.pi-ds1)),
           'pre_network':pre1,'post_network':post1,'a_rad_s':a,
           'maximum_overshoot_rad':float(max(0,delta1.max()-ds1)),
           'monotonic_without_overshoot':mono,'endpoint_error_rad':float(abs(delta1[-1]-ds1)),
           'matrix_vs_ODE_max_abs_angle_rad':e1,
           'maximum_current_pu':float(np.max(current(delta1,post1))),
           'trajectory_file':f'{name}_typeI.npz'}
    np.savez_compressed(HERE/typeI['trajectory_file'],time_since_disturbance_s=t1,delta_unwrapped_rad=delta1,
                        power_pu=post1['Pmax_pu']*np.sin(delta1),current_magnitude_pu=current(delta1,post1))

    XT,X1,X2,C=two
    pre=network(XT,X1,X2);fault=network(XT,X1,X2,C);post=network(XT,X1,X2,post=True)
    d0=float(np.arcsin(1/pre['Pmax_pu']));sep=float(np.arcsin(1/post['Pmax_pu']));uep=np.pi-sep
    bf=a*fault['Pmax_pu'];bp=a*post['Pmax_pu']
    cct=travel_time(a,bf,d0,uep)
    cct_paper=paper_cct_primitive(a,bf,d0,uep)
    cct_root=brentq(lambda t:scalar_at(a,bf,d0,t)-uep,0,2*cct,xtol=1e-13)
    cca_root=brentq(lambda d:1-post['Pmax_pu']*np.sin(d),np.pi/2,np.pi,xtol=1e-14)
    period_exact=2*np.pi/np.sqrt(a*a-bf*bf)
    period_quad=travel_time(a,bf,d0,d0+2*np.pi)
    checks=nodal_check(XT,X1,X2,C)
    cases={}
    fig,axes=plt.subplots(3,1,figsize=(10,8),sharex=True)
    for label,clearing in [('early',.5),('late',.7),('uncleared',None)]:
        color={'early':'#2464a6','late':'#d57721','uncleared':'#a72f3f'}[label]
        times=np.linspace(0,4,8001)
        if clearing is None:
            angles=exact_flow(a,bf,d0,times)
            powers=fault['Pmax_pu']*np.sin(angles)
            amps=current(angles,fault)
            clear_angle=None;basin=None;target=None
            sol=solve_ivp(lambda t,y:[a-bf*np.sin(y[0])],(0,4),[d0],t_eval=times,
                           method='DOP853',rtol=1e-11,atol=1e-13,max_step=.005)
            odeangles=sol.y[0]
        else:
            k=int(round(clearing/(times[1]-times[0])))
            fault_angles=exact_flow(a,bf,d0,times[:k+1]);clear_angle=float(fault_angles[-1])
            after=exact_flow(a,bp,clear_angle,times[k:]-times[k])
            angles=np.r_[fault_angles[:k],after]
            # Right-continuous topology at clearing, continuous angle.
            powers=np.r_[fault['Pmax_pu']*np.sin(fault_angles[:k]),post['Pmax_pu']*np.sin(after)]
            amps=np.r_[current(fault_angles[:k],fault),current(after,post)]
            basin=stable_basin(clear_angle,sep,uep);target=sep+2*np.pi*basin
            solf=solve_ivp(lambda t,y:[a-bf*np.sin(y[0])],(0,clearing),[d0],t_eval=np.clip(times[:k+1],0,clearing),
                           method='DOP853',rtol=1e-11,atol=1e-13,max_step=.005)
            solp=solve_ivp(lambda t,y:[a-bp*np.sin(y[0])],(clearing,4),[solf.y[0,-1]],t_eval=np.clip(times[k:],clearing,4),
                           method='DOP853',rtol=1e-11,atol=1e-13,max_step=.005)
            odeangles=np.r_[solf.y[0,:-1],solp.y[0]]
        over=np.flatnonzero(amps>=LIMIT)
        limit_t=None;limit_kind=None
        if len(over):
            k=int(over[0])
            if clearing is not None and abs(times[k]-clearing)<1e-12:
                limit_t=clearing;limit_kind='topology_jump_at_clearance'
            elif k==0:limit_t=0.;limit_kind='already_above_at_fault_inception'
            else:
                localnet=fault if clearing is None or times[k]<clearing else post
                def margin(t):
                    d=scalar_at(a,bf,d0,t) if localnet is fault else scalar_at(a,bp,clear_angle,t-clearing)
                    return float(current(d,localnet)-LIMIT)
                limit_t=brentq(margin,times[k-1],times[k],xtol=1e-13)
                limit_kind='continuous_crossing'
        err=float(np.max(abs(angles-odeangles)))
        record={'clearing_duration_s':clearing,'clear_angle_deg':None if clear_angle is None else float(np.degrees(clear_angle)),
                'clear_angle_rad':clear_angle,'attractor_basin_index':basin,
                'formal_scalar_model_classification':'unbounded_phase_running' if clearing is None else 'same_well_recapture' if basin==0 else f'{basin}_slip_recapture',
                'target_attractor_unwrapped_rad':target,'final_angle_unwrapped_rad':float(angles[-1]),
                'endpoint_distance_to_attractor_rad':None if target is None else float(abs(angles[-1]-target)),
                'first_current_limit_since_fault_s':None if limit_t is None else float(limit_t),
                'first_current_limit_absolute_s':None if limit_t is None else float(3+limit_t),
                'current_limit_crossing_type':limit_kind,'maximum_current_pu':float(max(amps)),
                'within_assumed_no_current_limit_region':limit_t is None,
                'matrix_vs_ODE_max_abs_angle_rad':err,
                'physical_scope':'Scalar PSC prediction only; no limiter or controller-mode switch is simulated',
                'trajectory_file':f'{name}_typeII_{label}.npz'}
        if clearing is None:record['slip_period_s']=float(period_exact)
        cases[label]=record
        np.savez_compressed(HERE/record['trajectory_file'],time_since_fault_s=times,time_absolute_s=times+3,
                            delta_unwrapped_rad=angles,delta_wrapped_rad=angles%(2*np.pi),
                            power_pu=powers,current_magnitude_pu=amps)
        axes[0].plot(times,angles,label=label,color=color)
        axes[1].plot(times,powers,label=label,color=color)
        axes[2].plot(times,amps,label=label,color=color)
        if limit_t is not None:axes[2].plot(limit_t,LIMIT,'o',ms=4,color=color)
    axes[0].axhline(uep,ls='--',c='gray',label='first UEP / CCA')
    axes[2].axhline(LIMIT,ls='--',c='black',label='paper stated current limit')
    for ax,ylabel in zip(axes,['Unwrapped angle (rad)','Power (pu)','Current magnitude (pu)']):
        ax.set_ylabel(ylabel);ax.grid(alpha=.25);ax.legend(fontsize=8)
    axes[2].set_xlabel('Time since disturbance (s)')
    fig.suptitle(f'{name}: scalar PSC model; current-limit crossing flags model-scope exit')
    fig.tight_layout();fig.savefig(HERE/f'{name}_typeII.png',dpi=160);plt.close(fig)
    boundary=[]
    for offset in [-.001,.001]:
        dc=scalar_at(a,bf,d0,cct+offset)
        boundary.append({'clearing_time_s':cct+offset,'offset_from_CCT_s':offset,
                         'clear_angle_deg':float(np.degrees(dc)),
                         'attractor_basin_index':stable_basin(dc,sep,uep)})
    typeII={'pre_network':pre,'fault_network':fault,'post_network':post,'a_rad_s':a,'b_fault_rad_s':bf,
            'initial_angle_deg':float(np.degrees(d0)),'post_sep_angle_deg':float(np.degrees(sep)),
            'CCA_deg':float(np.degrees(uep)),'CCA_root_deg':float(np.degrees(cca_root)),
            'CCT_s':float(cct),'CCT_paper_formula_s':float(cct_paper),'CCT_root_s':float(cct_root),
            'CCT_formula_quad_difference_s':float(abs(cct-cct_paper)),
            'CCT_root_quad_difference_s':float(abs(cct-cct_root)),
            'uncleared_slip_period_s':float(period_exact),'uncleared_period_quad_s':float(period_quad),
            'boundary_checks':boundary,'exact_UEP':'Unstable equilibrium; exact equality remains there in exact arithmetic, attraction is not guaranteed',
            'circuit_check':checks,'cases':cases}
    gate=PROTOCOL['gates']
    passed=(mono and e1<gate['matrix_vs_ODE_max_abs_angle_rad'] and
            all(x['matrix_vs_ODE_max_abs_angle_rad']<gate['matrix_vs_ODE_max_abs_angle_rad'] for x in cases.values()) and
            abs(cct-cct_paper)<gate['analytic_quadrature_CCT_max_abs_s'] and
            abs(cct-cct_root)<gate['root_quadrature_CCT_max_abs_s'] and
            checks['max_power_error_pu']<gate['nodal_vs_Thevenin_max_abs_power_pu'] and
            checks['max_kcl_residual']<gate['nodal_kcl_max_abs'] and
            all((row['attractor_basin_index']==0)==(row['clearing_duration_s']<cct)
                for label,row in cases.items() if label!='uncleared'))
    return {'typeI':typeI,'typeII':typeII,'internal_numeric_checks_passed':bool(passed)}

def main():
    start=time.perf_counter()
    branches={}
    branches['stated_pu']=evaluate_branch('stated_pu',[.02,.85,.85],[.8,.15,.8,.5],9.3)
    for key,table in PROTOCOL['physical_sensitivity'].items():
        if not isinstance(table,dict) or 'P_W' not in table:continue
        # L_pu = omega*L / (3*V_phase_RMS²/P_threephase), explicitly inferred branch.
        Zbase=3*table['V_table_V']**2/table['P_W']
        scale=2*np.pi*table['f_Hz']/Zbase
        one=[x*scale for x in table['typeI_H']]
        two=[x*scale for x in table['typeII_H']]
        name='table_SI_simulation' if key=='simulation_table' else 'table_SI_experiment'
        branches[name]=evaluate_branch(name,one,two,table['Ki']*table['P_W'])
        branches[name]['interpretation']={'voltage':'phase RMS, inferred from approximate agreement of inductance bases',
                                         'Zbase_ohm':Zbase,'Lbase_H':Zbase/(2*np.pi*table['f_Hz']),
                                         'typeI_recomputed_pu':one,'typeII_recomputed_pu':two,
                                         'status':'SI/table-rounding sensitivity, not substituted for main stated-pu protocol'}
    main=branches['stated_pu'];g=PROTOCOL['gates']
    targets={'typeI_initial_deg':abs(main['typeI']['initial_angle_deg']-26.4)<g['equilibrium_angle_vs_report_deg'],
             'typeI_post_deg':abs(main['typeI']['post_sep_angle_deg']-60.5)<g['equilibrium_angle_vs_report_deg'],
             'CCA_deg':abs(main['typeII']['CCA_deg']-108)<g['CCA_vs_report_deg'],
             'CCT_s':abs(main['typeII']['CCT_s']-.58)<g['CCT_vs_report_s']}
    sensitivity=[]
    for C in [.3,.5,.8]:
        n=network(.8,.15,.8,C)
        d0=np.radians(main['typeII']['initial_angle_deg']);cca=np.radians(main['typeII']['CCA_deg'])
        sensitivity.append({'Xgnd_pu':C,'fault_Pmax_pu':n['Pmax_pu'],'CCA_deg':main['typeII']['CCA_deg'],
                            'CCT_s':travel_time(9.3,9.3*n['Pmax_pu'],d0,cca)})
    # Shared exact sampling grid for subsequent cross-implementation comparison.
    # Read only this verifier's generated files; no primary output is consulted.
    full_time=np.round(np.linspace(2,7,5001),12)
    after=full_time>=3
    packed={'time_s':full_time}
    for branch in ['stated_pu','table_SI_simulation']:
        result_branch=branches[branch]
        specifications=[('type_i_line_open',result_branch['typeI']['trajectory_file'],result_branch['typeI']['pre_network'],np.radians(result_branch['typeI']['initial_angle_deg']))]
        for label,case in result_branch['typeII']['cases'].items():
            name={'early':'type_ii_clear_0p5','late':'type_ii_clear_0p7','uncleared':'type_ii_uncleared'}[label]
            specifications.append((name,case['trajectory_file'],result_branch['typeII']['pre_network'],np.radians(result_branch['typeII']['initial_angle_deg'])))
        for scenario,path,pre_net,d0 in specifications:
            with np.load(HERE/path) as data:
                source_time=data['time_since_disturbance_s'] if 'time_since_disturbance_s' in data else data['time_since_fault_s']
                indices=np.rint((full_time[after]-3)/(source_time[1]-source_time[0])).astype(int)
                if np.max(abs(source_time[indices]-(full_time[after]-3)))>1e-12:raise RuntimeError('Shared-grid alignment would require interpolation')
                for field,source_field,initial in [('angle_rad','delta_unwrapped_rad',d0),('power_pu','power_pu',1.),('current_pu','current_magnitude_pu',float(current(d0,pre_net)))]:
                    values=np.full(len(full_time),initial)
                    values[after]=data[source_field][indices]
                    packed[f'{branch}_{scenario}_{field}']=values
    np.savez_compressed(HERE/'trajectories.npz',**packed)
    result={'paper_doi':PROTOCOL['paper_doi'],'independence':'Only paper and independently frozen protocol used; no primary/other-agent implementation or derivation inspected',
            'source_sha256':PROTOCOL['source_sha256'],'protocol_sha256':digest(HERE/'protocol.json'),
            'implementation_sha256':digest(__file__),'branches':branches,'reported_main_targets_passed':targets,
            'fault_severity_sensitivity':sensitivity,
            'all_internal_numeric_checks_passed':all(x['internal_numeric_checks_passed'] for x in branches.values()),
            'all_reported_main_targets_passed':all(targets.values()),
            'scope_findings':['No-current-limiter assumption is audited by trajectory current; an exceeded 1.8 pu threshold is not simulated as a hardware controller switch.',
                              'CCA is fixed for a specified post-fault circuit; CCT also depends on fault-on dynamics and initial angle.',
                              'An unwrapped angle is monotone in the no-equilibrium fault case; plotted sawtooth/wrapped angle is not an oscillation of the unwrapped scalar state.',
                              'The real arctangent primitive needs a²>b² and branch care; Type I uses a different real representation.',
                              'A fixed-angle phase-slip re-capture in the ideal scalar equation does not establish continued PSC operation when current limiting should activate.'],
            'cross_comparison_export':{'file':'trajectories.npz','time_key':'time_s','time_grid':'round(linspace(2,7,5001),12)',
                                       'key_pattern':'{branch}_{scenario}_{field}', 'branches':['stated_pu','table_SI_simulation'],
                                       'scenarios':['type_i_line_open','type_ii_clear_0p5','type_ii_clear_0p7','type_ii_uncleared'],
                                       'fields':['angle_rad','power_pu','current_pu'],'event_convention':'right-continuous network and output at t=3,3.5,3.7; continuous unwrapped angle; no interpolation'},
            'execution':{'measured_numeric_script_wall_s':time.perf_counter()-start,'python':sys.version,'platform':platform.platform(),
                         'numpy':np.__version__,'scipy':scipy.__version__,'matplotlib':matplotlib.__version__}}
    (HERE/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'internal_passed':result['all_internal_numeric_checks_passed'],'reported_targets_passed':result['all_reported_main_targets_passed'],
                      'typeI':{k:main['typeI'][k] for k in ['initial_angle_deg','post_sep_angle_deg']},
                      'typeII':{k:main['typeII'][k] for k in ['CCA_deg','CCT_s','uncleared_slip_period_s']},
                      'scope_exits':{k:v['first_current_limit_since_fault_s'] for k,v in main['typeII']['cases'].items()}},indent=2))
    return 0 if result['all_internal_numeric_checks_passed'] and result['all_reported_main_targets_passed'] else 1

if __name__=='__main__':raise SystemExit(main())
