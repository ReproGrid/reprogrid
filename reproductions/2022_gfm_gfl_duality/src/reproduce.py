"""Reproduce the six single-inverter duality cases with source-locked inputs."""

from argparse import ArgumentParser
from dataclasses import replace
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import platform
from time import perf_counter

import matplotlib
import numpy as np
import scipy
from scipy.optimize import brentq, linear_sum_assignment

from author_data import display_frequency, marker_readback, mode_fit, raw_trace, replay_synchronization_loop
from linear_models import dominant_mode, from_protocol, gfl_nodal_matrix, gfm_state_matrix, mode_record, roots
from sampled_model import SampledCircuit, simulate
from plots import make_figures


ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+"\n")


def sampled_circuit(config, row, step):
    kind=row["kind"]
    p=config[kind.lower()]
    rf=p["filter_r_simulation_pu"] if kind=="GFM" else p["filter_r_gain_cases_simulation_pu"]
    extra=0.0
    breaker=1e6
    x=p["grid_x_pu"]
    if row["key"]=="GFM_Z":
        x,extra=0.1,0.2
    if row["key"]=="GFL_Y":
        extra,breaker=0.2,0.01
        rf=p["filter_r_grid_case_simulation_pu"]
    base=SampledCircuit(kind,row["key"],step,rf,x,extra_x=extra,grid_breaker=breaker,
                        wi_hz=p["current_index_hz"],wv_hz=p.get("voltage_index_hz",250))
    if extra:
        changed=replace(base,grid_breaker=config["time_domain"]["grid_breaker_r_pu"] if kind=="GFM" else 1e6)
    else:
        changed=base
    return base,changed


def analytic_cases(config, checks):
    loci=[]
    for row in config["sweeps"]:
        p=from_protocol(config,row["kind"])
        samples=[]
        plot=marker_readback(ROOT/"sources"/("Case2_"+row["key"]+".png"),row["kind"])
        for value,point in zip(np.linspace(row["start"],row["stop"],row["count"]),plot,strict=True):
            at=replace(p,**{row["field"]:float(value)})
            values=roots(at)
            lead=dominant_mode(values)
            if p.kind=="GFM":
                independent=np.linalg.eigvals(gfm_state_matrix(at))
                distance=abs(values[:,None]-independent[None,:])
                a,b=linear_sum_assignment(distance)
                error=float(np.max(distance[a,b]))
                passed=error<config["acceptance"]["gfm_cross_path_max_rad_s"]
                key="dq_cross_implementation_error_rad_s"
            else:
                # Check each oscillatory root directly in the original
                # rational MIMO nodal matrix, avoiding polynomial clearing.
                errors=[]
                for eigenvalue in values[values.imag>1e-6]:
                    matrix=gfl_nodal_matrix(at,eigenvalue/at.base_rad_s)
                    errors.append(abs(np.linalg.det(matrix))/np.linalg.norm(matrix)**2)
                error=float(max(errors))
                passed=error<config["acceptance"]["polynomial_matrix_normalized_residual"]
                key="direct_rational_nodal_root_residual"
            checks[f"{row['key']}_{value:g}_equations"]=passed
            plot_comparison=None
            if point:
                plot_comparison=point|{"real_delta_hz":float(abs(point["real_hz"]-lead.real/(2*np.pi))),
                                       "imag_delta_hz":float(abs(point["imag_hz"]-lead.imag/(2*np.pi)))}
                checks[f"{row['key']}_{value:g}_author_marker"]=(
                    plot_comparison["real_delta_hz"]<config["acceptance"]["plot_real_hz"]
                    and plot_comparison["imag_delta_hz"]<config["acceptance"]["plot_imag_hz"])
            else:
                checks[f"{row['key']}_{value:g}_author_marker"]=False
            samples.append({"parameter":float(value),"dominant":mode_record(lead),
                            "all_modes":[mode_record(z) for z in values],key:error,"author_marker":plot_comparison})
        def rightmost(value):
            return dominant_mode(roots(replace(p,**{row["field"]:value}))).real
        marginal=brentq(rightmost,min(row["start"],row["stop"]),max(row["start"],row["stop"]),xtol=1e-11)
        mode=dominant_mode(roots(replace(p,**{row["field"]:marginal})))
        stop=replace(p,**{row["field"]:row["stop"]})
        alternative=None
        if p.kind=="GFL":
            alternative=mode_record(dominant_mode(roots(stop,literal_off_diagonal=False)))
        loci.append(row|{"samples":samples,"marginal_parameter":marginal,"marginal_mode":mode_record(mode),
                         "gfl_divided_off_diagonal_sensitivity":alternative})
    return loci


def sampled_modes(circuit, scale):
    state,_=circuit.initial()
    mapping=circuit.propagator()
    delta=1e-6
    basis=np.eye(len(state))
    def columns(epsilon):
        return np.column_stack([(circuit.advance(state+epsilon*e,mapping,scale=scale)[0]
                                 -circuit.advance(state-epsilon*e,mapping,scale=scale)[0])/(2*epsilon)
                                for e in basis])
    jacobian=columns(delta)
    modes=np.log(np.linalg.eigvals(jacobian).astype(complex))/circuit.step
    fine=columns(delta/2)
    low_modes=modes[(modes.imag>1e-5) & (modes.imag<2*np.pi*100)]
    return {"all_modes":[mode_record(z) for z in modes],
            "dominant":mode_record(dominant_mode(modes)),
            "synchronization_mode":mode_record(dominant_mode(low_modes)),
            "jacobian_refinement_max":float(np.max(abs(jacobian-fine)))}


def time_cases(config,loci,checks):
    out={}
    all_arrays={}
    td=config["time_domain"]
    for row,locus in zip(config["sweeps"],loci,strict=True):
        key=row["key"]
        step=td["sample_step_s"]
        base,changed=sampled_circuit(config,row,step)
        time=td["start_s"]+np.arange(round((td["stop_s"]-td["start_s"])/step)+1)*step
        primary,equilibrium_residual=simulate(base,changed,time,td["change_s"],td["restore_s"],td["load_impulse_duration_s"])
        event_be,_=simulate(base,changed,time,td["change_s"],td["restore_s"],td["load_impulse_duration_s"],backward_euler_events=True)
        at,author=raw_trace(ROOT/"sources"/("Case2_"+key+".mat"))
        idx=np.rint((time-at[0])/(at[1]-at[0])).astype(int)
        w_author=author["w"][idx]
        selected_author={key:value[idx] for key,value in author.items()}
        loop_replay=replay_synchronization_loop(key,time,selected_author,td["change_s"],td["restore_s"])
        loop_max_hz=float(50*np.max(abs(loop_replay-w_author)))
        checks[key+"_synchronization_loop_replayed_on_recorded_inputs"]=loop_max_hz<1e-7
        stages=1 if key=="GFL_w_pll" else (2 if key=="GFL_w_i" else 0)
        w_primary=display_frequency(primary[:,6],step,stages)
        w_be=display_frequency(event_be[:,6],step,stages)
        freq_error=50*(w_primary-w_author)
        signal=50*(w_author-1)
        equilibria={name:float(np.max(abs(primary[0,slice(pos,pos+2)]-author[name][idx[0]])))
                    for pos,name in ((0,"ildq"),(2,"vcdq"),(4,"igdq"))}
        checks[key+"_sampled_equilibrium_residual"]=equilibrium_residual<1e-8
        checks[key+"_author_equilibrium_dq"]=max(equilibria.values())<5e-6
        final_hz=locus["samples"][-1]["dominant"]["imag_hz"]
        trace_fit=mode_fit(time,w_author,(2.04,2.18),final_hz)
        baseline_fit=mode_fit(time,w_author,(2.28,2.55),locus["samples"][0]["dominant"]["imag_hz"])
        detail={"initial_dq_error_pu":equilibria,"equilibrium_residual":equilibrium_residual,
                "author_frequency_min_max_hz":[float(50*w_author.min()),float(50*w_author.max())],
                "reconstructed_frequency_min_max_hz":[float(50*w_primary.min()),float(50*w_primary.max())],
                "event_be_frequency_min_max_hz":[float(50*w_be.min()),float(50*w_be.max())],
                "raw_frequency_nrmse_pct":float(100*np.linalg.norm(freq_error)/max(np.linalg.norm(signal),1e-20)),
                "recorded_input_synchronization_loop_max_error_hz":loop_max_hz,
                "author_fitted_unstable_mode":trace_fit,"author_fitted_restored_mode":baseline_fit,
                "simulation_assessment":"Original TBE event histories are unresolved. Quantitative trajectory matching remains open.",
                "sampled_dynamics":[]}
        ratio={"GFM_m":4.0,"GFL_w_pll":4.0,"GFM_w_v":0.6,"GFL_w_i":0.6}.get(key,1.0)
        for method,h in (("trapezoidal",1e-4),("trapezoidal",5e-5),("exact",1e-4),("exact",2.5e-5)):
            b,ch=sampled_circuit(config,row,h)
            for phase,plant,scale in (("baseline",b,1.0),("changed",ch,ratio)):
                c=replace(plant,integrator=method)
                modes=sampled_modes(c,scale)
                detail["sampled_dynamics"].append({"integrator":method,"step_s":h,"phase":phase}|modes)
                checks[f"{key}_{method}_{h}_{phase}_jacobian_refinement"]=modes["jacobian_refinement_max"]<1e-6
        out[key]=detail
        all_arrays[key+"_time_s"]=time
        all_arrays[key+"_model_dq_w_delta_vabs"]=primary
        all_arrays[key+"_event_be_dq_w_delta_vabs"]=event_be
        all_arrays[key+"_author_vcdq"]=author["vcdq"][idx]
        all_arrays[key+"_author_ildq"]=author["ildq"][idx]
        all_arrays[key+"_author_igdq"]=author["igdq"][idx]
        all_arrays[key+"_author_w_pu"]=w_author
        all_arrays[key+"_model_w_display_pu"]=w_primary
        all_arrays[key+"_be_event_w_display_pu"]=w_be
        all_arrays[key+"_recorded_input_loop_w_pu"]=loop_replay
    return out,all_arrays


def main():
    parser=ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"results")
    parser.add_argument("--figures",type=Path)
    args=parser.parse_args()
    figures=args.figures or args.output/"figures"
    args.output.mkdir(parents=True,exist_ok=True)
    figures.mkdir(parents=True,exist_ok=True)
    begin=perf_counter()
    started=datetime.now(timezone.utc).isoformat()
    config=json.loads((ROOT/"configs/protocol.json").read_text())
    checks={}
    loci=analytic_cases(config,checks)
    time_results,arrays=time_cases(config,loci,checks)
    metrics={"doi":config["doi"],"loci":loci,"time_cases":time_results,"checks":checks}
    write_json(args.output/"metrics.json",metrics)
    np.savez_compressed(args.output/"waveforms.npz",**arrays)
    figure_files=make_figures(metrics,arrays,figures)
    write_json(args.output/"run_manifest.json",{
        "doi":config["doi"],"started_utc":started,"finished_utc":datetime.now(timezone.utc).isoformat(),
        "wall_seconds":perf_counter()-begin,
        "environment":{"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__,"matplotlib":matplotlib.__version__,"platform":platform.platform()},
        "protocol_sha256":digest(ROOT/"configs/protocol.json"),
        "implementation_sha256":{p.name:digest(p) for p in sorted((ROOT/"src").glob("*.py")) if not p.name.startswith("._")},
        "data_sha256":{name:digest(args.output/name) for name in ("metrics.json","waveforms.npz")},
        "figures_sha256":{p.name:digest(p) for p in figure_files},
        "checks_passed":sum(checks.values()),"checks_total":len(checks),
    })
    print(f"{sum(checks.values())}/{len(checks)} numerical checks passed.")
    failed=[key for key,value in checks.items() if not value]
    if failed:print("Failed checks",failed)
    return bool(failed)


if __name__=="__main__":
    raise SystemExit(main())
