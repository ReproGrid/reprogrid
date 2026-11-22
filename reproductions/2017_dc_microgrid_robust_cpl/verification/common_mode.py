"""Check the homogeneous mode of the published nine-bus DC microgrid."""

from argparse import ArgumentParser
from datetime import datetime,timezone
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from numpy.polynomial import Polynomial
from scipy.optimize import brentq


PARAMETERS={"vref_v":400.0,"rs_ohm":0.05,"rl_ohm":0.05,"ls_h":0.0009,"ll_h":0.0009,
            "cb_f":0.00075,"cl_f":0.0007,"p_min_w":5000.0,"p_max_w":20000.0}


def operating_point(power,droop):
    p=PARAMETERS
    resistance=p["rs_ohm"]+droop+p["rl_ohm"]
    voltage=(p["vref_v"]+np.sqrt(p["vref_v"]**2-4*resistance*power))/2
    current=power/voltage
    bus_voltage=voltage+p["rl_ohm"]*current
    return current,bus_voltage,voltage


def jacobian(power,droop):
    p=PARAMETERS
    current,vbus,voltage=operating_point(power,droop)
    g=power/voltage**2
    ls,ll,cb,cl=(p[k] for k in ("ls_h","ll_h","cb_f","cl_f"))
    return np.array([[-(p["rs_ohm"]+droop)/ls,0,-1/ls,0],
                     [0,-p["rl_ohm"]/ll,1/ll,-1/ll],
                     [1/cb,-1/cb,0,0],[0,1/cl,0,g/cl]])


def impedance_characteristic(power,droop):
    p=PARAMETERS
    _,_,voltage=operating_point(power,droop)
    s=Polynomial([0,1])
    source=p["rs_ohm"]+droop+p["ls_h"]*s
    load=p["rl_ohm"]+p["ll_h"]*s
    shunt=p["cl_f"]*s-power/voltage**2
    return (p["cb_f"]*s*source+1)*(1+load*shunt)+source*shunt


def record(power,droop):
    modes=np.linalg.eigvals(jacobian(power,droop))
    independent=impedance_characteristic(power,droop).roots()
    error=max(min(abs(mode-independent)) for mode in modes)
    lead=max((z for z in modes if z.imag>=0),key=lambda z:z.real)
    current,vbus,voltage=operating_point(power,droop)
    return {"power_per_bus_w":power,"droop_ohm":droop,"load_voltage_v":voltage,"bus_voltage_v":vbus,
            "current_a":current,"rightmost_real_per_s":float(lead.real),"frequency_hz":float(lead.imag/2/np.pi),
            "independent_impedance_max_error_rad_s":float(error)}


def main():
    parser=ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=Path(__file__).resolve().parents[1]/"results/common_mode.json")
    args=parser.parse_args()
    started=datetime.now(timezone.utc).isoformat();start=perf_counter()
    cases=[record(p,d) for d in (0.06,0.1,0.2) for p in (5000.0,15000.0,16000.0,20000.0)]
    boundary=brentq(lambda p:max(np.linalg.eigvals(jacobian(p,.06)).real),15000,20000,xtol=1e-8)
    boundary_row=record(boundary,.06)
    sweep=[record(float(p),.2) for p in np.linspace(5000,20000,301)]
    checks={"independent_equations_match":max(c["independent_impedance_max_error_rad_s"] for c in cases+sweep)<1e-7,
            "published_15kw_common_mode_stable":record(15000,.06)["rightmost_real_per_s"]<0,
            "published_16kw_near_threshold":abs(boundary-16000)<500,
            "published_20kw_weak_droop_common_mode_unstable":record(20000,.06)["rightmost_real_per_s"]>0,
            "droop_0p2_common_mode_stable_over_sampled_load_range":max(c["rightmost_real_per_s"] for c in sweep)<0}
    payload={"doi":"10.1109/TPWRS.2017.2697765","scope":"Exactly homogeneous load and bus states. Interbus resistive currents vanish in the original connected topology.",
             "parameters":PARAMETERS,"cases":cases,"droop_0p06_common_mode_boundary":boundary_row,
             "droop_0p2_power_sweep":sweep,"checks":checks,"source_sha256":sha256(Path(__file__).read_bytes()).hexdigest(),
             "started_utc":started,"finished_utc":datetime.now(timezone.utc).isoformat(),"wall_seconds":perf_counter()-start,
             "pending":"Full 36-state network, mixed load uncertainty and the robust SDP feasibility problem remain to run."}
    args.out.write_text(json.dumps(payload,indent=2)+"\n")
    print('Common-mode critical power',boundary_row)
    print(checks)
    return not all(checks.values())


if __name__=="__main__":
    raise SystemExit(main())
