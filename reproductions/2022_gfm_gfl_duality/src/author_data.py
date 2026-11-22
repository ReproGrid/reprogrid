"""Read author MAT traces and calibrated dominant root-locus markers."""

import numpy as np
from PIL import Image
from scipy.io import loadmat
from scipy.optimize import least_squares


COLORS = ((0,114,189),(217,83,25),(237,177,32),(126,47,142),(119,172,48),(77,190,238),(162,20,47))


def raw_trace(path):
    values=loadmat(path,simplify_cells=True)[path.stem]
    output={row["label"].strip("<>"):row["values"] for row in values["signals"]}
    return values["time"],output


def marker_readback(path, kind):
    image=np.asarray(Image.open(path).convert("RGB"))
    if image.shape[:2] != (1281,1359):
        raise ValueError("Author axis calibration only supports the original 1359x1281 PNG.")
    bounds=(-20,10) if kind=="GFM" else (-10,5)
    points=[]
    for color in COLORS:
        y,x=np.where(np.all(image[760:882,650:1100]==color,axis=2))
        if len(x)<40:
            points.append(None)
            continue
        x+=650
        y+=760
        def diagonal(coordinates):
            positions,counts=np.unique(coordinates,return_counts=True)
            dense=counts>=0.6*counts.max()
            return np.average(positions[dense],weights=counts[dense])
        up,down=diagonal(x+y),diagonal(x-y)
        center_x,center_y=(up+down)/2,(up-down)/2
        real=bounds[0]+(bounds[1]-bounds[0])*(center_x-281)/(1229-281)
        imag=40-80*(center_y-749)/(1063-749)
        points.append({"real_hz":float(real),"imag_hz":float(imag),"colored_pixels":len(x)})
    return points


def display_frequency(wpu, step, stages):
    """Author display LPFs are forward-Euler transfers with one delay."""
    result=wpu.copy()
    a=step*2*np.pi*100
    for _ in range(stages):
        out=np.empty_like(result)
        state=1.0
        for k,value in enumerate(result):
            out[k]=state
            state+=a*(value-state)
        result=out
    return result


def replay_synchronization_loop(key, time, signals, change_s, restore_s):
    """Replay the published frequency loop on recorded measurement inputs."""
    step=time[1]-time[0]
    omega=2*np.pi*50
    filtered=0.0
    integral=0.0
    frequency=np.empty(len(time))
    for k,t in enumerate(time):
        scale=4.0 if key in ("GFM_m","GFL_w_pll") and change_s-1e-12<=t<restore_s-1e-12 else 1.0
        if key.startswith("GFM"):
            frequency[k]=1-0.05*scale*filtered
            filtered+=step*2*np.pi*15*(signals["igdq"][k,0]-filtered)
        else:
            kp=2*np.pi*15*scale
            ki=(2*np.pi*15)**2*scale**2/4
            frequency[k]=1+(kp*filtered+integral)/omega
            integral+=step*ki*filtered
            filtered+=step*2*np.pi*1000*(signals["vcdq"][k,1]-filtered)
    stages=1 if key=="GFL_w_pll" else (2 if key=="GFL_w_i" else 0)
    return display_frequency(frequency,step,stages)


def mode_fit(time, frequency_pu, window, initial_hz):
    """Estimate a mode from one published trace, without modifying the model."""
    mask=(time>=window[0]) & (time<window[1])
    t=time[mask]-window[0]
    y=frequency_pu[mask]-1
    scale=max(np.max(abs(y)),1e-14)
    if len(y)<100 or scale<1e-10:
        return None
    y=y/scale
    def solve_linear(parameters):
        sigma,hz=parameters
        carrier=np.exp(sigma*t)
        matrix=np.column_stack((carrier*np.cos(2*np.pi*hz*t),carrier*np.sin(2*np.pi*hz*t),np.ones_like(t),t))
        coef=np.linalg.lstsq(matrix,y,rcond=None)[0]
        return matrix@coef-y,coef
    answer=least_squares(lambda v:solve_linear(v)[0],np.array([0.0,initial_hz]),
                         bounds=([-200,max(2,initial_hz*0.5)],[150,min(100,initial_hz*1.5)]),
                         ftol=1e-11,xtol=1e-11,gtol=1e-11,max_nfev=250)
    error,_=solve_linear(answer.x)
    return {"real_per_s":float(answer.x[0]),"imag_hz":float(answer.x[1]),
            "relative_residual":float(np.linalg.norm(error)/np.linalg.norm(y-y.mean())),
            "window_s":list(window),"signal_peak_pu":float(scale)}
