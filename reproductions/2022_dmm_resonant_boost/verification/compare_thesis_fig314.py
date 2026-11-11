"""Compare the frozen replay with the published vector plot, without phase fitting."""
from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pymupdf

ROOT = Path(__file__).resolve().parents[1]
PDF_HASH = '3084bda44193154c10d00d423dd98a78d8744b5e4a0b14a2de7dea0da639fa8b'


def main():
    ap = ArgumentParser(description=__doc__)
    ap.add_argument('--pdf',type=Path,default=ROOT/'sources/chalangar_2021_thesis.pdf')
    ap.add_argument('--results',type=Path,default=ROOT/'results')
    ap.add_argument('--figures',type=Path,default=ROOT/'figures')
    args = ap.parse_args()
    if sha256(args.pdf.read_bytes()).hexdigest() != PDF_HASH:
        raise ValueError('Fig. 3.14 calibration is specific to the inspected PDF.')
    with pymupdf.open(args.pdf) as doc:
        drawings = {d['seqno']:d for d in doc[64].get_drawings()}
    # Fig. 3.14, printed page 44, PDF page 65. X ticks are 2 µs
    # apart; y ticks cover -3 to 3 A. Drawing IDs are pinned by PDF hash.
    xloc = [drawings[j]['rect'].x0 for j in range(36,45)]
    yloc = [drawings[j]['rect'].y0 for j in range(57,64)]
    tx = np.arange(37006.,37023.,2.)  # μs
    iy = np.arange(-3.,4.)
    cx = np.polyfit(xloc,tx,1)
    cy = np.polyfit(yloc,iy,1)

    curve = np.array([(q[-1].x,q[-1].y) for q in drawings[72]['items']])
    circles = [drawings[j]['rect'] for j in range(74,210,2)]
    markers = np.array([((r.x0+r.x1)/2,(r.y0+r.y1)/2) for r in circles])
    tc,ic = np.polyval(cx,curve[:,0]),np.polyval(cy,curve[:,1])
    tm,im = np.polyval(cx,markers[:,0]),np.polyval(cy,markers[:,1])
    assert np.all(np.diff(tm)>0) and np.all(np.diff(tc)>0)
    with np.load(args.results/'rbc_100ns.npz') as trace:
        t = trace['t_ns']/1000
        i = trace['dmm'][:,1]
    clip = (t>=tm[0]) & (t<=tm[-1])
    i_at_marker = np.interp(tm,t,i)
    i_curve_at_marker = np.interp(tm,tc,ic)
    i_at_curve = np.interp(tc,t,i)
    discrepancy = lambda a,b: float(np.linalg.norm(a-b)/np.linalg.norm(b))
    result = {
        'source':'Chalangar (2021), Fig. 3.14, PDF page 65, EMTP reference curve and DMM dots',
        'source_url':'https://publications.polymtl.ca/9473/','source_pdf_sha256':PDF_HASH,
        'pymupdf_version':pymupdf.VersionBind,'reference_vertices':len(tc),'dmm_markers':len(tm),
        'calibration_max_residual_us':float(np.max(np.abs(np.polyval(cx,xloc)-tx))),
        'calibration_max_residual_a':float(np.max(np.abs(np.polyval(cy,yloc)-iy))),
        'median_reference_vertex_interval_ns':float(1000*np.median(np.diff(tc))),
        'median_dmm_marker_interval_ns':float(1000*np.median(np.diff(tm))),
        'replay_vs_published_dmm_at_dots_relative_l2':discrepancy(i_at_marker,im),
        'replay_vs_published_emtp_curve_relative_l2':discrepancy(i_at_curve,ic),
        'published_dmm_vs_published_emtp_at_dots_relative_l2':discrepancy(im,i_curve_at_marker),
        'replay_min_max_a':[float(i[clip].min()),float(i[clip].max())],
        'published_dmm_min_max_a':[float(im.min()),float(im.max())],
        'published_emtp_min_max_a':[float(ic.min()),float(ic.max())],
        'scope':'Vector plot readback at the labeled time coordinates. No time shift, gain fit, or original raw traces. Plot point spacing does not identify author solver step or decimation.'
    }
    # Store only the aligned DMM-dot comparison; the underlying source PDF
    # stays separate from the scientific handoff.
    np.savetxt(args.results/'fig314_comparison.csv',np.column_stack((tm,im,i_at_marker,i_curve_at_marker)),
               delimiter=',',header='time_us,thesis_dmm_A,replay_dmm_interpolated_A,thesis_EMTP_readback_A',
               comments='',fmt='%.9g')
    (args.results/'fig314_comparison.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    fig,ax=plt.subplots(figsize=(9.5,3.8),layout='constrained')
    ax.plot(tc-37000,ic,color='#b7744f',lw=1.4,label='Thesis Fig. 3.14 — EMTP readback')
    ax.scatter(tm-37000,im,facecolors='none',edgecolors='#ab5532',s=22,zorder=3,label='Thesis Fig. 3.14 — DMM markers')
    ax.plot(t[clip]-37000,i[clip],color='#126b8f',lw=1.3,ls='--',label='Independent RBC replay, 100 ns')
    ax.grid(alpha=.25)
    ax.legend(loc='upper left',ncol=2,fontsize=8.6)
    ax.set(xlabel='Time relative to 37 ms (μs)',ylabel=r'Resonant current $i_r$ (A)',
           title='Published plot and offline replay at the original time coordinates',ylim=(-3.3,4.7))
    fig.savefig(args.figures/'fig314_comparison.png',dpi=190)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
