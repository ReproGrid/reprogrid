"""Generate publication-ready plots from the archived numerical results."""
from argparse import ArgumentParser
from pathlib import Path
import csv
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COLORS = {2: '#52525b', 4: '#02826e', 8: '#cf771a', 12: '#146895'}


def finish(fig, name, out):
    fig.savefig(out / name, dpi=190, facecolor='white')
    plt.close(fig)


def voltage_dip(data, out):
    fig, ax = plt.subplots(3, 2, figsize=(10.2, 7.8), layout='constrained', sharex='col')
    for col, eta in enumerate((.02, .06)):
        for order in (2, 4, 8, 12):
            prefix = f'case_i_eta_{eta:.3f}_order_{order}'
            t = data[f'{prefix}__t']
            x = data[f'{prefix}__x']
            for idx in (0, 1):
                ax[idx, col].plot(t, x[:, idx], color=COLORS[order], lw=1.2,
                                  label=f'{order}th order' if order != 2 else '2nd order',
                                  ls='-' if order in (2, 12) else '--')
        tracks = data[f'{prefix}__tracking_error']
        for j, label in enumerate(('dVOC', 'Grid current', 'Voltage loop', 'Current loop')):
            ax[2, col].plot(t, 100 * tracks[:, j], lw=1.2, label=label)
        ax[0, col].set_title(rf'$\eta/\omega_0={eta:.2f}$')
        ax[2, col].set_xlabel('Time (s)')
        for item in ax[:, col]:
            item.axvline(.5, color='#8a8a8a', lw=.85, ls=':')
            item.grid(alpha=.25)
    for j, label in enumerate((r'$\hat v_d$ (pu)', r'$\hat v_q$ (pu)', 'Normalized error (%)')):
        ax[j, 0].set_ylabel(label)
    ax[0, 0].legend(loc='upper right', fontsize=8)
    ax[2, 1].legend(loc='upper right', fontsize=8)
    fig.suptitle('Grid voltage dip | four model orders', fontsize=14)
    finish(fig, 'model_order_voltage_dip.png', out)


def stability_boundaries(data, rows, out):
    fig, ax = plt.subplots(2, 2, figsize=(10.3, 6.7), layout='constrained')
    for j, rg in enumerate((.08, .20)):
        matched = [r for r in rows if r['angle_branch'] == 'impedance_matched' and float(r['rg']) == rg]
        alpha = [float(r['alpha']) for r in matched]
        cutoff = [float(r['critical_eta_over_omega0'] or 'nan') for r in matched]
        certificate = [float(r['certificate_supremum'] or 'nan') for r in matched]
        a = ax[0, j]
        a.plot(alpha, cutoff, color='#b85c31', label='First eigenvalue crossing', lw=1.7)
        a.plot(alpha, certificate, color='#1b668a', label='Sufficient bound, Eq. (26a,b)', lw=1.7)
        a.fill_between(alpha, 0, certificate, color='#bedde9', alpha=.55)
        a.set(xlabel=r'$\alpha$ (pu)', ylabel=r'$\eta/\omega_0$', title=rf'$r_g={rg:.2f}$ pu')
        a.grid(alpha=.24)
        if j == 0:
            a.legend(fontsize=8.5)
    for j, eta in enumerate((.099, .101)):
        a = ax[1, j]
        for order in (2, 4):
            prefix = f'case_ii_eta_{eta:.3f}_order_{order}'
            t = data[f'{prefix}__t']
            x = data[f'{prefix}__x']
            view = t <= 3
            a.plot(t[view], np.linalg.norm(x[view, :2], axis=1), color=COLORS[order],
                   lw=1.2, label=f'{order} state')
        a.set(xlabel='Time (s)', ylabel=r'$|\hat{v}|$ (pu)', title=rf'$\eta/\omega_0={eta:.3f}$')
        a.axvline(2., color='#8a8a8a', lw=.85, ls=':')
        a.grid(alpha=.24)
        if j == 0:
            a.legend(fontsize=8.5)
    finish(fig, 'stability_boundary_and_response.png', out)


def phase_portraits(data, metrics, out):
    fig = plt.figure(figsize=(9.7, 7.1), layout='constrained')
    grid = fig.add_gridspec(2, 2, height_ratios=[.65, 1.])
    time_axis = fig.add_subplot(grid[0, :])
    for order in (2, 12):
        prefix = f'case_iii_alpha_3.0_order_{order}'
        t = data[f'{prefix}__t']
        x = data[f'{prefix}__x']
        view = t <= 4.
        time_axis.plot(t[view], np.linalg.norm(x[view, :2], axis=1), color=COLORS[order],
                       lw=1.2, ls='--' if order == 2 else '-', label=f'{order} state')
    time_axis.axvline(2., color='#8a8a8a', lw=.85, ls=':')
    time_axis.set(xlabel='Time (s)', ylabel=r'$|\hat v|$ (pu)', title=r'Voltage dip at $\alpha=3$')
    time_axis.legend(fontsize=9)
    time_axis.grid(alpha=.25)

    for j, alpha in enumerate((3., 1.)):
        a = fig.add_subplot(grid[1, j])
        key = f'phase_alpha_{alpha:.1f}__x'
        for trajectory in data[key]:
            a.plot(trajectory[:, 0], trajectory[:, 1], color='#679fb6', lw=.8, alpha=.7)
            a.scatter(*trajectory[0], s=9, color='#a75932')
        late = data[key][0, -270:]
        a.plot(late[:, 0], late[:, 1], color='#0a526e', lw=1.55)
        a.scatter(*metrics[f'case_iii_alpha_{alpha:.1f}_order_2']['post_equilibrium_v'],
                  marker='x', s=35, color='#27272a', zorder=4)
        a.set(xlabel=r'$\hat v_d$ (pu)', ylabel=r'$\hat v_q$ (pu)', title=rf'$\alpha={alpha:g}$')
        a.set_aspect('equal')
        a.grid(alpha=.24)
    finish(fig, 'limit_cycle_and_phase_portraits.png', out)


def main():
    ap = ArgumentParser(description=__doc__)
    ap.add_argument('--results', type=Path, default=ROOT / 'results')
    ap.add_argument('--figures', type=Path, default=ROOT / 'figures')
    args = ap.parse_args()
    args.figures.mkdir(parents=True, exist_ok=True)
    with (args.results / 'stability_sweep.csv').open() as handle:
        sweep = list(csv.DictReader(handle))
    metrics = json.loads((args.results / 'case_metrics.json').read_text())
    with np.load(args.results / 'trajectories.npz') as data:
        voltage_dip(data, args.figures)
        stability_boundaries(data, sweep, args.figures)
        phase_portraits(data, metrics, args.figures)


if __name__ == '__main__':
    main()
