"""Figures from exported fixed-frequency trajectories and the frequency schedule."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


LABEL = {'be': ('#147257', '-', 'DMM + BE with blocking'), 'be_pairs': ('#2b5ba1', '--', 'BE, conducting pairs'),
         'fe_pairs': ('#b94b2e', '-.', 'FE, conducting pairs')}


def save(fig, path):
    fig.savefig(path, dpi=175, metadata={'Software': 'rebuildIEEE LLC DMM'})
    plt.close(fig)


def axes_style(ax, ylabel=None):
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(alpha=0.2)


def block_spans(ax, wave, zero):
    x = (wave[:, 0] - zero) * 1e-6
    blocked = wave[:, 7] == 0
    edges = np.flatnonzero(np.diff(np.r_[False, blocked, False]))
    for i, j in edges.reshape(-1, 2):
        ax.axvspan(x[i], x[min(j, len(x) - 1)], color='#d5e9dd', alpha=0.85, linewidth=0)


def show_current(out, waves):
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 6.3), layout='constrained')
    fig.suptitle('LLC resonant current at 15 ns on the same circuit and clock')
    for row, (case, title, period_ps) in enumerate((('set1_100000', 'Set #1, 100 kHz assumption', 10_000_000),
                                                     ('set2_312500', 'Set #2, 312.5 kHz', 3_200_000))):
        full = waves[f'{case}_be']
        first = 47_900_000_000
        current_clock = full[(full[:, 0] >= first) & (full[:, 0] <= first + 3 * period_ps)]
        blocked = current_clock[:, 7] == 0
        borders = np.flatnonzero(np.diff(np.r_[False, blocked, False])).reshape(-1, 2)
        i, j = next((i, j) for i, j in borders if i > 0 and j < len(blocked))
        event_start = current_clock[i, 0]
        event_end = current_clock[j, 0]
        middle = event_start + (event_end - event_start) / 2
        window = [(first, first + 2 * period_ps), (middle - 0.46 * period_ps, middle + 0.46 * period_ps)]
        for col, (a, b) in enumerate(window):
            ax = axes[row, col]
            for method, (color, style, label) in LABEL.items():
                wave = waves[f'{case}_{method}']
                selected = wave[(wave[:, 0] >= a) & (wave[:, 0] <= b)]
                ax.plot((selected[:, 0] - a) * 1e-6, selected[:, 1], style, color=color, label=label, linewidth=1.4)
                if method == 'be':
                    block_spans(ax, selected, a)
            ax.set_title(title if col == 0 else 'Detail around a blocked interval')
            axes_style(ax, 'Resonant current (A)')
            ax.set_xlabel('Time in window (µs)')
    axes[0, 0].legend(frameon=False, fontsize=8, loc='upper right')
    fig.supxlabel('Green shading marks the blocked state in the full BE solution.', fontsize=9)
    save(fig, out / 'rectifier_blocking.png')


def show_refinement(out, metrics, mains):
    fig, axes = plt.subplots(1, len(mains), figsize=(11.6, 3.6), sharey=True, layout='constrained')
    fig.suptitle('Resonant-current error against the 1.875 ns BE run')
    for ax, main in zip(axes, mains, strict=True):
        label = main.removesuffix('_be')
        for clock, color, marker in (('native', '#b94b2e', 'o'), ('locked', '#147257', 's')):
            steps = (7.5, 15, 30) if clock == 'native' else (3.75, 7.5, 15)
            parts = [f'{int(step * 1000)}ps' if step != 15 else '15ns' for step in steps]
            names = [f'{label}_{clock}_{part}' for part in parts]
            y = [metrics['comparisons'][key]['ir_nrmse_pct'] for key in names]
            ax.plot(steps, y, marker=marker, color=color,
                    label='Native switching clock' if clock == 'native' else '15 ns gate times held fixed')
        row = metrics['runs'][main]
        ax.set_title(f"{row['set'].replace('set', 'Set #')}, {row['frequency_hz'] / 1000:g} kHz")
        ax.set_xticks([3.75, 7.5, 15, 30])
        ax.set_xlabel('Solver step (ns)')
        axes_style(ax)
    axes[0].set_ylabel('Resonant-current NRMSE (%)')
    axes[0].legend(frameon=False, fontsize=8)
    save(fig, out / 'step_size.png')


def show_schedule(out, wave):
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.55), layout='constrained')
    fig.suptitle('Set #2 frequency schedule at nominal load, 25 ns DMM')
    t = wave[:, 0] * 1e-12
    axes[0].plot(t, wave[:, 4], color='#147257', linewidth=1.4)
    axes[0].axvspan(0.25, 0.48, color='#d5e9dd', alpha=0.85, linewidth=0, label='312.5 to 500 kHz ramp')
    axes[0].set(xlabel='Time (s)', xlim=(0, 0.6))
    axes[0].legend(frameon=False, fontsize=8)
    axes_style(axes[0], 'Output voltage (V)')
    m = np.abs(t - 0.55) <= 4e-6
    axes[1].plot((t[m] - 0.55) * 1e6, wave[m, 1], color='#2b5ba1', linewidth=1.4)
    axes_style(axes[1], 'Resonant current (A)')
    axes[1].set_xlabel('Time relative to 0.55 s at 500 kHz (µs)')
    save(fig, out / 'frequency_schedule.png')


def figures(out, waves, metrics, mains):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titlesize': 10})
    show_current(out, waves)
    show_refinement(out, metrics, mains)
    show_schedule(out, waves['set2_section_iv_d_nominal_load'])
