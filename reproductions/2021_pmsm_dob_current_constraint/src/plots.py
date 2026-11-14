"""Scientific figures from the independently reconstructed trajectories."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


LABELS = {'dobc_high': 'High-gain DOBC', 'dobc_low': 'Low-gain DOBC', 'ccpid': 'CCPID',
          'dob_cc_eq16': 'DOB-CC, Eq. (16)', 'dob_cc_eq17': 'DOB-CC, Eq. (17)'}
COLORS = {'dobc_high': '#2066a2', 'dobc_low': '#6c7698', 'ccpid': '#bc7b19',
          'dob_cc_eq16': '#11765d', 'dob_cc_eq17': '#b34560'}
STYLES = {'dobc_high': '-', 'dobc_low': '--', 'ccpid': '-.', 'dob_cc_eq16': '-', 'dob_cc_eq17': '--'}


def save(fig, destination):
    fig.savefig(destination, dpi=175, metadata={'Software': 'rebuildIEEE PMSM DOB-CC'})
    plt.close(fig)


def basic_axes(ax, label, xlim=None):
    ax.set_ylabel(label)
    ax.grid(axis='both', alpha=0.19)
    ax.spines[['top', 'right']].set_visible(False)
    if xlim:
        ax.set_xlim(*xlim)


def plot_case(ax, wave, controller, channel, offset=0.0, scale=1.0):
    ax.plot((wave['t_s'] - offset) * scale, wave[channel], linestyle=STYLES[controller], color=COLORS[controller],
            linewidth=1.35, label=LABELS[controller])


def make_figures(waves, metrics, out):
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titlesize': 11})
    names = list(LABELS)
    fig, axes = plt.subplots(3, 2, figsize=(11, 7.5), sharex='col', layout='constrained')
    fig.suptitle('PMSM benchmark reconstruction at ±220 V, $n_p=4$')
    keys = ('speed_rpm', 'iq_a', 'uq_v')
    for col, xlim in enumerate(((0, 0.17), (0.399, 0.405))):
        axes[0, col].set_title(('Start from rest at 1 N·m', 'First step, 1 → 3 N·m')[col])
        for row, (key, label) in enumerate(zip(keys, ('Speed (rpm)', '$i_q$ (A)', '$u_q$ (V)'), strict=True)):
            ax = axes[row, col]
            for name in names:
                plot_case(ax, waves[f'equation_symbol_4__{name}'], name, key)
            basic_axes(ax, label, xlim)
        axes[0, col].axhline(3000, color='#333333', linestyle=':', linewidth=0.8)
        axes[1, col].axhline(10, color='#333333', linestyle=':', linewidth=0.8)
        axes[2, col].set_xlabel('Time (s)')
    axes[0, 0].legend(loc='lower right', fontsize=8, ncols=2, frameon=False)
    axes[0, 1].set_ylim(2992, 3014)
    axes[1, 1].set_ylim(0, 17)
    save(fig, out / 'startup_and_load_step.png')

    fig, axes = plt.subplots(2, 2, figsize=(11, 5.8), sharex='col', layout='constrained')
    fig.suptitle('Approach to the singular current boundary under overload')
    designs = [('equation_symbol_4', 'Ramp reconstruction, four pole pairs', (0.94, 1.13), 0, 1),
               ('four_physical_poles', 'Four physical poles at the first load step', (-30, 100), 0.4, 1e6)]
    for col, (poles, title, xlim, offset, scale) in enumerate(designs):
        upper, lower = axes[:, col]
        for name in ('dobc_high', 'dob_cc_eq16', 'dob_cc_eq17'):
            key = f'{poles}__{name}'
            wave = waves[key]
            plot_case(lower, wave, name, 'iq_a', offset, scale)
            if name != 'dobc_high' and metrics['cases'][key]['status'] == 'near_boundary':
                lower.plot((wave['t_s'][-1] - offset) * scale, wave['iq_a'][-1], 'o', color=COLORS[name], markersize=4)
        wave = waves[f'{poles}__dobc_high']
        upper.plot((wave['t_s'] - offset) * scale, wave['load_nm'], color='#333333', linewidth=1.3,
                   label='Applied load')
        upper.axhline(metrics['capacity'][poles]['load_at_ceiling_nm'], color='#6d6d6d', linestyle=':',
                       label='10 A torque capacity at 3000 rpm')
        lower.axhline(10, color='#333333', linestyle=':', linewidth=0.8)
        basic_axes(upper, 'Load torque (N·m)', xlim)
        basic_axes(lower, '$i_q$ (A)', xlim)
        lower.set_ylim((9.4, 10.4) if col == 0 else (6.9, 13.2))
        if col == 0:
            upper.set_ylim(2.84, 3.13)
        upper.set_title(title)
        lower.set_xlabel('Time from load step (µs)' if col else 'Time (s)')
    axes[0, 0].legend(loc='upper left', fontsize=8, frameon=False)
    axes[1, 0].legend(loc='upper left', fontsize=8, frameon=False)
    save(fig, out / 'overload_boundary.png')

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout='constrained')
    fig.suptitle('Voltage ceiling sensitivity, $n_p=4$')
    families = ['voltage_110v', 'equation_symbol_4', 'voltage_unlimited']
    for p, name in enumerate(('dobc_high', 'dob_cc_eq16', 'dob_cc_eq17')):
        entries = [metrics['cases'][f'{family}__{name}']['published_metric_comparison']['peak_current_a']
                   for family in families]
        axes[0].bar(np.arange(3) + 0.24 * (p - 1), entries, 0.23, label=LABELS[name], color=COLORS[name])
    axes[0].set_xticks(range(3), ('110 V', '220 V', 'Unlimited'))
    axes[0].set_title('Startup at 1 N·m')
    axes[0].set_yscale('log')
    axes[0].set_ylim(6, 32000)
    basic_axes(axes[0], 'Peak $|i_q|$ (A)')
    axes[0].axhline(10, color='#333333', linestyle=':', linewidth=0.8)
    axes[0].legend(fontsize=8, frameon=False)
    for family, color, label in (('equation_symbol_4', '#11765d', '220 V'), ('voltage_110v', '#bc7b19', '110 V')):
        wave = waves[f'{family}__dob_cc_eq16']
        axes[1].plot((wave['t_s'] - 0.4) * 1000, wave['speed_rpm'], color=color, label=label)
    axes[1].axhline(2940, color='#333333', linestyle=':', linewidth=0.8, label='2% lower speed bound')
    axes[1].set_title('DOB-CC, Eq. (16), at the 1 → 3 N·m step')
    axes[1].set_xlabel('Time from load step (ms)')
    axes[1].legend(fontsize=8, frameon=False, loc='lower right')
    basic_axes(axes[1], 'Speed (rpm)', (-10, 140))
    axes[1].set_ylim(2860, 3010)
    save(fig, out / 'voltage_authority.png')
