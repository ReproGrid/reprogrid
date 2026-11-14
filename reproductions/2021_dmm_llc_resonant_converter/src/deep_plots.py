"""Plot the measured differences and the reproduced LLC switching sequence."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from deep_metrics import COL
from plots import axes_style, save


def overview(destination, wave, source):
    fig, (upper, lower) = plt.subplots(2, 1, figsize=(10, 5.4), sharex=True, layout='constrained')
    fig.suptitle('Set #2 · 25 ns DMM on the 0.6 s sequence')
    t = wave[:, 0] * 1e-12
    upper.plot(*source['fig10a_vo'].T, color='0.55', linewidth=1.3, label='Fig. 10(a), PDF trace')
    upper.plot(t, wave[:, 4], color='#167a60', linewidth=1.15, label='Reconstruction, ideal output short')
    lower.plot(*source['fig10b_fs_khz'].T, color='0.55', linewidth=2, label='Fig. 10(b), PDF trace')
    lower.step(t, wave[:, 9] / 1000, where='post', color='#167a60', linewidth=1.15, label='Integer half-period schedule')
    for ax in (upper, lower):
        ax.axvspan(0.1, 0.13, color='#ecdccd', linewidth=0, zorder=-1)
        ax.legend(loc='best', frameon=False, fontsize=8.5)
    axes_style(upper, 'Output voltage (V)')
    axes_style(lower, 'Switching frequency (kHz)')
    lower.set_xlabel('Time (s)')
    lower.set_xlim(0, 0.6)
    save(fig, destination)


def panels(destination, waves, source, protocol):
    fig, axes = plt.subplots(4, 2, figsize=(10.9, 10.2), layout='constrained')
    fig.suptitle('Fig. 11 checkout · source reference and both gate phase policies')
    for row, record in enumerate(protocol['figure_windows']):
        center = record['center_s']
        vpanel, ipanel = record['panels'].split(',')
        for axis, signal_names, label, half, panel in (
            (axes[row, 0], ('vo',), 'Output voltage (V)', record['v_half_s'], vpanel),
            (axes[row, 1], ('ir', 'im'), 'Tank currents (A)', record['i_half_s'], ipanel),
        ):
            for channel in signal_names:
                name = f'fig11{panel}_{channel}'
                t, y = source[name].T
                style = '-' if channel != 'im' else ':'
                label_signal = ('Output voltage' if channel == 'vo' else f'$i_{channel[-1]}$')
                axis.plot((t - center) * 1e6, y, style, linewidth=1.5, color='#262626', label=f'Paper {label_signal}')
                for case, color, title in (('ideal_continuous', '#167a60', 'Continuous phase'),
                                            ('ideal_reset', '#ba6039', 'Reset at frequency change')):
                    wave = waves[case]
                    time = wave[:, 0] * 1e-12
                    m = np.abs(time - center) <= half * 1.001
                    axis.plot((time[m] - center) * 1e6, wave[m, COL[channel]], style, color=color, linewidth=1.1,
                              label=title if channel != 'im' else None)
            axis.set_xlim(-half * 1e6, half * 1e6)
            axis.set_title(f'Fig. 11({panel}) · {center:g} s')
            axes_style(axis, label)
            axis.set_xlabel('Time relative to window center (µs)')
    axes[0, 0].legend(loc='best', frameon=False, fontsize=8)
    axes[0, 1].legend(loc='best', frameon=False, fontsize=7.7, ncols=2)
    save(fig, destination)


def fault(destination, waves):
    fig, ax = plt.subplots(2, 2, figsize=(10.5, 6.0), layout='constrained')
    fig.suptitle('Output shunt and restoration sensitivity')
    for name, label, color, style in (
        ('ideal_continuous', 'Ideal short', '#167a60', '-'), ('shunt_0p1m', '0.1 mΩ', '#425eb2', '-.'),
        ('shunt_1m', '1 mΩ', '#ba6039', '--'), ('shunt_10m', '10 mΩ', '#6e4a96', ':'),
    ):
        wave = waves[name]
        t = wave[:, 0] * 1e-12
        stress = np.abs(t - 0.1) <= 8e-6
        comeback = (t >= 0.129995) & (t < 0.13025)
        for row, col in enumerate((4, 1)):
            ax[row, 0].plot((t[stress] - 0.1) * 1e6, wave[stress, col], style, label=label, color=color)
        ax[0, 1].plot((t[comeback] - 0.13) * 1e6, wave[comeback, 4], style, color=color)
        steady = (t > 0.105) & (t < 0.125)
        ax[1, 1].plot(t[steady] * 1000, wave[steady, 4], style, color=color)
    labels = (('Short insertion (µs)', 'Output voltage (V)'), ('Time from restoration (µs)', 'Output voltage (V)'),
              ('Short insertion (µs)', 'Resonant current (A)'), ('Time during short (ms)', 'Output voltage (V)'))
    for axis, (x, y) in zip(ax.flat, labels, strict=True):
        axis.set_xlabel(x)
        axes_style(axis, y)
    ax[0, 0].legend(loc='lower left', frameon=False, fontsize=8, ncols=2)
    ax[0, 0].axvline(0, color='0.5', linewidth=0.7)
    ax[1, 0].axvline(0, color='0.5', linewidth=0.7)
    save(fig, destination)


def stress(destination, study, protocol):
    detail = protocol['figure_3_4_detail']
    freqs = detail['stress_set2_hz']
    steps = detail['stress_step_ps']
    mat = np.array([[study[f'set2_{f}_{d}_forward_pairs']['ir_nrmse_pct'] or np.nan for f in freqs] for d in steps])
    fig, axis = plt.subplots(figsize=(8.6, 3.7), layout='constrained')
    plot = axis.imshow(np.ma.masked_invalid(mat), aspect='auto', cmap='YlOrBr')
    fig.colorbar(plot, ax=axis, label='$i_r$ NRMSE vs 15 ns full BE (%)')
    for row, col in np.ndindex(mat.shape):
        number = mat[row, col]
        axis.text(col, row, f'{number:.1f}' if np.isfinite(number) else 'cutoff', ha='center', va='center', color='black')
    axis.set_xticks(range(len(freqs)), [f'{f/1000:g}' for f in freqs])
    axis.set_yticks(range(len(steps)), [f'{d/1000:g}' for d in steps])
    axis.set(xlabel='Switching frequency (kHz)', ylabel='Forward Euler step (ns)', title='Set #2 explicit two-state stress sweep')
    save(fig, destination)


def figure34(destination, waveforms, source, protocol):
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 5.7), layout='constrained')
    fig.suptitle('Figs. 3–4 at 15 ns · source current and independent cases')
    pairs = (('set1', 'fig3a', (0.047893, 0.047904)), ('set2', 'fig4a', (0.0479004, 0.0479063)))
    nominal = protocol['figure_3_4_detail']['step_ps']
    for (set_key, panel, window), (left, right) in zip(pairs, axes, strict=True):
        hz = protocol['figure_3_4_detail']['frequencies_hz'][set_key]
        reference = source[panel + '_fe_ir']
        axis = (reference[:, 0] - window[0]) * 1e6
        left.plot(axis, reference[:, 1], color='#262626', linewidth=1.2, label='Paper FE trace')
        for method, color, style, label in (('dmm', '#167a60', '-', 'Full BE'),
                                            ('backward_pairs', '#355b9e', '--', 'Two-state BE'),
                                            ('forward_pairs', '#ba6039', '-.', 'Two-state FE')):
            w = waveforms[f'{set_key}_{hz}_{nominal}_{method}']
            t = w[:, 0]*1e-12
            active = (t >= window[0]) & (t <= window[1])
            right.plot((t[active] - window[0]) * 1e6, w[active, 1], style, color=color, linewidth=1.2, label=label)
        left.set_title(f'{panel.capitalize()} source with original phase')
        right.set_title(f'Set #{set_key[-1]} at {hz/1000:g} kHz, zero initial phase')
        for ax in (left, right):
            ax.set_xlim(0, (window[1]-window[0])*1e6)
            ax.set_xlabel('Time in source window (µs)')
            axes_style(ax, 'Resonant current (A)')
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[0, 1].legend(frameon=False, fontsize=8)
    save(fig, destination)
