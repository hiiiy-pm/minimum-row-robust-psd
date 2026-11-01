from pathlib import Path
import argparse
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator, PercentFormatter, LogLocator, NullLocator

BASE = Path(__file__).resolve().parents[1]
DATA, OUT = BASE / 'data', BASE / 'figures'
OUT.mkdir(exist_ok=True)
BLUE, ORANGE, INK, GREY = '#215B8F', '#CB7227', '#252A2E', '#78828A'
GRID = '#EEF0F2'
MANUSCRIPT = None
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['STIXGeneral'],
    'mathtext.fontset': 'stix', 'font.size': 11,
    'axes.labelsize': 11, 'axes.titlesize': 11, 'legend.fontsize': 11,
    'xtick.labelsize': 11, 'ytick.labelsize': 11,
    'axes.linewidth': 1.2, 'lines.linewidth': 1.8,
    'lines.markeredgewidth': 1.2, 'patch.linewidth': 1.2,
    'xtick.major.width': 1.2, 'ytick.major.width': 1.2,
    'xtick.minor.width': 1.2, 'ytick.minor.width': 1.2,
    'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'savefig.facecolor': 'white', 'figure.facecolor': 'white',
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'axes.labelcolor': INK, 'text.color': INK,
    'xtick.color': INK, 'ytick.color': INK,
})


def panel(ax, letter, title):
    ax.set_title(f'({letter})  {title}', loc='left', pad=10)
    ax.grid(axis='y', color=GRID, linewidth=1.2)
    ax.set_axisbelow(True)


def save(fig, name):
    for ext in ['pdf', 'svg', 'png']:
        kw = dict(bbox_inches='tight', pad_inches=.065)
        if ext == 'png':
            kw['dpi'] = 600
        destination = MANUSCRIPT if ext == 'pdf' and MANUSCRIPT is not None else OUT
        fig.savefig(destination / f'{name}.{ext}', **kw)
    svg = OUT / f'{name}.svg'
    svg.write_text(re.sub(r'<!--.*?-->', '', svg.read_text(encoding='utf-8'), flags=re.S), encoding='utf-8')
    plt.close(fig)


def parity():
    costs = pd.read_csv(DATA / 'epsilon_closed_form.csv')
    spectra = pd.read_csv(DATA / 'epsilon_spectra.csv')
    d = costs[np.isclose(costs.epsilon, .1)].sort_values('m')
    m, target = d.m.to_numpy(), d.target_trace.to_numpy()
    alpha, kappa = float(d.alpha.iloc[0]), float(d.kappa.iloc[0])
    odd = m % 2 == 1
    fig, ax = plt.subplots(1, 3, figsize=(7.35, 2.85), layout='constrained',
                           gridspec_kw={'width_ratios': [1.13, 1., 1.]})
    panel(ax[0], 'a', r'Energy, $\varepsilon=0.1$')
    ax[0].plot(m, m * alpha, '--', c=GREY, label=r'Separate $m\alpha$')
    ax[0].plot(m, m * kappa, ':', c=INK, label=r'Moment $m\kappa$')
    ax[0].plot(m, target, '-', c=BLUE, label=r'Optimum $T_m$')
    ax[0].scatter(m[~odd], target[~odd], s=13, facecolor='white', edgecolor=BLUE, zorder=4)
    ax[0].scatter(m[odd], target[odd], s=12, color=ORANGE, marker='s', zorder=4)
    ax[0].set(xlabel='Repetitions $m$', ylabel='Total measurement energy', ylim=(0, 175), xlim=(.5, 16.5))
    ax[0].set_xticks([1, 4, 8, 12, 16])
    ax[0].legend(loc='upper left', frameon=False, fontsize=11.0)
    panel(ax[1], 'b', 'Exact parity penalty')
    penalty = (target - m * kappa) / (alpha - kappa)
    ax[1].axhline(1, color=ORANGE, lw=1.2, ls=':', alpha=.6)
    ax[1].vlines(m[odd], 0, penalty[odd], color=ORANGE, lw=1.2)
    ax[1].scatter(m[~odd], penalty[~odd], s=23, facecolor='white', edgecolor=BLUE, label='Even $m$', zorder=3)
    ax[1].scatter(m[odd], penalty[odd], s=23, color=ORANGE, marker='s', label='Odd $m$', zorder=3)
    ax[1].set(xlabel='Repetitions $m$', ylabel=r'$(T_m-m\kappa)/(\alpha-\kappa)$',
              ylim=(-.12, 1.3), xlim=(.5, 16.5), yticks=[0, 1])
    ax[1].set_xticks([1, 4, 8, 12, 16])
    ax[1].legend(loc='upper center', ncol=2, frameon=False, fontsize=11.0, columnspacing=.65, handletextpad=.2)
    panel(ax[2], 'c', 'Optimal spectrum')
    for count, shift, color, marker in [(6, -.11, BLUE, 'o'), (7, .11, ORANGE, 's')]:
        ss = spectra[np.isclose(spectra.epsilon, .1) & (spectra.m == count)].sort_values('index')
        xx, yy = ss['index'].to_numpy() + shift, ss.eigenvalue.to_numpy()
        ax[2].vlines(xx, 0, yy, color=color, lw=1.2, alpha=.6)
        ax[2].scatter(xx, yy, s=21, marker=marker, facecolors='white' if count == 6 else color,
                      edgecolors=color, label=f'$m={count}$', zorder=3)
    ax[2].axhline(kappa, color=INK, lw=1.2, ls=':')
    ax[2].set(xlabel='Ordered index $i$', ylabel=r'$\lambda_i(W^T W)$', xlim=(.5, 7.6), ylim=(-.15, 11.8),
              xticks=[1, 3, 5, 7], yticks=[0, kappa, alpha], yticklabels=['0', r'$\kappa$', r'$\alpha$'])
    ax[2].legend(loc='upper left', frameon=False, fontsize=11.0)
    save(fig, 'fig01_parity')


def frontier():
    d = pd.read_csv(DATA / 'figure_energy_frontier.csv')
    e, a, k = d.epsilon.to_numpy(), d.alpha.to_numpy(), d.kappa.to_numpy()
    fig, ax = plt.subplots(1, 3, figsize=(7.35, 2.8), layout='constrained')
    panel(ax[0], 'a', 'Scalar and paired costs')
    ax[0].loglog(e, a, c=ORANGE, label=r'Scalar $\alpha$')
    ax[0].loglog(e, k, c=BLUE, label=r'Moment $\kappa$')
    ax[0].set(xlabel=r'Family parameter $\varepsilon$', ylabel='Energy per coordinate')
    ax[0].legend(frameon=False, loc='upper right', fontsize=11.0, handlelength=1.4)
    panel(ax[1], 'b', 'Even-dimensional saving')
    ax[1].semilogx(e, d.saving_even, c=BLUE)
    ax[1].axhline(1, c=GREY, lw=1.2, ls=':')
    gain = .9 ** 2 / 1.1
    ax[1].scatter([.1], [gain], c=ORANGE, s=24, zorder=3)
    ax[1].annotate('73.64%', xy=(.1, gain), xytext=(.014, .48), fontsize=11.0,
                   arrowprops={'arrowstyle': '-', 'color': GREY, 'lw': 1.2})
    ax[1].set(xlabel=r'Family parameter $\varepsilon$', ylabel=r'$1-\kappa/\alpha$', ylim=(0, 1.045))
    ax[1].yaxis.set_major_formatter(PercentFormatter(1))
    panel(ax[2], 'c', 'One additional row, $m=3$')
    ax[2].loglog(e, d.T_3_minimum_rows, c=ORANGE, label='3-row optimum')
    ax[2].loglog(e, d.T_3_four_rows_upper_bound, c=BLUE, ls='--', label='4-row bound')
    ax[2].axvline(.5, ymax=.64, c=GREY, lw=1.2, ls=':')
    ax[2].set(xlabel=r'Family parameter $\varepsilon$', ylabel='Total measurement energy')
    ax[2].legend(frameon=False, loc='upper right', fontsize=11.0, handlelength=1.4)
    for axis in ax:
        axis.set_xticks([1e-3, 1e-2, 1e-1, 1])
        axis.xaxis.set_minor_locator(NullLocator())
        axis.yaxis.set_minor_locator(NullLocator())
    save(fig, 'fig02_energy_frontier')


def structure_map(ax, matrix, title, lo=-1, hi=1):
    cmap = LinearSegmentedColormap.from_list('structure', [BLUE, '#FFFFFF', ORANGE])
    n = len(matrix)
    image = ax.pcolormesh(np.arange(n + 1), np.arange(n + 1), matrix, cmap=cmap,
                          vmin=lo, vmax=hi, shading='flat', rasterized=False, antialiased=False)
    ax.set_aspect('equal')
    ax.set(xlim=(0, n), ylim=(n, 0), xticks=[], yticks=[])
    ax.set_title(title, loc='left', pad=9, fontsize=11.0)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)
    return image


def spectrum_map(ax, matrices, m, alpha, kappa, letter):
    panel(ax, letter, f'Gram spectrum, $m={m}$')
    for index, H in enumerate(matrices):
        eig = np.linalg.eigvalsh(H)
        eig[np.abs(eig) < 1e-10] = 0
        xx = np.arange(1, 2 * m + 1) + (index - .5) * .16
        color, marker = (GREY, 'o') if index == 0 else (BLUE, 'x')
        ax.vlines(xx, 0, eig, color=color, alpha=.55, lw=1.2)
        kw = dict(s=23, marker=marker, label='Aligned' if index == 0 else 'Rotated', linewidths=1.2, zorder=3)
        if index == 0:
            kw.update(facecolors='white', edgecolors=color)
        else:
            kw.update(c=color)
        ax.scatter(xx, eig, **kw)
    ax.axhline(kappa, c=INK, lw=1.2, ls=':')
    ticks, labels = [0, kappa], ['0', r'$\kappa$']
    if m % 2:
        ticks.append(alpha)
        labels.append(r'$\alpha$')
    ax.set(xlabel='Eigenvalue index', ylabel=r'$\lambda_i(H)$',
           yticks=ticks, yticklabels=labels, xlim=(.5, 2 * m + .6),
           ylim=(-.08 * (alpha if m % 2 else kappa), 1.18 * (alpha if m % 2 else kappa)))
    ax.set_xticks([1, m, 2 * m])


def structures():
    H = [pd.read_csv(DATA / f'figure_optimizer_H_{i}.csv').to_numpy() for i in range(1, 5)]
    e = .1
    r, t, alpha = 2 / (1 + e), (1 - e) / (1 + e), 1 / e
    q, kappa = np.sqrt(r * t), r + t
    Jpaired, Jrotated = H[0][:6, 6:] / q, H[1][:6, 6:] / q
    projector = (H[3][:5, :5] - r * np.eye(5)) / (alpha - r)
    Jodd = H[3][:5, 5:] / q
    residuals = {
        'even_complex_structure': float(np.linalg.norm(Jrotated @ Jrotated + np.eye(6))),
        'odd_projector_idempotence': float(np.linalg.norm(projector @ projector - projector)),
        'odd_structure_squared': float(np.linalg.norm(Jodd @ Jodd + np.eye(5) - projector)),
        'odd_axis_annihilation': float(np.linalg.norm(Jodd @ projector)),
    }
    assert max(residuals.values()) < 1e-10
    fig = plt.figure(figsize=(7.35, 4.4), layout='constrained')
    grid = fig.add_gridspec(3, 3, height_ratios=[1., 1., .055], width_ratios=[1., 1., 1.15])
    ax = np.array([[fig.add_subplot(grid[i, j]) for j in range(3)] for i in range(2)])
    color_image = structure_map(ax[0, 0], Jpaired, r'(a)  Paired $J$, $m=6$')
    structure_map(ax[0, 1], Jrotated, r'(b)  Rotated $J$, $m=6$')
    spectrum_map(ax[0, 2], H[:2], 6, alpha, kappa, 'c')
    structure_map(ax[1, 0], projector, r'(d)  Axis projector $uu^T$')
    structure_map(ax[1, 1], Jodd, r'(e)  Complementary $J$, $m=5$')
    spectrum_map(ax[1, 2], H[2:], 5, alpha, kappa, 'f')
    color_axis = fig.add_subplot(grid[2, :2])
    cb = fig.colorbar(color_image, cax=color_axis, orientation='horizontal', ticks=[-1, 0, 1])
    cb.solids.set_rasterized(False)
    cb.solids.set_edgecolor('face')
    cb.solids.set_linewidth(1.2)
    cb.ax.tick_params(labelsize=11.0, length=3)
    cb.set_label('Matrix entry', fontsize=11.0, labelpad=2)
    legend_axis = fig.add_subplot(grid[2, 2])
    legend_axis.set_axis_off()
    handles, labels = ax[1, 2].get_legend_handles_labels()
    legend_axis.legend(handles, labels, loc='center', frameon=False, fontsize=11.0, ncol=2,
                       handletextpad=.2, columnspacing=.8)
    save(fig, 'fig03_optimizer_structure')


def direct():
    d = pd.read_csv(DATA / 'nonconvex_runs.csv')
    if d['feasible'].dtype == object:
        d['feasible'] = d['feasible'].astype(str).str.lower().eq('true')
    d['repairable'] = d.scaled_repair_relative_gap.notna()
    threshold = 1e-6
    d['close'] = d.repairable & (d.scaled_repair_relative_gap <= threshold)
    d['larger'] = d.repairable & ~d.close
    d['unrepairable'] = ~d.repairable
    grouped = d.groupby('m')[['close', 'larger', 'unrepairable']].sum()
    fig = plt.figure(figsize=(7.35, 3.5), layout='constrained')
    grid = fig.add_gridspec(2, 3, height_ratios=[1., .13], hspace=.04)
    axes = []
    offsets = np.linspace(-.31, .31, 8)
    plotted, unavailable = [], []
    for j, epsilon in enumerate([.03, .2, .7]):
        ax = fig.add_subplot(grid[0, j], sharey=axes[0] if axes else None)
        strip = fig.add_subplot(grid[1, j], sharex=ax)
        axes.append(ax)
        panel(ax, 'abc'[j], rf'$\varepsilon={epsilon:g}$')
        ax.set_yscale('log')
        ax.set(xlim=(1.5, 4.5), ylim=(8e-13, 1e4), xticks=[2, 3, 4],
               yticks=[1e-12, 1e-8, 1e-4, 1, 1e4])
        ax.axhline(threshold, color=GREY, ls='--', lw=1.2, zorder=1)
        ax.tick_params(axis='x', labelbottom=False, bottom=False)
        ax.yaxis.set_minor_locator(NullLocator())
        if j:
            ax.tick_params(axis='y', labelleft=False)
        else:
            ax.set_ylabel(r'Repaired relative gap $(T-T_m)/T_m$')
            ax.annotate(r'$10^{-6}$', xy=(1.52, threshold), xytext=(3, 4),
                        textcoords='offset points', color=GREY, fontsize=11.0)
        subset = d[np.isclose(d.epsilon, epsilon)]
        for m in [2, 3, 4]:
            alpha, kappa = 1 / epsilon, (3 - epsilon) / (1 + epsilon)
            target = (m - m % 2) * kappa + (m % 2) * alpha
            baseline = (m * alpha - target) / target
            ax.hlines(baseline, m - .34, m + .34, colors=INK, linewidth=1.4, zorder=2)
        for row in subset.itertuples():
            if not row.repairable:
                unavailable.append(row.factor_archive_key)
                continue
            gap = float(row.scaled_repair_relative_gap)
            assert gap > 0
            color, marker = (BLUE, 'o') if row.feasible else (ORANGE, '^')
            ax.scatter(row.m + offsets[row.start_id], gap, s=21 if row.feasible else 29,
                       marker=marker, facecolors='white' if row.feasible else color,
                       edgecolors=color, linewidths=1.2, zorder=3)
            plotted.append(row.factor_archive_key)
        strip.set(ylim=(-.5, .5), yticks=[], xticks=[2, 3, 4])
        strip.grid(False)
        strip.spines['left'].set_visible(False)
        strip.spines['bottom'].set_visible(False)
        strip.tick_params(axis='x', bottom=False, pad=3)
        for m in [2, 3, 4]:
            count = int(subset.loc[subset.m == m, 'unrepairable'].sum())
            strip.text(m, 0, str(count), ha='center', va='center', fontsize=11.0,
                       color=ORANGE if count else GREY)
        if j == 0:
            strip.set_ylabel('No repair', rotation=0, ha='right', va='center', fontsize=11.0, labelpad=9)
        if j == 1:
            strip.set_xlabel('Repetitions $m$', labelpad=4)
    handles = [Line2D([], [], color=BLUE, marker='o', markerfacecolor='white', ls='None',
                      markersize=4.5, label='Raw feasible'),
               Line2D([], [], color=ORANGE, marker='^', ls='None', markersize=5, label='Raw infeasible'),
               Line2D([], [], color=INK, lw=1.4, label='Coordinatewise optimum')]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, -.08), ncol=3,
               frameon=False, fontsize=11.0, handletextpad=.4, columnspacing=1.2)
    assert int(grouped.to_numpy().sum()) == len(d) == 72
    assert len(plotted) == 71 and len(unavailable) == 1
    save(fig, 'fig04_direct_optimization')


def stability():
    d = pd.read_csv(DATA / 'stability_paths.csv')
    fits = pd.read_csv(DATA / 'stability_fits.csv')
    selected = d[(d.m == 2) & (((d.family == 'epsilon') & np.isclose(d.epsilon, .2)) |
                                  (d.family == 'rank_one_degenerate')) & (d.path != 'radial_reference')]
    fig, ax = plt.subplots(1, 2, figsize=(7.35, 3.0), layout='constrained')
    panel(ax[0], 'a', 'Distance to the optimal set')
    panel(ax[1], 'b', 'Distance-to-gap ratio')
    legend, labels, fit_slopes = [], [], {}
    for path, color, marker, style, label in [
        ('full_random_perturbation', BLUE, 'o', '-', r'Full factor, $\varepsilon=0.2$'),
        ('skew_amplitude', GREY, '^', ':', r'Skew amplitude, $\varepsilon=0.2$'),
        ('tilted_scalar_axis', ORANGE, 's', '--', 'Rank-one control')]:
        g = selected[selected.path == path].sort_values('gap')
        g = g[(g.gap > 1e-13) & (g.distance > 1e-13)]
        line, = ax[0].loglog(g.gap, g.distance, c=color, marker=marker, ls=style, ms=3.6, label=label)
        ax[1].loglog(g.gap, g.distance / g.gap, c=color, marker=marker, ls=style, ms=3.6)
        legend.append(line)
        labels.append(label)
        ff = fits[(fits.path == path) & (fits.m == 2)]
        if path != 'tilted_scalar_axis':
            ff = ff[np.isclose(ff.epsilon, .2)]
        fit_slopes[path] = float(ff.log_distance_vs_log_gap_slope.iloc[0])
    ax[0].text(.04, .91, r'$p=1$' + '\n' + rf'fits ${fit_slopes["full_random_perturbation"]:.3f}$, ${fit_slopes["skew_amplitude"]:.3f}$', transform=ax[0].transAxes,
                fontsize=11.0, color=BLUE, va='top')
    ax[0].text(.08, .56, r'$p=\frac{1}{2}$' + '\n' + rf'fit ${fit_slopes["tilted_scalar_axis"]:.3f}$', transform=ax[0].transAxes,
                fontsize=11.0, color=ORANGE, va='top')
    ax[1].text(.15, .68, r'$\Delta^{-1/2}$', transform=ax[1].transAxes, color=ORANGE, fontsize=11.0)
    ax[1].text(.6, .13, 'Bounded ratio', transform=ax[1].transAxes, color=BLUE, fontsize=11.0)
    ax[0].set(xlabel=r'Cost gap $\Delta$', ylabel=r'$\operatorname{dist}_*(H,\mathcal{M})$')
    ax[1].set(xlabel=r'Cost gap $\Delta$', ylabel=r'$\operatorname{dist}_*(H,\mathcal{M})/\Delta$')
    for a in ax:
        a.xaxis.set_major_locator(LogLocator(base=10, numticks=7))
        a.xaxis.set_minor_locator(NullLocator())
        a.yaxis.set_minor_locator(NullLocator())
    fig.legend(legend, labels, loc='lower center', bbox_to_anchor=(.5, -.09), ncol=3,
                frameon=False, fontsize=11.0, handlelength=2, columnspacing=1.4)
    save(fig, 'fig05_stability')


def graph():
    curves = pd.read_csv(DATA / 'figure_graph_curves.csv')
    checked = pd.read_csv(DATA / 'k4_scan.csv')
    g0, g1 = 327 / 512, 169 / 256
    high = curves[curves.gamma >= g0]
    gap = high.alpha.to_numpy() - high.kappa.to_numpy()
    saving = 100 * gap / high.alpha.to_numpy()
    fig, ax = plt.subplots(1, 2, figsize=(7.35, 3.15), layout='constrained')
    panel(ax[0], 'a', 'Full threshold range')
    ax[0].axvspan(g0, g1, color=ORANGE, alpha=.12, lw=0)
    ax[0].plot(curves.gamma, curves.alpha, c=ORANGE, label=r'Scalar $\alpha$')
    ax[0].plot(curves.gamma, curves.kappa, c=BLUE, ls='--', label=r'Moment $\kappa$')
    ax[0].set(xlabel=r'Safety threshold $\gamma$', ylabel='Energy per coordinate',
               xlim=(49 / 256 - .007, g1 + .008), ylim=(-.01, .56))
    for bound, label, align in [(g0, '$G$', 'right'), (g1, '$S$', 'left')]:
        ax[0].axvline(bound, color=GREY, lw=1.2, ls=':', zorder=1)
        ax[0].text(bound, .548, label, ha=align, va='top', fontsize=11.0)
    ax[0].scatter([49 / 256, g1], [0, 1059 / 2048], s=27, facecolors='white',
                   edgecolors=INK, linewidths=1.2, zorder=4)
    ax[0].legend(loc='lower right', bbox_to_anchor=(.89, .035), frameon=False, fontsize=11.0)
    inset = ax[0].inset_axes([.15, .59, .52, .34])
    inset.plot(high.gamma, gap * 1e4, color=INK, lw=1.25)
    inset.set(xlim=(g0, g1), ylim=(-.02, 4.8), xticks=[g0, .65, g1],
              xticklabels=['$G$', '0.65', '$S$'], yticks=[0, 2, 4])
    inset.set_title(r'$10^4(\alpha-\kappa)$', fontsize=11.0, loc='left', pad=4)
    inset.tick_params(labelsize=11.0, length=2)
    inset.spines['top'].set_visible(True)
    inset.spines['right'].set_visible(True)
    inset.grid(axis='y', color=GRID, lw=1.2)
    panel(ax[1], 'b', r'Strict gap, $G<\gamma<S$')
    ax[1].plot(high.gamma, saving, c=BLUE, lw=1.8)
    cert = checked[(checked.gamma > g0) & (checked.gamma < g1)]
    ax[1].scatter(cert.gamma, cert.relative_saving * 100, s=19, marker='D', facecolors='white',
                   edgecolors=BLUE, linewidths=1.2, zorder=3)
    ax[1].scatter([g0], [0], s=28, c=BLUE, zorder=5)
    ax[1].scatter([g1], [0], s=28, facecolors='white', edgecolors=BLUE,
                   linewidths=1.2, zorder=5)
    for bound in [g0, g1]:
        ax[1].axvline(bound, c=GREY, lw=1.2, ls=':', zorder=1)
    witness = checked[np.isclose(checked.gamma, .65)].iloc[0]
    gain = 100 * float(witness.relative_saving)
    ax[1].axvline(.65, c=GREY, lw=1.2, ls=':')
    ax[1].scatter([.65], [gain], s=28, c=ORANGE, zorder=4)
    ax[1].annotate(r'$\gamma=0.65$' + '\n' + f'{gain:.5f}%', xy=(.65, gain),
                   xytext=(.659, .096), fontsize=11.0, ha='right', va='top',
                   arrowprops={'arrowstyle': '-', 'lw': 1.2, 'color': GREY})
    ax[1].set(xlabel=r'Safety threshold $\gamma$', ylabel='Even-dimensional saving (%)',
               xlim=(g0 - .001, g1 + .001), ylim=(-.004, .104),
               xticks=[g0, .645, .65, .655, g1],
               xticklabels=['$G$', '0.645', '0.650', '0.655', '$S$'])
    ax[1].yaxis.set_major_locator(MaxNLocator(nbins=5))
    save(fig, 'fig06_graph_box')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manuscript-dir', type=Path)
    args = parser.parse_args()
    MANUSCRIPT = args.manuscript_dir
    if MANUSCRIPT is not None:
        MANUSCRIPT.mkdir(parents=True, exist_ok=True)
    functions = [parity, frontier, structures, direct, stability, graph]
    for function in functions:
        function()
    print(f'Exported {len(functions)} figures in PDF, SVG and 600 dpi PNG.')
