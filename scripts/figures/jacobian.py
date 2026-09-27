from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots
import matplotlib
plt.style.use(['science','no-latex','grid',])
matplotlib.rcParams['font.family']='Source Han Serif'
matplotlib.rcParams['font.sans-serif'] = ['Source Han Serif', 'Arial', 'Helvetica']
matplotlib.rcParams.update({"pdf.fonttype": 42, "svg.fonttype": "none"})


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs' / 'paper_figures'
SOURCE = ROOT / 'results/paper/source_data'

SYSTEMS = [
    ('competition', 'Variable-delay competition', ['overall', 'r1', 'r2'],
     ['Overall', r'$r_1$', r'$r_2$']),
    ('sei', 'Delayed SEI', ['overall', 'b', 'a'],
     ['Overall', r'$b$', r'$a$']),
    ('nicholson', 'Four-patch Nicholson', ['overall', 'beta', 'tau0'],
     ['Overall', r'$\beta$', r'$\tau_0$']),
]


def load_ratios():
    return pd.read_csv(SOURCE / 'sensitivity_jacobian_ratio.csv')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    SOURCE.mkdir(parents=True, exist_ok=True)
    data = load_ratios()

    matplotlib.rcParams.update({
        'font.size': 8.0,
        'axes.titlesize': 9.0,
        'axes.labelsize': 8.5,
        'xtick.labelsize': 8.0,
        'ytick.labelsize': 8.0,
        'axes.linewidth': 0.7,
        'grid.alpha': 0.32,
        'grid.linewidth': 0.45,
        'savefig.bbox': 'tight',
    })

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.65), sharey=True)
    signal = '#C6423A'
    point = '#60758A'
    for panel, (ax, (system, title, parameters, labels)) in enumerate(zip(axes, SYSTEMS)):
        subset = data[data['system'] == system]
        arrays = []
        for position, parameter in enumerate(parameters, start=1):
            values = subset.loc[subset['parameter'] == parameter, 'ratio'].to_numpy()
            if np.any(values <= 0):
                raise ValueError('Jacobian-error ratios must be positive for a logarithmic axis.')
            arrays.append(values)
            jitter = np.linspace(-0.16, 0.16, values.size)
            ax.scatter(
                position + jitter, values, s=5.5, color=point, alpha=0.20,
                linewidths=0, rasterized=True, zorder=1,
            )

        bp = ax.boxplot(
            arrays, positions=np.arange(1, 4), widths=0.48, showfliers=False,
            patch_artist=True,
            medianprops={'color': '#202020', 'linewidth': 1.15},
            boxprops={'facecolor': signal, 'edgecolor': signal, 'alpha': 0.34,
                      'linewidth': 0.9},
            whiskerprops={'color': signal, 'linewidth': 0.9},
            capprops={'color': signal, 'linewidth': 0.9},
        )
        for median in bp['medians']:
            median.set_zorder(3)

        ax.axhline(1.0, color='#333333', linestyle='--', linewidth=0.9, zorder=0)
        ax.set_yscale('log')
        ax.set_xticks([1, 2, 3], labels)
        ax.set_title(title, pad=4)
        ax.text(-0.15, 1.04, chr(ord('a') + panel), transform=ax.transAxes,
                fontsize=9.5, fontweight='bold', va='bottom')
        ax.set_xlim(0.55, 3.45)
        ax.set_ylim(0.025, 2.3)
        ax.minorticks_off()
        ax.grid(True, axis='y', which='major')
        ax.grid(False, axis='x')

    axes[0].set_ylabel(
        'Paired Jacobian-error ratio\n(with / without sensitivity supervision)'
    )
    fig.subplots_adjust(left=0.10, right=0.995, bottom=0.19, top=0.87, wspace=0.10)

    stem = OUT / 'sensitivity_jacobian_ratio'
    fig.savefig(stem.with_suffix('.pdf'))
    fig.savefig(stem.with_suffix('.svg'))
    fig.savefig(stem.with_suffix('.png'), dpi=400)
    fig.savefig(stem.with_suffix('.tiff'), dpi=600, pil_kwargs={'compression': 'tiff_lzw'})
    plt.close(fig)


if __name__ == '__main__':
    main()
