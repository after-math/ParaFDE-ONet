from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots
import matplotlib

plt.style.use(['science', 'no-latex', 'grid',])
matplotlib.rcParams['font.family'] = 'Source Han Serif'
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['axes.unicode_minus'] = False


ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = ROOT / 'outputs/paper_figures'
SOURCE_DIR = ROOT / 'results/paper/source_data'
SEEDS = [20261001, 20261002, 20261003, 20261004, 20261005]


def average_rank(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method='average', ascending=True).to_numpy()


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)

    histories = []
    for seed in SEEDS:
        path = SOURCE_DIR / f'smart_grid_training_seed_{seed}.csv'
        frame = pd.read_csv(path)
        frame['seed'] = seed
        histories.append(frame)
    training = pd.concat(histories, ignore_index=True)

    source = pd.read_csv(SOURCE_DIR / 'smart_grid_survivability_predictions.csv')
    direct = source['direct_survivability'].to_numpy()
    para_mean = source['parafdeonet_mean_survivability'].to_numpy()
    para_std = source['parafdeonet_sample_std_survivability'].to_numpy()
    gbt = source['gbt_survivability'].to_numpy()
    direct_rank = average_rank(direct)
    para_rank = average_rank(para_mean)
    gbt_rank = average_rank(gbt)

    blue = '#0072B2'
    orange = '#D55E00'
    gray = '#3A3A3A'

    fig, axes = plt.subplots(1, 3, figsize=(7.35, 2.48))

    ax = axes[0]
    pivot = training.pivot(index='iteration', columns='seed', values='validation_normalized_mse')
    x = pivot.index.to_numpy(dtype=float) / 1000.0
    y = pivot.to_numpy(dtype=float)
    for column in range(y.shape[1]):
        ax.plot(x, y[:, column], color=blue, alpha=0.28, lw=0.7)
    mean = y.mean(axis=1)
    lower_envelope = y.min(axis=1)
    upper_envelope = y.max(axis=1)
    ax.fill_between(x, lower_envelope, upper_envelope, color=blue, alpha=0.13, linewidth=0)
    ax.plot(x, mean, color=blue, lw=1.35, label='Five-seed mean')
    ax.set_yscale('log')
    ax.set_xlabel('Training iteration ($10^3$)')
    ax.set_ylabel('Validation normalized MSE')
    ax.legend(loc='upper right', frameon=False, fontsize=6.5)
    ax.text(-0.18, 1.04, '(a)', transform=ax.transAxes, fontweight='bold', fontsize=9)

    ax = axes[1]
    low = 50.0
    high = 94.0
    ax.plot([low, high], [low, high], color=gray, lw=0.8, ls='--', zorder=1)
    para_handle = ax.errorbar(
        100.0 * direct,
        100.0 * para_mean,
        yerr=100.0 * para_std,
        fmt='o',
        ms=2.7,
        mew=0.35,
        color=blue,
        ecolor=blue,
        elinewidth=0.45,
        capsize=1.0,
        alpha=0.88,
        label='ParaFDEONet (0.145 pp)',
        zorder=3,
    )
    gbt_handle = ax.scatter(
        100.0 * direct,
        100.0 * gbt,
        s=12,
        marker='s',
        facecolors='none',
        edgecolors=orange,
        linewidths=0.65,
        label='GBT (2.798 pp)',
        zorder=2,
    )
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_xticks([50, 60, 70, 80, 90])
    ax.set_yticks([50, 60, 70, 80, 90])
    ax.set_xlabel('Direct DDE survivability (%)')
    ax.set_ylabel('Predicted survivability (%)')
    ax.legend([para_handle, gbt_handle], ['ParaFDEONet (0.145 pp)', 'GBT (2.798 pp)'],
              loc='upper left', frameon=False, fontsize=6.1, handletextpad=0.35)
    ax.text(-0.18, 1.04, '(b)', transform=ax.transAxes, fontweight='bold', fontsize=9)

    ax = axes[2]
    ax.fill_between([0.5, 7.5], 0.5, 7.5, color='#E6E6E6', zorder=0)
    ax.fill_between([0.5, 4.5], 0.5, 4.5, color='#C9C9C9', zorder=0)
    ax.plot([0.5, 64.5], [0.5, 64.5], color=gray, lw=0.8, ls='--', zorder=1)
    ax.scatter(direct_rank, para_rank, s=13, color=blue, alpha=0.85, label=r'ParaFDEONet ($\rho=0.999$)', zorder=3)
    ax.scatter(direct_rank, gbt_rank, s=14, marker='s', facecolors='none', edgecolors=orange,
               linewidths=0.65, label=r'GBT ($\rho=0.843$)', zorder=2)
    ax.set_xlim(0.5, 64.5)
    ax.set_ylim(0.5, 64.5)
    ax.set_xticks([1, 16, 32, 48, 64])
    ax.set_yticks([1, 16, 32, 48, 64])
    ax.set_xlabel('Direct DDE risk rank')
    ax.set_ylabel('Predicted risk rank')
    ax.legend(loc='lower right', frameon=False, fontsize=6.1, handletextpad=0.35)
    ax.text(-0.18, 1.04, '(c)', transform=ax.transAxes, fontweight='bold', fontsize=9)

    for ax in axes:
        ax.tick_params(labelsize=6.7, width=0.6, length=2.4)
        ax.xaxis.label.set_size(7.2)
        ax.yaxis.label.set_size(7.2)
        for spine in ax.spines.values():
            spine.set_linewidth(0.65)

    fig.subplots_adjust(left=0.067, right=0.995, bottom=0.205, top=0.94, wspace=0.34)
    output = FIGURE_DIR / 'smart_grid_survivability'
    fig.savefig(output.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(output.with_suffix('.svg'), bbox_inches='tight')
    fig.savefig(output.with_suffix('.png'), dpi=300, bbox_inches='tight')
    fig.savefig(output.with_suffix('.tiff'), dpi=300, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
