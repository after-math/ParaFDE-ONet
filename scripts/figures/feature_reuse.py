from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import scienceplots
import matplotlib
import numpy as np
import pandas as pd


plt.style.use(['science', 'no-latex', 'grid',])
matplotlib.rcParams['font.family'] = 'Source Han Serif'
matplotlib.rcParams['font.serif'] = ['Source Han Serif', 'Arial']
matplotlib.rcParams.update({
    'svg.fonttype': 'none',
    'pdf.fonttype': 42,
})


ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = ROOT / 'outputs' / 'paper_figures'
SOURCE_DIR = ROOT / 'results/paper/source_data'
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
SOURCE_DIR.mkdir(parents=True, exist_ok=True)

summary_path = SOURCE_DIR / 'feature_reuse_timing_summary.csv'
correctness_path = SOURCE_DIR / 'feature_reuse_correctness.csv'
raw_path = SOURCE_DIR / 'feature_reuse_raw_timings.csv'

summary = pd.read_csv(summary_path)
raw = pd.read_csv(raw_path)

if np.any(summary['query_count'].to_numpy() <= 0):
    raise ValueError('Query counts must be strictly positive for a logarithmic axis.')
if np.any(summary['steady_mean_seconds'].to_numpy() <= 0):
    raise ValueError('Timing values must be strictly positive for a logarithmic axis.')


method_order = ['JI-DeepONet', 'RP-MIONet', 'PFDEONet']
colors = {
    'JI-DeepONet': '#555555',
    'RP-MIONet': '#6F7FAF',
    'PFDEONet': '#C44E52',
}
markers = {
    'JI-DeepONet': 's',
    'RP-MIONet': '^',
    'PFDEONet': 'o',
}

fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.85))

for method in method_order:
    part = summary.loc[summary['method'] == method].sort_values('query_count')
    axes[0].errorbar(
        part['query_count'],
        part['steady_mean_seconds'],
        yerr=part['steady_sample_sd_seconds'],
        color=colors[method],
        marker=markers[method],
        markersize=4.2,
        linewidth=1.25,
        elinewidth=0.8,
        capsize=2.0,
        label=method,
        zorder=3,
    )

axes[0].set_xscale('log')
axes[0].set_yscale('log')
axes[0].set_xlabel('Number of parameter queries, $K$')
axes[0].set_ylabel('Steady-state time (s)')
axes[0].text(-0.16, 1.04, 'a', transform=axes[0].transAxes,
             fontsize=10, fontweight='bold')

for method in ['JI-DeepONet', 'RP-MIONet']:
    part = summary.loc[summary['method'] == method].sort_values('query_count')
    axes[1].plot(
        part['query_count'],
        part['speedup_over_pfdeonet'],
        color=colors[method],
        marker=markers[method],
        markersize=4.2,
        linewidth=1.25,
        label=f'{method} / PFDEONet',
        zorder=3,
    )

axes[1].axhline(1.0, color='#777777', linestyle='--', linewidth=0.9)
axes[1].set_xscale('log')
axes[1].set_ylim(0.75, 4.65)
axes[1].set_xlabel('Number of parameter queries, $K$')
axes[1].set_ylabel('Ratio of mean steady-state times')
axes[1].legend(loc='center right', frameon=True, fontsize=7)
axes[1].text(-0.16, 1.04, 'b', transform=axes[1].transAxes,
             fontsize=10, fontweight='bold')

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc='upper center',
    bbox_to_anchor=(0.5, 1.03),
    ncol=3,
    frameon=False,
    fontsize=8,
)

fig.subplots_adjust(left=0.10, right=0.985, bottom=0.19, top=0.82, wspace=0.34)

stem = FIGURE_DIR / 'nicholson_feature_reuse_timing'
fig.savefig(stem.with_suffix('.pdf'), bbox_inches='tight')
fig.savefig(stem.with_suffix('.svg'), bbox_inches='tight')
fig.savefig(stem.with_suffix('.png'), dpi=600, bbox_inches='tight')
fig.savefig(stem.with_suffix('.tiff'), dpi=600, bbox_inches='tight')
plt.close(fig)

# A compact audit file records the observed memory range for each method.
memory = (
    raw.groupby('method', as_index=False)['peak_memory_bytes']
    .agg(['min', 'max'])
    .reset_index()
)
memory.to_csv(FIGURE_DIR / 'feature_reuse_peak_memory.csv', index=False)
