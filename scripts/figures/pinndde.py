from pathlib import Path

import matplotlib.pyplot as plt
import scienceplots
import matplotlib
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, FuncFormatter

plt.style.use(['science','no-latex','grid',])
matplotlib.rcParams['font.family']='Source Han Serif'

matplotlib.rcParams.update(
    {
        "font.family": "Source Han Serif",
        "font.sans-serif": ["Source Han Serif", "Arial", "Helvetica"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "axes.linewidth": 0.7,
        "grid.linewidth": 0.35,
        "grid.alpha": 0.28,
        "savefig.transparent": False,
    }
)

PROJECT_DIR = Path(__file__).resolve().parents[2]
SOURCE_FILE = PROJECT_DIR / "results/paper/source_data/pinndde_full_state_source_data.xlsx"
OUTPUT_STEM = PROJECT_DIR / "outputs/paper_figures/pinndde_full_state_feasibility"
OUTPUT_STEM.parent.mkdir(parents=True, exist_ok=True)

SYSTEMS = ["Competition", "Delayed SEI", "Nicholson"]
NOISES = [0.005, 0.010, 0.020, 0.050]
COLORS = {
    "Competition": "#356D8A",
    "Delayed SEI": "#B06A55",
    "Nicholson": "#5D7C68",
}


def draw_case_distribution(ax, values_by_noise, color):
    """Draw every case together with the median and interquartile range."""
    for position, values in enumerate(values_by_noise):
        values = np.asarray(values, dtype=float)
        if values.size != 10:
            raise ValueError(f"Expected 10 cases, found {values.size}")
        if np.any(~np.isfinite(values)) or np.any(values <= 0):
            raise ValueError("Log-scale metrics must be finite and strictly positive")

        jitter = np.linspace(-0.105, 0.105, values.size)
        q1, median, q3 = np.quantile(values, [0.25, 0.50, 0.75])
        ax.scatter(
            position + jitter,
            values,
            s=10,
            color=color,
            alpha=0.52,
            edgecolors="none",
            rasterized=False,
            zorder=2,
        )
        ax.vlines(position, q1, q3, color=color, linewidth=2.2, zorder=3)
        ax.hlines([q1, q3], position - 0.055, position + 0.055, color=color, linewidth=0.8, zorder=3)
        ax.scatter(
            [position],
            [median],
            marker="D",
            s=22,
            facecolor="white",
            edgecolor=color,
            linewidth=1.15,
            zorder=4,
        )

    ax.set_xticks(range(len(NOISES)), [f"{noise:.3f}" for noise in NOISES])
    ax.set_xlim(-0.35, len(NOISES) - 0.65)
    ax.set_yscale("log")
    ax.grid(True, which="major", axis="y")
    ax.grid(False, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main():
    data = pd.read_excel(SOURCE_FILE, sheet_name="Raw", engine="openpyxl")
    if len(data) != 120:
        raise ValueError(f"Expected 120 full-state rows, found {len(data)}")
    if set(data["status"]) != {"completed"}:
        raise ValueError("The selected data contain an incomplete request")

    expected_groups = data.groupby(["system", "noise_standard_deviation"]).size()
    if len(expected_groups) != 12 or not np.all(expected_groups.to_numpy() == 10):
        raise ValueError("Expected 12 system-noise groups with 10 cases in each group")

    width_mm = 183
    height_mm = 112
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(width_mm / 25.4, height_mm / 25.4),
        sharey="row",
        constrained_layout=False,
    )

    for column, system in enumerate(SYSTEMS):
        subset = data[data["system"] == system]
        parameter_values = [
            subset[np.isclose(subset["noise_standard_deviation"], noise)][
                "parameter_normalized_rmse"
            ].to_numpy()
            for noise in NOISES
        ]
        time_values = [
            subset[np.isclose(subset["noise_standard_deviation"], noise)][
                "online_end_to_end_seconds"
            ].to_numpy()
            for noise in NOISES
        ]

        draw_case_distribution(axes[0, column], parameter_values, COLORS[system])
        draw_case_distribution(axes[1, column], time_values, COLORS[system])
        axes[0, column].set_title(system, pad=5, fontweight="semibold")
        axes[1, column].set_xlabel("Noise standard deviation")

    for ax in axes[0, :]:
        ax.set_ylim(0.0025, 1.1)
        ax.yaxis.set_major_locator(FixedLocator([0.003, 0.01, 0.03, 0.1, 0.3, 1.0]))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    for ax in axes[1, :]:
        ax.set_ylim(600, 25000)
        ax.yaxis.set_major_locator(FixedLocator([1000, 3000, 10000]))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))

    axes[0, 0].set_ylabel("Normalized parameter RMSE")
    axes[1, 0].set_ylabel("Online time (s)")

    panel_labels = ["a", "b", "c", "d", "e", "f"]
    for label, ax in zip(panel_labels, axes.flat):
        ax.text(
            -0.19,
            1.05,
            label,
            transform=ax.transAxes,
            fontsize=8,
            fontweight="bold",
            va="bottom",
            ha="left",
        )

    fig.subplots_adjust(left=0.085, right=0.995, bottom=0.13, top=0.91, wspace=0.16, hspace=0.32)
    OUTPUT_STEM.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_STEM.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(OUTPUT_STEM.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(OUTPUT_STEM.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
