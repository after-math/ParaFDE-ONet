from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import scienceplots
import matplotlib
import numpy as np


plt.style.use(['science','no-latex','grid',])
matplotlib.rcParams['font.family']='Source Han Serif'
matplotlib.rcParams.update({
    "font.size": 7.2,
    "axes.titlesize": 8.0,
    "axes.labelsize": 7.5,
    "xtick.labelsize": 6.8,
    "ytick.labelsize": 6.8,
    "legend.fontsize": 6.4,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/paper/cstr/industrial_lm_20events"
OUTPUT = ROOT / "outputs/paper_figures/cstr_recalibration_workflow"
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

COLORS = {
    "ParaFDEONet-LM": "#C44E52",
    "DDE-LM": "#4C72B0",
    "DDE-DE": "#555555",
    "Static": "#D28E2B",
    "Oracle": "#2A9D6F",
}
METHOD_NAMES = {
    "ParaFDEONet LM": "ParaFDEONet-LM",
    "Direct LM": "DDE-LM",
    "Direct DE": "DDE-DE",
}


def load_inversion_rows() -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    with (RESULTS / "inversion_results.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            event = int(row["event"])
            if event < 0:
                continue
            rows.append(
                {
                    "event": event,
                    "noise": float(row["noise_fraction"]),
                    "method": METHOD_NAMES[row["method"]],
                    "error_percent": 100.0 * float(row["normalized_parameter_error"]),
                }
            )
    return rows


def load_cycle_times() -> tuple[np.ndarray, np.ndarray]:
    frozen: list[float] = []
    direct: list[float] = []
    for event_path in sorted((RESULTS / "events").glob("event_*.json")):
        with event_path.open() as stream:
            record = json.load(stream)
        cycle = record["pipeline"]["cycle_time_seconds"]
        frozen.append(float(cycle["ParaFDEONet"]))
        direct.append(float(cycle["Direct physics"]))
    return np.asarray(frozen), np.asarray(direct)


def panel_inversion(ax: plt.Axes) -> None:
    rows = load_inversion_rows()
    grouped: dict[tuple[float, str], list[tuple[int, float]]] = defaultdict(list)
    for row in rows:
        grouped[(float(row["noise"]), str(row["method"]))].append(
            (int(row["event"]), float(row["error_percent"]))
        )

    noises = [0.0, 0.01, 0.02, 0.05]
    methods = ["ParaFDEONet-LM", "DDE-LM", "DDE-DE"]
    offsets = [-0.20, 0.0, 0.20]
    markers = ["o", "s", "^"]
    for method, offset, marker in zip(methods, offsets, markers):
        for index, noise in enumerate(noises):
            values_with_event = sorted(grouped[(noise, method)])
            events = np.asarray([item[0] for item in values_with_event])
            values = np.asarray([item[1] for item in values_with_event])
            jitter = ((events % 7) - 3) * 0.008
            x = index + offset + jitter
            ax.scatter(
                x,
                values,
                s=8,
                marker=marker,
                facecolors="none",
                edgecolors=COLORS[method],
                linewidths=0.55,
                alpha=0.42,
                rasterized=True,
            )
            median = float(np.median(values))
            q25, q75 = np.quantile(values, [0.25, 0.75])
            ax.vlines(index + offset, q25, q75, color=COLORS[method], lw=1.35)
            ax.scatter(
                index + offset,
                median,
                s=21,
                marker=marker,
                color=COLORS[method],
                edgecolor="white",
                linewidth=0.35,
                zorder=5,
                label=method if index == 0 else None,
            )

    ax.set_yscale("log")
    ax.set_xticks(range(len(noises)), ["0", "0.01", "0.02", "0.05"])
    ax.set_xlabel("Noise fraction")
    ax.set_ylabel("Normalized parameter error (%)")
    ax.set_title("Parameter recalibration")
    ax.legend(
        loc="lower right",
        frameon=True,
        framealpha=0.92,
        handletextpad=0.35,
        labelspacing=0.25,
        borderpad=0.25,
    )


def panel_operating_map(ax: plt.Axes) -> None:
    maps = np.load(RESULTS / "representative_maps.npz")
    controls = maps["controls"]
    true_safe = maps["true_safe"]
    oracle_features = maps["oracle_features"]
    dilution = np.unique(controls[:, 0])
    coolant = np.unique(controls[:, 1])
    shape = (len(dilution), len(coolant))
    conversion = oracle_features[:, 1]
    productivity = controls[:, 0] * conversion
    safe_productivity = np.ma.masked_where(~true_safe.reshape(shape), productivity.reshape(shape))

    ax.set_facecolor("#F0F0F0")
    image = ax.pcolormesh(
        dilution,
        coolant,
        safe_productivity.T,
        shading="auto",
        cmap="YlGnBu",
        rasterized=True,
    )
    ax.contour(
        dilution,
        coolant,
        true_safe.reshape(shape).T.astype(float),
        levels=[0.5],
        colors="#444444",
        linewidths=0.75,
    )

    with (RESULTS / "industrial_summary.json").open() as stream:
        summary = json.load(stream)
    recommendations = summary["representative"]["pipeline"]["recommendations"]
    points = {
        "Static": recommendations["Static"],
        "Frozen = Direct": recommendations["ParaFDEONet"],
        "Oracle": recommendations["Oracle"],
    }
    point_style = {
        "Static": ("s", COLORS["Static"]),
        "Frozen = Direct": ("o", COLORS["ParaFDEONet-LM"]),
        "Oracle": ("*", COLORS["Oracle"]),
    }
    for label, point in points.items():
        marker, color = point_style[label]
        ax.scatter(
            point["D"],
            point["Tc"],
            s=45 if marker == "*" else 29,
            marker=marker,
            color=color,
            edgecolor="white",
            linewidth=0.55,
            zorder=5,
            label=label,
            clip_on=False,
        )

    ax.set_xlim(dilution.min(), dilution.max())
    ax.set_ylim(coolant.min(), coolant.max() + 0.025)
    ax.set_xlabel("Dilution rate $D$")
    ax.set_ylabel("Coolant temperature $T_c$")
    ax.set_title("Representative operating map")
    ax.legend(
        loc="lower right",
        frameon=True,
        framealpha=0.92,
        handletextpad=0.35,
        labelspacing=0.25,
        borderpad=0.25,
    )
    colorbar = ax.figure.colorbar(image, ax=ax, pad=0.02, fraction=0.05)
    colorbar.set_label("Feasible productivity", fontsize=6.8)
    colorbar.ax.tick_params(labelsize=6.2)


def panel_cycle_time(ax: plt.Axes) -> None:
    frozen, direct = load_cycle_times()
    for frozen_value, direct_value in zip(frozen, direct):
        ax.plot([0, 1], [frozen_value, direct_value], color="#B7B7B7", lw=0.55, alpha=0.7)
    ax.scatter(
        np.zeros_like(frozen), frozen, s=15, color=COLORS["ParaFDEONet-LM"],
        edgecolor="white", linewidth=0.35, zorder=3,
    )
    ax.scatter(
        np.ones_like(direct), direct, s=15, color=COLORS["DDE-LM"],
        edgecolor="white", linewidth=0.35, zorder=3,
    )
    medians = [float(np.median(frozen)), float(np.median(direct))]
    ax.plot([0, 1], medians, color="#222222", lw=1.15, zorder=4)
    ax.scatter([0, 1], medians, s=31, marker="D", color="#222222", zorder=5)
    ax.set_yscale("log")
    ax.set_xlim(-0.35, 1.35)
    ax.set_xticks([0, 1], ["Frozen\noperator", "Direct\nDDE"])
    ax.set_ylabel("Online cycle time (s)")
    ax.set_title("Twenty paired events")
    ax.text(
        0.5,
        0.42,
        "median speedup\n$28.55\\times$",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=6.8,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 1.5},
    )


def main() -> None:
    figure = plt.figure(figsize=(7.2, 2.55), constrained_layout=True)
    grid = figure.add_gridspec(1, 3, width_ratios=[1.15, 1.38, 0.82])
    axes = [figure.add_subplot(grid[0, index]) for index in range(3)]

    panel_inversion(axes[0])
    panel_operating_map(axes[1])
    panel_cycle_time(axes[2])

    for label, ax in zip("abc", axes):
        ax.text(-0.14, 1.06, label, transform=ax.transAxes, fontsize=9, fontweight="bold")

    for extension, options in {
        "pdf": {},
        "svg": {},
        "png": {"dpi": 600},
    }.items():
        figure.savefig(OUTPUT.with_suffix(f".{extension}"), bbox_inches="tight", **options)
    plt.close(figure)


if __name__ == "__main__":
    main()
