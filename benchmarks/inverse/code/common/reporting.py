"""Generate compact diagnostic figures from aggregated CSV outputs."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401
import numpy as np

plt.style.use(["science", "no-latex", "grid"])
matplotlib.rcParams['font.family'] = 'Source Han Serif'
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams['font.family'] = 'Source Han Serif'


COLORS = {"frozen": "#0072B2", "lm": "#009E73", "de": "#D55E00"}
LABELS = {
    "frozen": "Frozen operator",
    "lm": "Direct LM",
    "de": "Direct DE",
}
METHOD_ORDER = ("frozen", "lm", "de")


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _save(fig: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def plot_noise_accuracy(full_dir: Path, figure_dir: Path) -> None:
    rows = _read_csv(full_dir / "per_panel_results.csv")
    if not rows:
        return
    fig, axis = plt.subplots(figsize=(5.2, 3.6))
    for method in METHOD_ORDER:
        method_rows = [row for row in rows if row["method_key"] == method]
        if not method_rows:
            continue
        noises = sorted({float(row["noise_standard_deviation"]) for row in method_rows})
        means, lower, upper = [], [], []
        for sigma in noises:
            values = np.asarray(
                [
                    float(row["parameter_normalized_rmse_panel_mean"])
                    for row in method_rows
                    if float(row["noise_standard_deviation"]) == sigma
                ]
            )
            mean = float(np.mean(values))
            half = 1.96 * float(np.std(values, ddof=1)) / np.sqrt(values.size) if values.size > 1 else 0.0
            means.append(mean)
            lower.append(mean - half)
            upper.append(mean + half)
        axis.plot(noises, means, marker="o", color=COLORS[method], label=LABELS[method])
        axis.fill_between(noises, lower, upper, color=COLORS[method], alpha=0.16)
    axis.set_xlabel("Gaussian noise standard deviation")
    axis.set_ylabel("Parameter normalized RMSE")
    axis.legend(frameon=False)
    axis.grid(alpha=0.2)
    _save(fig, figure_dir / "noise_parameter_accuracy")


def plot_amortized_timing(full_dir: Path, figure_dir: Path) -> None:
    rows = _read_csv(full_dir / "amortized_timing.csv")
    if not rows:
        return
    fig, axis = plt.subplots(figsize=(5.2, 3.6))
    for method in METHOD_ORDER:
        method_rows = [row for row in rows if row["method_key"] == method]
        ks = sorted({int(row["request_count_k"]) for row in method_rows})
        values = [
            np.median(
                [
                    float(row["online_amortized_seconds"])
                    for row in method_rows
                    if int(row["request_count_k"]) == k
                ]
            )
            for k in ks
        ]
        if ks:
            axis.plot(ks, values, marker="o", color=COLORS[method], label=LABELS[method])
    axis.set_xlabel("Repeated inverse requests, K")
    axis.set_ylabel("Online amortized seconds per request")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.legend(frameon=False)
    axis.grid(alpha=0.2, which="both")
    _save(fig, figure_dir / "amortized_online_time")


def _plot_method_pair(
    rows: list[dict[str, str]],
    figure_dir: Path,
    method_x: str,
    method_y: str,
    output_name: str,
) -> None:
    lookup = {
        (
            row["method_key"],
            int(row["panel_index"]),
            int(row["case_index"]),
            float(row["noise_standard_deviation"]),
        ): float(row["parameter_normalized_rmse"])
        for row in rows
        if row.get("parameter_normalized_rmse") not in (None, "", "nan")
    }
    pairs = []
    for (method, panel, case, noise), value_y in lookup.items():
        if method != method_y:
            continue
        value_x = lookup.get((method_x, panel, case, noise))
        if value_x is not None and np.isfinite(value_y) and np.isfinite(value_x):
            pairs.append((value_x, value_y, noise))
    if not pairs:
        return
    fig, axis = plt.subplots(figsize=(4.2, 4.0))
    for noise in sorted({row[2] for row in pairs}):
        selected = np.asarray([(x, y) for x, y, value in pairs if value == noise])
        axis.scatter(selected[:, 0], selected[:, 1], s=12, alpha=0.55, label=f"σ={noise:g}")
    maximum = max(max(x, y) for x, y, _ in pairs)
    axis.plot([0.0, maximum], [0.0, maximum], linestyle="--", color="black", linewidth=0.8)
    axis.set_xlabel(f"{LABELS[method_x]} parameter NRMSE")
    axis.set_ylabel(f"{LABELS[method_y]} parameter NRMSE")
    axis.set_aspect("equal", adjustable="box")
    axis.legend(frameon=False, fontsize=8)
    axis.grid(alpha=0.2)
    _save(fig, figure_dir / output_name)


def plot_paired_scatter(full_dir: Path, figure_dir: Path) -> None:
    rows = _read_csv(full_dir / "per_case_results.csv")
    _plot_method_pair(
        rows, figure_dir, "lm", "frozen", "paired_parameter_error"
    )
    _plot_method_pair(
        rows, figure_dir, "de", "frozen", "paired_parameter_error_frozen_vs_de"
    )
    _plot_method_pair(
        rows, figure_dir, "de", "lm", "paired_parameter_error_lm_vs_de"
    )


def generate_figures(full_dir: Path) -> None:
    figure_dir = full_dir.parent / "figures"
    plot_noise_accuracy(full_dir, figure_dir)
    plot_amortized_timing(full_dir, figure_dir)
    plot_paired_scatter(full_dir, figure_dir)
