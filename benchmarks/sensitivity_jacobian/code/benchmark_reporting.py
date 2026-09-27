"""CSV summaries and manuscript-ready figures for the fixed-checkpoint benchmark.

The module name is intentionally unique.  The reused experiment source trees also
contain a top-level ``reporting.py``; a distinct name prevents multiprocessing
workers from importing that unrelated module when they start with ``spawn``.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

import matplotlib.pyplot as plt
import scienceplots
import matplotlib
plt.style.use(['science','no-latex','grid',])
matplotlib.rcParams['font.family'] = 'Source Han Serif'
matplotlib.rcParams['font.family'] = 'Source Han Serif'


COLORS = {
    "without_sensitivity": "#4C72B0",
    "with_sensitivity": "#C44E52",
}
DISPLAY = {
    "without_sensitivity": "Without sensitivity supervision",
    "with_sensitivity": "With sensitivity supervision",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV file as dictionaries."""
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _save_figure(figure: Any, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_jacobian_errors(output_dir: Path, systems: list[str]) -> None:
    """Plot fixed-checkpoint held-out error distributions for both parameters."""
    figure, axes = plt.subplots(1, len(systems), figsize=(4.0 * len(systems), 3.3))
    if len(systems) == 1:
        axes = [axes]
    for panel, (axis, system) in enumerate(zip(axes, systems)):
        rows = read_csv(output_dir / system / "per_case_jacobian_metrics.csv")
        parameters: list[str] = []
        for row in rows:
            parameter = row["parameter"]
            if parameter != "overall" and parameter not in parameters:
                parameters.append(parameter)
        positions = np.arange(len(parameters), dtype=float)
        width = 0.27
        for offset, variant in ((-width / 1.6, "without_sensitivity"), (width / 1.6, "with_sensitivity")):
            values = [
                np.asarray(
                    [
                        float(row["relative_error"])
                        for row in rows
                        if row["parameter"] == parameter and row["variant"] == variant
                    ]
                )
                for parameter in parameters
            ]
            box = axis.boxplot(
                values,
                positions=positions + offset,
                widths=width,
                patch_artist=True,
                showfliers=False,
                medianprops={"color": "white", "linewidth": 1.2},
                whiskerprops={"color": COLORS[variant]},
                capprops={"color": COLORS[variant]},
            )
            for patch in box["boxes"]:
                patch.set_facecolor(COLORS[variant])
                patch.set_alpha(0.82)
        axis.set_yscale("log")
        axis.set_xticks(positions, parameters)
        axis.set_title(system.capitalize())
        axis.set_ylabel("Held-out Jacobian relative error" if panel == 0 else "")
        axis.text(-0.14, 1.03, chr(ord("a") + panel), transform=axis.transAxes, fontweight="bold")
    handles = [
        plt.Line2D([0], [0], color=COLORS[key], linewidth=6, label=DISPLAY[key])
        for key in ("without_sensitivity", "with_sensitivity")
    ]
    figure.legend(handles=handles, loc="upper center", ncol=2, frameon=False)
    figure.subplots_adjust(top=0.80, wspace=0.30)
    _save_figure(figure, output_dir / "figures" / "jacobian_error_comparison")


def plot_gradient_cosines(output_dir: Path, systems: list[str]) -> None:
    """Plot the distribution of Jacobian-only gradient-direction cosines."""
    figure, axes = plt.subplots(1, len(systems), figsize=(4.0 * len(systems), 3.3), sharey=True)
    if len(systems) == 1:
        axes = [axes]
    for panel, (axis, system) in enumerate(zip(axes, systems)):
        rows = read_csv(output_dir / system / "gradient_direction.csv")
        values = [
            np.asarray(
                [
                    float(row["jacobian_only_gradient_cosine"])
                    for row in rows
                    if row["variant"] == variant
                ]
            )
            for variant in ("without_sensitivity", "with_sensitivity")
        ]
        violin = axis.violinplot(values, positions=[0, 1], showmedians=True, widths=0.75)
        for body, variant in zip(
            violin["bodies"], ("without_sensitivity", "with_sensitivity")
        ):
            body.set_facecolor(COLORS[variant])
            body.set_edgecolor(COLORS[variant])
            body.set_alpha(0.72)
        axis.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
        axis.set_xticks([0, 1], ["Without", "With"])
        axis.set_ylim(-1.05, 1.05)
        axis.set_title(system.capitalize())
        axis.set_ylabel("Gradient-direction cosine" if panel == 0 else "")
        axis.text(-0.14, 1.03, chr(ord("a") + panel), transform=axis.transAxes, fontweight="bold")
    figure.subplots_adjust(top=0.86, wspace=0.18)
    _save_figure(figure, output_dir / "figures" / "gradient_cosine_comparison")


def plot_convergence(output_dir: Path, systems: list[str]) -> None:
    """Plot adjacent-level changes for finite-difference and solver refinements."""
    figure, axes = plt.subplots(1, 2, figsize=(8.2, 3.3), sharey=True)
    studies = ("finite_difference_step", "solver_step")
    titles = ("Finite-difference refinement", "Solver-step refinement")
    markers = ("o", "s", "^")
    for axis, study, title in zip(axes, studies, titles):
        for system_index, system in enumerate(systems):
            rows = [
                row
                for row in read_csv(output_dir / system / "convergence.csv")
                if row["study"] == study
            ]
            values = np.asarray([float(row["aggregate_relative_change"]) for row in rows])
            axis.scatter(
                np.arange(values.size) + 0.08 * system_index,
                values,
                label=system.capitalize(),
                marker=markers[system_index % len(markers)],
                s=28,
            )
        axis.set_yscale("log")
        axis.set_title(title)
        axis.set_xlabel("Adjacent refinement comparison")
    axes[0].set_ylabel("Aggregate relative change")
    axes[1].legend(frameon=False)
    axes[0].text(-0.12, 1.03, "a", transform=axes[0].transAxes, fontweight="bold")
    axes[1].text(-0.12, 1.03, "b", transform=axes[1].transAxes, fontweight="bold")
    figure.subplots_adjust(wspace=0.22)
    _save_figure(figure, output_dir / "figures" / "step_convergence")


def make_all_figures(output_dir: Path, systems: list[str]) -> None:
    """Create all figures after every requested system has completed."""
    if not systems:
        return
    plot_jacobian_errors(output_dir, systems)
    plot_gradient_cosines(output_dir, systems)
    plot_convergence(output_dir, systems)
