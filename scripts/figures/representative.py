#!/usr/bin/env python3
"""Combine the three forward and inverse representative examples."""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.pyplot as plt
import scienceplots
import matplotlib
import numpy as np
import pandas as pd


plt.style.use(['science','no-latex','grid',])
matplotlib.rcParams['font.family']='Source Han Serif'
matplotlib.rcParams.update(
    {
        "font.serif": ["Source Han Serif", "Arial", "Helvetica"],
        "font.size": 6.2,
        "axes.titlesize": 6.8,
        "axes.labelsize": 6.4,
        "xtick.labelsize": 5.7,
        "ytick.labelsize": 5.7,
        "legend.fontsize": 5.8,
        "axes.linewidth": 0.55,
        "grid.linewidth": 0.38,
        "grid.alpha": 0.45,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)


ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = ROOT / "outputs" / "paper_figures"
SOURCE_DIR = ROOT / "results/paper/source_data"

REFERENCE = "#202020"
PREDICTION = "#C44E52"
DIRECT_LM = "#4C72B0"
DIRECT_DE = "#55A868"
OBSERVATIONS = "#777777"
STATE_COLORS = ("#0072B2", "#E69F00", "#009E73", "#8172B2")


def save_figure(figure: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(figure)


def load_competition_forward() -> pd.DataFrame:
    """Read the archived curve data recovered from the original vector figure.

    This is a plotting source, not a replacement for full-precision solver output.
    """
    return pd.read_csv(SOURCE_DIR / "competition_forward_recovered_from_svg.csv")


def load_forward_sources() -> list[dict[str, object]]:
    competition = load_competition_forward()
    sei = pd.read_csv(SOURCE_DIR / "sei_forward_minimum_case.csv")
    nicholson = pd.read_csv(SOURCE_DIR / "nicholson_forward_minimum_case.csv")
    return [
        {
            "name": "Competition",
            "source": competition,
            "states": [("x1", r"Species $x_1$"), ("x2", r"Species $x_2$")],
        },
        {
            "name": "Delayed SEI",
            "source": sei,
            "states": [("S", "Susceptible"), ("E", "Exposed"), ("I", "Infectious")],
        },
        {
            "name": "Four-patch Nicholson",
            "source": nicholson,
            "states": [(f"patch_{index}", f"Patch {index}") for index in range(1, 5)],
        },
    ]


def plot_forward() -> None:
    systems = load_forward_sources()
    figure = plt.figure(figsize=(7.2, 6.55), constrained_layout=True)
    subfigures = figure.subfigures(3, 1, hspace=0.075)
    global_handles = None
    panel_letter = ord("a")

    for row_index, (subfigure, system) in enumerate(zip(subfigures, systems)):
        source = system["source"]
        states = system["states"]
        axes = subfigure.subplots(1, len(states) + 1)
        time = source["time"].to_numpy(dtype=float)
        for state_index, ((key, title), axis) in enumerate(zip(states, axes[:-1])):
            reference_line = axis.plot(
                time,
                source[f"{key}_reference"],
                color=REFERENCE,
                linewidth=1.05,
                label="Reference",
                zorder=2,
            )[0]
            prediction_line = axis.plot(
                time,
                source[f"{key}_prediction"],
                color=PREDICTION,
                linewidth=0.95,
                linestyle="--",
                label="ParaFDEONet",
                zorder=3,
            )[0]
            if global_handles is None:
                global_handles = (reference_line, prediction_line)
            axis.set_title(title, pad=2)
            axis.set_xlim(float(time[0]), float(time[-1]))
            axis.set_xlabel("Time")
            if state_index == 0:
                axis.set_ylabel("State value")
            axis.ticklabel_format(axis="y", style="plain", useOffset=False)
            axis.text(
                -0.18,
                1.08,
                chr(panel_letter),
                transform=axis.transAxes,
                fontweight="bold",
                fontsize=7.2,
                va="top",
            )
            panel_letter += 1

        error_axis = axes[-1]
        for state_index, (key, title) in enumerate(states):
            error_axis.plot(
                time,
                source[f"{key}_absolute_error"],
                color=STATE_COLORS[state_index],
                linewidth=0.82,
                label=title.replace("Species ", ""),
            )
        error_axis.set_title("Pointwise absolute error", pad=2)
        error_axis.set_xlim(float(time[0]), float(time[-1]))
        error_axis.set_xlabel("Time")
        error_axis.set_ylabel("Absolute error")
        error_axis.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2), useMathText=True)
        error_axis.legend(loc="upper right", ncol=1, frameon=False, handlelength=1.2)
        error_axis.text(
            -0.18,
            1.08,
            chr(panel_letter),
            transform=error_axis.transAxes,
            fontweight="bold",
            fontsize=7.2,
            va="top",
        )
        panel_letter += 1
        subfigure.suptitle(system["name"], x=0.005, ha="left", fontsize=7.2, fontweight="bold")

    figure.legend(
        global_handles,
        ("Reference", "ParaFDEONet"),
        loc="outside upper center",
        ncol=2,
        frameon=False,
        handlelength=2.0,
        columnspacing=1.5,
    )
    save_figure(figure, FIGURE_DIR / "forward_representative_three_systems")


def load_inverse_sources() -> list[dict[str, object]]:
    competition_path = SOURCE_DIR / "competition_inverse_reconstruction.csv"
    competition = pd.read_csv(competition_path)
    competition = competition.rename(
        columns={
            "reference_x1": "reference_state_1",
            "reference_x2": "reference_state_2",
            "pfdeonet_x1": "pfdeonet_state_1",
            "pfdeonet_x2": "pfdeonet_state_2",
            "direct_lm_x1": "direct_lm_state_1",
            "direct_lm_x2": "direct_lm_state_2",
            "direct_de_x1": "direct_de_state_1",
            "direct_de_x2": "direct_de_state_2",
            "observation_x1": "observation_state_1",
            "observation_x2": "observation_state_2",
        }
    )
    sei = pd.read_csv(SOURCE_DIR / "sei_inverse_reconstruction.csv")
    nicholson = pd.read_csv(SOURCE_DIR / "nicholson_inverse_reconstruction.csv")
    # Preserve the archived reconstruction values; normalize column names only.
    method_columns = {
        f"{source}_state_{index}": f"{target}_state_{index}"
        for source, target in (
            ("frozen_lm", "pfdeonet"), ("lm", "direct_lm"), ("de", "direct_de")
        )
        for index in range(1, 5)
    }
    competition = competition.rename(columns=method_columns)
    sei = sei.rename(columns=method_columns)
    nicholson = nicholson.rename(columns=method_columns)
    return [
        {"name": "Competition", "source": competition, "titles": [r"State $x_1$", r"State $x_2$"]},
        {"name": "Delayed SEI", "source": sei, "titles": ["Susceptible", "Exposed", "Infectious"]},
        {"name": "Four-patch Nicholson", "source": nicholson, "titles": [f"Patch {index}" for index in range(1, 5)]},
    ]


def plot_inverse() -> None:
    systems = load_inverse_sources()
    figure = plt.figure(figsize=(7.2, 8.45), constrained_layout=True)
    subfigures = figure.subfigures(3, 1, hspace=0.08)
    legend_handles = None
    panel_letter = ord("a")

    for subfigure, system in zip(subfigures, systems):
        source = system["source"]
        titles = system["titles"]
        axes = subfigure.subplots(
            2,
            len(titles),
            sharex="col",
            gridspec_kw={"height_ratios": (2.0, 0.9)},
        )
        if len(titles) == 1:
            axes = np.asarray(axes).reshape(2, 1)
        time = source["time"].to_numpy(dtype=float)
        for state_index, title in enumerate(titles, start=1):
            top = axes[0, state_index - 1]
            bottom = axes[1, state_index - 1]
            reference_line = top.plot(
                time,
                source[f"reference_state_{state_index}"],
                color=REFERENCE,
                linewidth=1.0,
                label="Reference",
                zorder=4,
            )[0]
            pf_line = top.plot(
                time,
                source[f"pfdeonet_state_{state_index}"],
                color=PREDICTION,
                linewidth=0.9,
                linestyle="--",
                label="ParaFDEONet-LM",
                zorder=3,
            )[0]
            lm_line = top.plot(
                time,
                source[f"direct_lm_state_{state_index}"],
                color=DIRECT_LM,
                linewidth=0.9,
                linestyle=":",
                marker="s",
                markersize=1.25,
                markerfacecolor="white",
                markeredgewidth=0.4,
                markevery=(40, 80),
                label="DDE-LM",
                zorder=2.7,
            )[0]
            de_line = top.plot(
                time,
                source[f"direct_de_state_{state_index}"],
                color=DIRECT_DE,
                linewidth=0.82,
                linestyle="-.",
                marker="D",
                markersize=1.35,
                markerfacecolor="white",
                markeredgewidth=0.4,
                markevery=80,
                label="DDE-DE",
                zorder=2.5,
            )[0]
            observed = source[f"observation_state_{state_index}"].to_numpy(dtype=float)
            mask = np.isfinite(observed)
            observation_points = top.scatter(
                time[mask],
                observed[mask],
                s=7.5,
                facecolors="white",
                edgecolors=OBSERVATIONS,
                linewidths=0.45,
                label="Noisy observations",
                zorder=5,
            )
            if legend_handles is None:
                legend_handles = (
                    reference_line,
                    pf_line,
                    lm_line,
                    de_line,
                    observation_points,
                )
            top.set_title(title, pad=2)
            if state_index == 1:
                top.set_ylabel("State value")
                bottom.set_ylabel("Absolute error")
            top.text(
                -0.15,
                1.07,
                chr(panel_letter),
                transform=top.transAxes,
                fontweight="bold",
                fontsize=7.1,
                va="top",
            )
            panel_letter += 1
            reference = source[f"reference_state_{state_index}"].to_numpy(dtype=float)
            bottom.plot(
                time,
                np.abs(source[f"pfdeonet_state_{state_index}"].to_numpy(dtype=float) - reference),
                color=PREDICTION,
                linewidth=0.85,
            )
            bottom.plot(
                time,
                np.abs(source[f"direct_lm_state_{state_index}"].to_numpy(dtype=float) - reference),
                color=DIRECT_LM,
                linewidth=0.85,
                linestyle=":",
                marker="s",
                markersize=1.2,
                markerfacecolor="white",
                markeredgewidth=0.4,
                markevery=(40, 80),
            )
            bottom.plot(
                time,
                np.abs(source[f"direct_de_state_{state_index}"].to_numpy(dtype=float) - reference),
                color=DIRECT_DE,
                linewidth=0.78,
                linestyle="-.",
                marker="D",
                markersize=1.25,
                markerfacecolor="white",
                markeredgewidth=0.4,
                markevery=80,
            )
            bottom.set_xlabel("Time")
            bottom.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2), useMathText=True)
        subfigure.suptitle(system["name"], x=0.005, ha="left", fontsize=7.2, fontweight="bold")

    figure.legend(
        legend_handles,
        ("Reference", "ParaFDEONet-LM", "DDE-LM", "DDE-DE", "Noisy observations"),
        loc="outside upper center",
        ncol=5,
        frameon=False,
        handlelength=2.0,
        columnspacing=1.3,
    )
    save_figure(figure, FIGURE_DIR / "inverse_representative_three_systems")


def main() -> None:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    plot_forward()
    plot_inverse()
    metadata = json.loads(
        (SOURCE_DIR / "nicholson_forward_minimum_case_metadata.json").read_text()
    )
    print(json.dumps({"nicholson_forward": metadata}, indent=2))


if __name__ == "__main__":
    main()
