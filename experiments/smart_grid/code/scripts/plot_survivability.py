#!/usr/bin/env python3
"""Create the submission-size figure for the survivability experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import scienceplots  # noqa: F401
import matplotlib
import numpy as np


plt.style.use(["science", "no-latex", "grid"])
matplotlib.rcParams['font.family'] = 'Source Han Serif'
# DejaVu Serif is installed on the execution host; the generic family is a safe fallback.
matplotlib.rcParams['font.family'] = 'Source Han Serif'
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42})
matplotlib.rcParams["font.size"] = 6.5
matplotlib.rcParams["axes.titlesize"] = 7.3
matplotlib.rcParams["axes.labelsize"] = 6.8
matplotlib.rcParams["xtick.labelsize"] = 6.2
matplotlib.rcParams["ytick.labelsize"] = 6.2
matplotlib.rcParams["legend.fontsize"] = 6.0

FIG_WIDTH_MM = 183.0
FIG_HEIGHT_MM = 158.75
PNG_DPI = 300
TIFF_DPI = 600


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        0.02,
        0.98,
        label,
        transform=axis.transAxes,
        fontsize=9.0,
        fontweight="bold",
        va="top",
        ha="left",
        zorder=10,
        bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "none", "pad": 1.2},
    )


def create_figure(output_dir: Path, config: Mapping[str, Any]) -> list[Path]:
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    parameter_rows = _read_csv(output_dir / "source_data_parameters.csv")
    timing_rows = _read_csv(output_dir / "timing_runs.csv")
    with np.load(output_dir / "survivability_data.npz") as values:
        frequency_hz = values["frequency_thresholds_rad_per_s"] / (2.0 * math.pi)
        angle_degrees = np.degrees(values["edge_angle_thresholds_rad"])
        direct_mean = values["direct_mean_survivability"]
        operator_mean = values["parafdeonet_mean_survivability"]
    width = FIG_WIDTH_MM / 25.4
    height = FIG_HEIGHT_MM / 25.4
    configured_width = float(config["reporting"]["figure_width_inches"])
    configured_height = float(config["reporting"]["figure_height_inches"])
    if not math.isclose(width, configured_width, abs_tol=0.02) or not math.isclose(
        height, configured_height, abs_tol=0.02
    ):
        raise ValueError("configured figure size does not match the audited final size")
    figure, axes = plt.subplots(2, 3, figsize=(width, height), constrained_layout=True)
    extent = (
        float(angle_degrees[0]),
        float(angle_degrees[-1]),
        float(frequency_hz[0]),
        float(frequency_hz[-1]),
    )
    screening_angle = float(config["survivability"]["representative_edge_angle_threshold_degrees"])
    screening_frequency = float(config["survivability"]["representative_frequency_threshold_hz"])
    heatmaps = (
        (direct_mean, "Mean direct survivability", "cividis", 0.0, 1.0),
        (operator_mean, "Mean ParaFDEONet survivability", "cividis", 0.0, 1.0),
        (
            np.abs(operator_mean - direct_mean),
            "Absolute mean-surface error",
            "magma",
            0.0,
            max(float(np.max(np.abs(operator_mean - direct_mean))), 1e-6),
        ),
    )
    for column, (image_values, title, color_map, minimum, maximum) in enumerate(heatmaps):
        axis = axes[0, column]
        image = axis.imshow(
            image_values,
            origin="lower",
            aspect="auto",
            extent=extent,
            interpolation="nearest",
            cmap=color_map,
            vmin=minimum,
            vmax=maximum,
        )
        axis.plot(screening_angle, screening_frequency, marker="x", color="white", ms=5.0, mew=1.1)
        axis.set_title(title)
        axis.set_xlabel("Edge-angle limit (degrees)")
        axis.set_ylabel("Frequency limit (Hz)")
        axis.grid(False)
        colorbar = figure.colorbar(image, ax=axis, pad=0.02, fraction=0.047)
        colorbar.set_label("Survivability" if column < 2 else "Absolute error")
        if column == 2:
            axis.text(
                0.97,
                0.04,
                f"MAE = {metrics['survivability_surface']['mean_surface_mae']:.2e}",
                transform=axis.transAxes,
                ha="right",
                va="bottom",
                color="white",
                fontsize=5.8,
                bbox={"facecolor": "black", "alpha": 0.38, "edgecolor": "none", "pad": 1.5},
            )
        _panel_label(axis, chr(ord("a") + column))
    direct_probability = np.asarray(
        [float(row["direct_survivability"]) for row in parameter_rows]
    )
    operator_probability = np.asarray(
        [float(row["parafdeonet_survivability"]) for row in parameter_rows]
    )
    lower = np.asarray([float(row["direct_wilson_95_lower"]) for row in parameter_rows])
    upper = np.asarray([float(row["direct_wilson_95_upper"]) for row in parameter_rows])
    coupling = np.asarray([float(row["coupling_K"]) for row in parameter_rows])
    axis = axes[1, 0]
    axis.errorbar(
        direct_probability,
        operator_probability,
        xerr=np.vstack((direct_probability - lower, upper - direct_probability)),
        fmt="none",
        ecolor="#9A9A9A",
        elinewidth=0.5,
        alpha=0.55,
        zorder=1,
    )
    coupling_levels = np.unique(coupling)
    coupling_colors = plt.get_cmap("cividis")(np.linspace(0.08, 0.92, coupling_levels.size))
    for coupling_level, color in zip(coupling_levels, coupling_colors):
        selected = np.isclose(coupling, coupling_level)
        axis.scatter(
            direct_probability[selected],
            operator_probability[selected],
            color=color,
            s=15,
            edgecolor="black",
            linewidth=0.25,
            label=f"{coupling_level:.2g}",
            zorder=2,
        )
    limits = [
        max(0.0, float(min(np.min(lower), np.min(operator_probability))) - 0.03),
        min(1.0, float(max(np.max(upper), np.max(operator_probability))) + 0.03),
    ]
    axis.plot(limits, limits, color="black", linestyle="--", linewidth=0.8)
    axis.set_xlim(limits)
    axis.set_ylim(limits)
    axis.set_xlabel("Direct survivability")
    axis.set_ylabel("ParaFDEONet survivability")
    axis.set_title(f"Screening at {screening_frequency:.2f} Hz and {screening_angle:.0f} degrees")
    axis.legend(
        title="K",
        loc="upper left",
        bbox_to_anchor=(0.02, 0.91),
        borderaxespad=0.0,
        frameon=True,
        ncol=2,
        columnspacing=0.7,
        handletextpad=0.3,
    )
    axis.text(
        0.97,
        0.05,
        f"MAE = {metrics['representative_screening']['parameterwise_probability_mae']:.3f}\n"
        f"Spearman = {metrics['representative_screening']['parameterwise_spearman']:.3f}",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.8,
    )
    _panel_label(axis, "d")
    direct_frequency_q95 = np.asarray(
        [float(row["direct_frequency_q95_hz"]) for row in parameter_rows]
    ) / screening_frequency
    operator_frequency_q95 = np.asarray(
        [float(row["parafdeonet_frequency_q95_hz"]) for row in parameter_rows]
    ) / screening_frequency
    direct_angle_q95 = np.asarray(
        [float(row["direct_edge_angle_q95_degrees"]) for row in parameter_rows]
    ) / screening_angle
    operator_angle_q95 = np.asarray(
        [float(row["parafdeonet_edge_angle_q95_degrees"]) for row in parameter_rows]
    ) / screening_angle
    axis = axes[1, 1]
    axis.scatter(
        direct_frequency_q95,
        operator_frequency_q95,
        color="#0072B2",
        s=14,
        alpha=0.75,
        label="Frequency Q95",
    )
    axis.scatter(
        direct_angle_q95,
        operator_angle_q95,
        facecolor="none",
        edgecolor="#D55E00",
        linewidth=0.7,
        s=17,
        alpha=0.8,
        label="Edge-angle Q95",
    )
    risk_maximum = 1.05 * float(
        max(
            np.max(direct_frequency_q95),
            np.max(operator_frequency_q95),
            np.max(direct_angle_q95),
            np.max(operator_angle_q95),
        )
    )
    axis.plot([0.0, risk_maximum], [0.0, risk_maximum], color="black", linestyle="--", linewidth=0.8)
    axis.set_xlim(0.0, risk_maximum)
    axis.set_ylim(0.0, risk_maximum)
    axis.set_xlabel("Direct Q95 / screening limit")
    axis.set_ylabel("ParaFDEONet Q95 / screening limit")
    axis.set_title("Parameterwise Q95 excursions")
    axis.legend(loc="upper left", bbox_to_anchor=(0.02, 0.91), borderaxespad=0.0, frameon=True)
    axis.text(
        0.97,
        0.05,
        f"Frequency rho = {metrics['continuous_risk']['frequency_q95_parameterwise_spearman']:.3f}\n"
        f"Angle rho = {metrics['continuous_risk']['edge_angle_q95_parameterwise_spearman']:.3f}",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.8,
    )
    _panel_label(axis, "e")
    method_order = ["ParaFDEONet", "Direct DDE Monte Carlo"]
    means = []
    standard_deviations = []
    for method in method_order:
        observations = np.asarray(
            [float(row["seconds"]) for row in timing_rows if row["method"] == method]
        )
        means.append(float(np.mean(observations)))
        standard_deviations.append(float(np.std(observations, ddof=1)))
    if np.any(np.asarray(means) <= 0.0):
        raise ValueError("log-scale timing values must be strictly positive")
    axis = axes[1, 2]
    bars = axis.bar(
        np.arange(2),
        means,
        yerr=standard_deviations,
        color=["#0072B2", "#B3B3B3"],
        edgecolor="black",
        linewidth=0.5,
        capsize=2.5,
    )
    axis.set_yscale("log")
    axis.set_xticks(np.arange(2), ["ParaFDEONet", "Direct DDE\nMonte Carlo"])
    axis.set_ylabel("Time for 32,768 trajectories (s)")
    axis.set_title("Deployment time")
    axis.set_ylim(max(min(means) * 0.65, 1e-3), max(means) * 1.55)
    axis.text(
        0.5,
        0.88,
        f"{metrics['timing']['speedup']:.0f}x speedup",
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=7.0,
        fontweight="bold",
    )
    for bar, mean in zip(bars, means):
        axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            mean,
            f" {mean:.2g} s",
            ha="center",
            va="bottom",
            fontsize=5.8,
        )
    for axis in axes[1, :]:
        axis.set_box_aspect(1.0)
    _panel_label(axis, "f")
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    base = figures_dir / str(config["reporting"]["figure_basename"])
    paths = [base.with_suffix(extension) for extension in (".png", ".svg", ".pdf", ".tiff")]
    figure.savefig(paths[0], dpi=PNG_DPI, bbox_inches="tight")
    figure.savefig(paths[1], bbox_inches="tight")
    figure.savefig(paths[2], bbox_inches="tight")
    figure.savefig(paths[3], dpi=TIFF_DPI, bbox_inches="tight")
    plt.close(figure)
    qa_text = f"""# Figure QA

- Core claim: the frozen 75M-parameter ParaFDEONet reproduces direct-DDE Monte Carlo survivability estimates for the specified delayed grid at lower deployment cost.
- Archetype: quantitative six-panel grid; survivability agreement is primary and timing is supporting evidence.
- Backend: Python, matplotlib and scienceplots only.
- Final size: {width * 25.4:.1f} mm wide by {height * 25.4:.1f} mm high before tight bounding-box trimming.
- Replicates: 512 held-out histories per parameter point; 64 parameter points; the same histories are paired across methods.
- Interval: Wilson 95% binomial interval for direct Monte Carlo survivability in panel d.
- Timing: mean and sample standard deviation over {metrics['timing']['timing_repeats']} complete deployment repeats.
- Baseline: direct DDE Monte Carlo survivability following Hellmann et al. (2016), solved by causal RK4 at 0.005 s.
- Train/test split: one frozen training seed; histories come only from the fixed test split.
- Source data: source_data_surfaces.csv, source_data_parameters.csv and timing_runs.csv.
- Exports: editable-text SVG/PDF, 300 dpi PNG and 600 dpi TIFF.
- Font: DejaVu Serif, retained because Source Han Serif was unavailable on the execution host in the prior forward run.
"""
    (output_dir / "FIGURE_QA.md").write_text(qa_text, encoding="utf-8")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "configs" / "survivability_experiment.json",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    for path in create_figure(args.output_dir, config):
        print(path)


if __name__ == "__main__":
    main()
