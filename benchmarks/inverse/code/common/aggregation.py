"""Panel-level summaries, paired bootstrap and amortized timing."""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from common.io_utils import read_json, write_csv, write_json


def _finite(values: Iterable[Any]) -> np.ndarray:
    result = np.asarray([float(value) for value in values], dtype=np.float64)
    return result[np.isfinite(result)]


def collect_results(full_dir: Path) -> list[dict[str, Any]]:
    paths = sorted(full_dir.glob("panel_*/jobs/*/case_*/noise_*/result.json"))
    rows = [read_json(path) for path in paths]
    return sorted(
        rows,
        key=lambda row: (
            str(row["method_key"]),
            int(row["panel_index"]),
            int(row["case_index"]),
            float(row["noise_standard_deviation"]),
        ),
    )


def panel_macro_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, float, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[
            (
                str(row["method_key"]),
                float(row["noise_standard_deviation"]),
                int(row["panel_index"]),
            )
        ].append(row)
    result = []
    for (method, sigma, panel), group in sorted(groups.items()):
        parameter = _finite(row.get("parameter_normalized_rmse", np.nan) for row in group)
        trajectory = _finite(row.get("trajectory_relative_l2", np.nan) for row in group)
        timing = _finite(row.get("warm_online_seconds", np.nan) for row in group)
        candidate_evaluations = _finite(
            row.get(
                "candidate_objective_evaluations",
                float(row.get("direct_trajectory_solves", np.nan))
                / max(float(row.get("history_count", 1)), 1.0),
            )
            for row in group
        )
        trajectory_solves = _finite(
            row.get("direct_trajectory_solves", np.nan) for row in group
        )
        solver_calls = _finite(row.get("solver_batch_calls", np.nan) for row in group)
        row_result = {
                "method_key": method,
                "noise_standard_deviation": sigma,
                "panel_index": panel,
                "case_count_total": len(group),
                "case_count_successful": int(parameter.size),
                "failure_count": int(len(group) - parameter.size),
                "parameter_normalized_rmse_panel_mean": (
                    float(np.mean(parameter)) if parameter.size else np.nan
                ),
                "trajectory_relative_l2_panel_mean": (
                    float(np.mean(trajectory)) if trajectory.size else np.nan
                ),
                "warm_online_seconds_panel_mean": (
                    float(np.mean(timing)) if timing.size else np.nan
                ),
                "warm_online_seconds_panel_median": (
                    float(np.median(timing)) if timing.size else np.nan
                ),
                "candidate_objective_evaluations_panel_mean": (
                    float(np.mean(candidate_evaluations))
                    if candidate_evaluations.size else np.nan
                ),
                "direct_trajectory_solves_panel_mean": (
                    float(np.mean(trajectory_solves))
                    if trajectory_solves.size else np.nan
                ),
                "solver_batch_calls_panel_mean": (
                    float(np.mean(solver_calls)) if solver_calls.size else np.nan
                ),
                "boundary_count": int(
                    sum(int(row.get("boundary_estimate", 0)) for row in group)
                ),
            }
        component_keys = sorted(
            key
            for row in group
            for key in row
            if str(key).startswith("normalized_absolute_error_")
        )
        for key in sorted(set(component_keys)):
            values = _finite(row.get(key, np.nan) for row in group)
            row_result[f"{key}_panel_mean"] = (
                float(np.mean(values)) if values.size else np.nan
            )
        result.append(row_result)
    return result


def noise_summary_rows(panel_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, float], list[Mapping[str, Any]]] = defaultdict(list)
    for row in panel_rows:
        groups[(str(row["method_key"]), float(row["noise_standard_deviation"]))].append(row)
    result = []
    for (method, sigma), group in sorted(groups.items()):
        parameter = _finite(
            row["parameter_normalized_rmse_panel_mean"] for row in group
        )
        timing = _finite(row["warm_online_seconds_panel_median"] for row in group)
        candidate_evaluations = _finite(
            row.get("candidate_objective_evaluations_panel_mean", np.nan)
            for row in group
        )
        trajectory_solves = _finite(
            row.get("direct_trajectory_solves_panel_mean", np.nan) for row in group
        )
        solver_calls = _finite(
            row.get("solver_batch_calls_panel_mean", np.nan) for row in group
        )
        row_result = {
                "method_key": method,
                "noise_standard_deviation": sigma,
                "panel_count": len(group),
                "parameter_nrmse_panel_macro_mean": float(np.mean(parameter)),
                "parameter_nrmse_panel_standard_deviation": (
                    float(np.std(parameter, ddof=1)) if parameter.size > 1 else 0.0
                ),
                "warm_online_seconds_panel_median_mean": float(np.mean(timing)),
                "candidate_objective_evaluations_panel_macro_mean": (
                    float(np.mean(candidate_evaluations))
                    if candidate_evaluations.size else np.nan
                ),
                "direct_trajectory_solves_panel_macro_mean": (
                    float(np.mean(trajectory_solves))
                    if trajectory_solves.size else np.nan
                ),
                "solver_batch_calls_panel_macro_mean": (
                    float(np.mean(solver_calls)) if solver_calls.size else np.nan
                ),
                "total_failures": int(sum(int(row["failure_count"]) for row in group)),
                "total_boundary_estimates": int(
                    sum(int(row["boundary_count"]) for row in group)
                ),
            }
        component_keys = sorted(
            key
            for row in group
            for key in row
            if str(key).startswith("normalized_absolute_error_")
            and str(key).endswith("_panel_mean")
        )
        for key in sorted(set(component_keys)):
            values = _finite(row.get(key, np.nan) for row in group)
            row_result[f"{key.removesuffix('_panel_mean')}_panel_macro_mean"] = (
                float(np.mean(values)) if values.size else np.nan
            )
        result.append(row_result)
    return result


def paired_comparisons(
    panel_rows: Sequence[Mapping[str, Any]], repetitions: int, seed: int
) -> list[dict[str, Any]]:
    lookup = {
        (
            str(row["method_key"]),
            float(row["noise_standard_deviation"]),
            int(row["panel_index"]),
        ): float(row["parameter_normalized_rmse_panel_mean"])
        for row in panel_rows
    }
    noises = sorted({float(row["noise_standard_deviation"]) for row in panel_rows})
    rng = np.random.default_rng(seed)
    result = []
    for sigma in noises:
        panels = sorted(
            panel
            for method, noise, panel in lookup
            if method == "frozen"
            and noise == sigma
            and ("lm", noise, panel) in lookup
            and np.isfinite(lookup[("frozen", noise, panel)])
            and np.isfinite(lookup[("lm", noise, panel)])
        )
        if not panels:
            continue
        frozen = np.asarray([lookup[("frozen", sigma, panel)] for panel in panels])
        lm = np.asarray([lookup[("lm", sigma, panel)] for panel in panels])
        difference = frozen - lm
        draws = np.empty(repetitions, dtype=np.float64)
        for index in range(repetitions):
            selected = rng.integers(0, len(panels), size=len(panels))
            draws[index] = float(np.mean(difference[selected]))
        lm_mean = float(np.mean(lm))
        result.append(
            {
                "noise_standard_deviation": sigma,
                "paired_panel_count": len(panels),
                "frozen_panel_macro_mean": float(np.mean(frozen)),
                "lm_panel_macro_mean": lm_mean,
                "paired_difference_frozen_minus_lm": float(np.mean(difference)),
                "paired_difference_ci95_lower": float(np.percentile(draws, 2.5)),
                "paired_difference_ci95_upper": float(np.percentile(draws, 97.5)),
                "relative_difference": float(np.mean(difference) / max(lm_mean, 1e-15)),
                "within_five_percent_noninferiority_margin": int(
                    float(np.mean(difference) / max(lm_mean, 1e-15)) <= 0.05
                ),
            }
        )
    return result


def all_pairwise_comparisons(
    panel_rows: Sequence[Mapping[str, Any]],
    repetitions: int,
    seed: int,
    noninferiority_margin: float,
) -> list[dict[str, Any]]:
    lookup = {
        (
            str(row["method_key"]),
            float(row["noise_standard_deviation"]),
            int(row["panel_index"]),
        ): float(row["parameter_normalized_rmse_panel_mean"])
        for row in panel_rows
    }
    preferred_order = ["frozen", "lm", "de"]
    available = {method for method, _, _ in lookup}
    methods = [method for method in preferred_order if method in available]
    methods.extend(sorted(available - set(methods)))
    noises = sorted({noise for _, noise, _ in lookup})
    rng = np.random.default_rng(seed)
    result = []
    for sigma in noises:
        for method_a, method_b in combinations(methods, 2):
            panels = sorted(
                panel
                for method, noise, panel in lookup
                if method == method_a
                and noise == sigma
                and (method_b, noise, panel) in lookup
                and np.isfinite(lookup[(method_a, noise, panel)])
                and np.isfinite(lookup[(method_b, noise, panel)])
            )
            if not panels:
                continue
            values_a = np.asarray(
                [lookup[(method_a, sigma, panel)] for panel in panels]
            )
            values_b = np.asarray(
                [lookup[(method_b, sigma, panel)] for panel in panels]
            )
            differences = values_a - values_b
            draws = np.empty(repetitions, dtype=np.float64)
            for index in range(repetitions):
                selected = rng.integers(0, len(panels), size=len(panels))
                draws[index] = float(np.mean(differences[selected]))
            mean_b = float(np.mean(values_b))
            ci_lower = float(np.percentile(draws, 2.5))
            ci_upper = float(np.percentile(draws, 97.5))
            margin_absolute = float(noninferiority_margin * mean_b)
            result.append(
                {
                    "noise_standard_deviation": float(sigma),
                    "method_a": method_a,
                    "method_b": method_b,
                    "paired_panel_count": len(panels),
                    "method_a_panel_macro_mean": float(np.mean(values_a)),
                    "method_b_panel_macro_mean": mean_b,
                    "paired_difference_a_minus_b": float(np.mean(differences)),
                    "paired_difference_ci95_lower": ci_lower,
                    "paired_difference_ci95_upper": ci_upper,
                    "relative_difference_vs_method_b": float(
                        np.mean(differences) / max(mean_b, 1e-15)
                    ),
                    "noninferiority_margin_relative": float(noninferiority_margin),
                    "noninferiority_margin_absolute": margin_absolute,
                    "method_a_noninferior_to_method_b": int(
                        ci_upper <= margin_absolute
                    ),
                }
            )
    return result


def amortized_timing_rows(
    rows: Sequence[Mapping[str, Any]],
    full_dir: Path,
    requested_k: Sequence[int],
) -> list[dict[str, Any]]:
    deployments = [
        read_json(path)
        for path in sorted((full_dir / "deployment_initialization").glob("worker_*.json"))
    ]
    deploy_values = _finite(
        row.get("total_deployment_initialization_seconds", np.nan)
        for row in deployments
    )
    mean_deploy = float(np.mean(deploy_values)) if deploy_values.size else 0.0
    groups: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["method_key"]), int(row["panel_index"]))].append(row)
    result = []
    for (method, panel), group in sorted(groups.items()):
        ordered = sorted(
            group,
            key=lambda row: (
                int(row["case_index"]), float(row["noise_standard_deviation"])
            ),
        )
        times = _finite(row.get("warm_online_seconds", np.nan) for row in ordered)
        if not times.size:
            continue
        panel_seconds = (
            float(ordered[0].get("panel_initialization_seconds", 0.0))
            if method == "frozen"
            else 0.0
        )
        deployment_seconds = mean_deploy if method == "frozen" else 0.0
        cumulative = np.cumsum(times)
        for value in requested_k:
            k = int(value)
            if k > times.size:
                continue
            result.append(
                {
                    "method_key": method,
                    "panel_index": panel,
                    "request_count_k": k,
                    "deployment_seconds": deployment_seconds,
                    "panel_initialization_seconds": panel_seconds,
                    "warm_query_sum_seconds": float(cumulative[k - 1]),
                    "online_amortized_seconds": float(
                        (panel_seconds + cumulative[k - 1]) / k
                    ),
                    "total_amortized_seconds": float(
                        (deployment_seconds + panel_seconds + cumulative[k - 1]) / k
                    ),
                }
            )
    return result


def aggregate(
    full_dir: Path,
    reporting_config: Mapping[str, Any],
) -> dict[str, Any]:
    rows = collect_results(full_dir)
    if not rows:
        raise RuntimeError(f"no result.json files found under {full_dir}")
    panel_rows = panel_macro_rows(rows)
    noise_rows = noise_summary_rows(panel_rows)
    comparisons = paired_comparisons(
        panel_rows,
        int(reporting_config["bootstrap_repetitions"]),
        int(reporting_config["bootstrap_seed"]),
    )
    all_comparisons = all_pairwise_comparisons(
        panel_rows,
        int(reporting_config["bootstrap_repetitions"]),
        int(reporting_config["bootstrap_seed"]),
        float(reporting_config.get("noninferiority_margin_relative", 0.05)),
    )
    amortized = amortized_timing_rows(
        rows, full_dir, reporting_config["amortization_k"]
    )
    deployments = [
        read_json(path)
        for path in sorted((full_dir / "deployment_initialization").glob("worker_*.json"))
    ]
    panel_cache_rows = [
        read_json(path)
        for path in sorted(full_dir.glob("panel_*/panel_cache.json"))
    ]
    timing_breakdown = []
    for method in sorted({str(row["method_key"]) for row in rows}):
        method_rows = [row for row in rows if str(row["method_key"]) == method]
        warm = _finite(row.get("warm_online_seconds", np.nan) for row in method_rows)
        timing_breakdown.append(
            {
                "method_key": method,
                "request_count": len(method_rows),
                "warm_online_mean_seconds": float(np.mean(warm)) if warm.size else np.nan,
                "warm_online_median_seconds": float(np.median(warm)) if warm.size else np.nan,
                "deployment_worker_count": len(deployments) if method == "frozen" else 0,
                "deployment_mean_seconds": (
                    float(np.mean(_finite(
                        row.get("total_deployment_initialization_seconds", np.nan)
                        for row in deployments
                    ))) if method == "frozen" and deployments else 0.0
                ),
                "panel_initialization_mean_seconds": (
                    float(np.mean(_finite(
                        row.get("history_branch_cache_seconds", np.nan)
                        for row in panel_cache_rows
                    ))) if method == "frozen" and panel_cache_rows else 0.0
                ),
            }
        )
    write_csv(full_dir / "per_case_results.csv", rows)
    write_csv(full_dir / "per_panel_results.csv", panel_rows)
    write_csv(full_dir / "per_noise_summary.csv", noise_rows)
    write_csv(full_dir / "comparison.csv", comparisons)
    write_csv(full_dir / "all_pairwise_comparison.csv", all_comparisons)
    write_csv(full_dir / "amortized_timing.csv", amortized)
    write_csv(full_dir / "timing_breakdown.csv", timing_breakdown)
    summary = {
        "result_count": len(rows),
        "method_counts": {
            method: sum(str(row["method_key"]) == method for row in rows)
            for method in sorted({str(row["method_key"]) for row in rows})
        },
        "panel_macro_primary_unit": True,
        "noise_summary": noise_rows,
        "paired_comparison": comparisons,
        "all_pairwise_comparison": all_comparisons,
        "timing_breakdown": timing_breakdown,
        "bootstrap_repetitions": int(reporting_config["bootstrap_repetitions"]),
    }
    write_json(full_dir / "comparison.json", summary)
    from common.reporting import generate_figures

    generate_figures(full_dir)
    return summary
