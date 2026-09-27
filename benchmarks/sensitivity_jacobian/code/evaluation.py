"""Network JVP evaluation, paired Jacobian metrics, and gradient-direction checks."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from reference import per_case_relative
from systems import SystemData


VARIANT_DISPLAY = {
    "without_sensitivity": "Without sensitivity supervision",
    "with_sensitivity": "With sensitivity supervision",
}


def freeze_model(model: torch.nn.Module) -> None:
    """Disable weight gradients while retaining gradients with respect to inputs."""
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def evaluate_model_jacobian(
    model: torch.nn.Module,
    histories: np.ndarray,
    parameters: np.ndarray,
    output_times: np.ndarray,
    parameter_spans: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return predictions and range-scaled parameter JVPs in physical coordinates."""
    freeze_model(model)
    count = histories.shape[0]
    predictions: list[np.ndarray] = []
    sensitivities: list[np.ndarray] = []
    times = torch.as_tensor(output_times, dtype=torch.float32, device=device)
    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        history_tensor = torch.as_tensor(
            histories[start:stop], dtype=torch.float32, device=device
        )
        parameter_tensor = torch.as_tensor(
            parameters[start:stop], dtype=torch.float32, device=device
        )

        def operator(values: torch.Tensor) -> torch.Tensor:
            return model(history_tensor, values, times)

        with torch.no_grad():
            prediction = operator(parameter_tensor)
        directions: list[torch.Tensor] = []
        for parameter_index, span in enumerate(parameter_spans):
            tangent = torch.zeros_like(parameter_tensor)
            tangent[:, parameter_index] = 1.0
            _, derivative = torch.autograd.functional.jvp(
                operator,
                parameter_tensor,
                tangent,
                create_graph=False,
                strict=False,
            )
            directions.append(derivative * float(span))
        predictions.append(prediction.detach().cpu().numpy().astype(np.float64))
        sensitivities.append(
            torch.stack(directions, dim=-1).detach().cpu().numpy().astype(np.float64)
        )
    return np.concatenate(predictions), np.concatenate(sensitivities)


def _bootstrap_median_interval(
    values: np.ndarray, repeats: int, seed: int
) -> tuple[float, float]:
    """Return a case-level percentile interval conditional on fixed checkpoints."""
    if repeats < 1:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    count = values.size
    medians = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        medians[index] = np.median(values[rng.integers(0, count, size=count)])
    return float(np.quantile(medians, 0.025)), float(np.quantile(medians, 0.975))


def jacobian_metric_rows(
    data: SystemData,
    reference: np.ndarray,
    estimates: dict[str, np.ndarray],
    epsilon: float,
    bootstrap_repeats: int,
    bootstrap_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build per-case and summary rows for overall and parameter-wise errors."""
    if reference.ndim != 4 or reference.shape[-1] != len(data.parameter_names):
        raise ValueError("reference sensitivity must have shape [N,Q,S,P]")
    per_case_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    labels = ["overall", *data.parameter_names]
    references = [reference, *[reference[..., k] for k in range(reference.shape[-1])]]
    per_variant_errors: dict[tuple[str, str], np.ndarray] = {}

    for variant, estimate in estimates.items():
        if estimate.shape != reference.shape:
            raise ValueError(f"{variant} Jacobian shape differs from reference")
        estimate_views = [estimate, *[estimate[..., k] for k in range(estimate.shape[-1])]]
        for label_index, (label, estimate_view, reference_view) in enumerate(
            zip(labels, estimate_views, references)
        ):
            errors = per_case_relative(estimate_view, reference_view, epsilon)
            per_variant_errors[(variant, label)] = errors
            difference = estimate_view - reference_view
            aggregate = float(
                np.linalg.norm(difference) / max(np.linalg.norm(reference_view), epsilon)
            )
            absolute_rmse = float(np.sqrt(np.mean(np.square(difference))))
            low, high = _bootstrap_median_interval(
                errors,
                bootstrap_repeats,
                bootstrap_seed + 100 * label_index + (0 if variant == "without_sensitivity" else 1),
            )
            summary_rows.append(
                {
                    "system": data.name,
                    "variant": variant,
                    "variant_display": VARIANT_DISPLAY[variant],
                    "parameter": label,
                    "aggregate_relative_error": aggregate,
                    "absolute_scaled_rmse": absolute_rmse,
                    "median_relative_error": float(np.median(errors)),
                    "q25_relative_error": float(np.quantile(errors, 0.25)),
                    "q75_relative_error": float(np.quantile(errors, 0.75)),
                    "q95_relative_error": float(np.quantile(errors, 0.95)),
                    "bootstrap_median_ci_low": low,
                    "bootstrap_median_ci_high": high,
                    "case_count": int(errors.size),
                    "training_seed_count": 1,
                }
            )
            for position, error in enumerate(errors):
                row: dict[str, Any] = {
                    "system": data.name,
                    "variant": variant,
                    "parameter": label,
                    "selected_position": position,
                    "test_index": int(data.indices[position]),
                    "relative_error": float(error),
                }
                if data.strata is not None:
                    value = data.strata[position]
                    row["history_stratum"] = value.item() if hasattr(value, "item") else value
                per_case_rows.append(row)

    for label in labels:
        baseline = per_variant_errors[("without_sensitivity", label)]
        improved = per_variant_errors[("with_sensitivity", label)]
        fraction = float(np.mean(improved < baseline))
        baseline_summary = next(
            row
            for row in summary_rows
            if row["variant"] == "without_sensitivity" and row["parameter"] == label
        )
        improved_summary = next(
            row
            for row in summary_rows
            if row["variant"] == "with_sensitivity" and row["parameter"] == label
        )
        reduction = 1.0 - float(improved_summary["aggregate_relative_error"]) / max(
            float(baseline_summary["aggregate_relative_error"]), epsilon
        )
        for row in (baseline_summary, improved_summary):
            row["fraction_cases_improved_with_sensitivity"] = fraction
            row["aggregate_error_reduction_with_sensitivity"] = reduction
    return per_case_rows, summary_rows


def candidate_parameters(
    truth: np.ndarray,
    bounds: np.ndarray,
    fraction: float,
    seed: int,
) -> np.ndarray:
    """Construct deterministic interior candidates at a fixed range-scaled offset."""
    rng = np.random.default_rng(seed)
    span = bounds[:, 1] - bounds[:, 0]
    signs = rng.choice(np.array([-1.0, 1.0]), size=truth.shape)
    candidate = truth + fraction * span[None, :] * signs
    below = candidate < bounds[:, 0]
    above = candidate > bounds[:, 1]
    signs[below | above] *= -1.0
    candidate = truth + fraction * span[None, :] * signs
    if np.any(candidate < bounds[:, 0]) or np.any(candidate > bounds[:, 1]):
        raise RuntimeError("failed to construct an interior candidate parameter")
    return candidate


def cosine_rows(
    data: SystemData,
    truth_solution: np.ndarray,
    candidate_reference_solution: np.ndarray,
    candidate_reference_sensitivity: np.ndarray,
    candidate_network_predictions: dict[str, np.ndarray],
    candidate_network_sensitivities: dict[str, np.ndarray],
    observation_count: int,
    epsilon: float,
) -> list[dict[str, Any]]:
    """Compare Jacobian-only and end-to-end parameter-gradient directions."""
    if observation_count < 2 or observation_count > data.output_times.size - 1:
        raise ValueError("invalid observation count")
    observation_indices = np.unique(
        np.linspace(1, data.output_times.size - 1, observation_count, dtype=np.int64)
    )
    reference_residual = (
        candidate_reference_solution[:, observation_indices]
        - truth_solution[:, observation_indices]
    )
    reference_sensitivity = candidate_reference_sensitivity[:, observation_indices]
    reference_gradient = np.mean(
        reference_sensitivity * reference_residual[..., None], axis=(1, 2)
    )
    rows: list[dict[str, Any]] = []
    for variant in ("without_sensitivity", "with_sensitivity"):
        network_sensitivity = candidate_network_sensitivities[variant][:, observation_indices]
        network_residual = (
            candidate_network_predictions[variant][:, observation_indices]
            - truth_solution[:, observation_indices]
        )
        jacobian_only_gradient = np.mean(
            network_sensitivity * reference_residual[..., None], axis=(1, 2)
        )
        end_to_end_gradient = np.mean(
            network_sensitivity * network_residual[..., None], axis=(1, 2)
        )
        for position in range(data.parameters.shape[0]):
            reference_norm = np.linalg.norm(reference_gradient[position])
            jacobian_norm = np.linalg.norm(jacobian_only_gradient[position])
            end_to_end_norm = np.linalg.norm(end_to_end_gradient[position])
            jacobian_cosine = float(
                np.dot(reference_gradient[position], jacobian_only_gradient[position])
                / max(reference_norm * jacobian_norm, epsilon)
            )
            end_to_end_cosine = float(
                np.dot(reference_gradient[position], end_to_end_gradient[position])
                / max(reference_norm * end_to_end_norm, epsilon)
            )
            row: dict[str, Any] = {
                "system": data.name,
                "variant": variant,
                "selected_position": position,
                "test_index": int(data.indices[position]),
                "jacobian_only_gradient_cosine": jacobian_cosine,
                "end_to_end_gradient_cosine": end_to_end_cosine,
                "reference_gradient_norm": float(reference_norm),
                "observation_count": int(observation_indices.size),
            }
            if data.strata is not None:
                value = data.strata[position]
                row["history_stratum"] = value.item() if hasattr(value, "item") else value
            rows.append(row)
    return rows


def gradient_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize cosine distributions for each fixed checkpoint."""
    result = []
    for system in sorted({str(row["system"]) for row in rows}):
        for variant in ("without_sensitivity", "with_sensitivity"):
            selected = [
                row for row in rows if row["system"] == system and row["variant"] == variant
            ]
            for metric in (
                "jacobian_only_gradient_cosine",
                "end_to_end_gradient_cosine",
            ):
                values = np.asarray([float(row[metric]) for row in selected])
                result.append(
                    {
                        "system": system,
                        "variant": variant,
                        "metric": metric,
                        "median": float(np.median(values)),
                        "q25": float(np.quantile(values, 0.25)),
                        "q75": float(np.quantile(values, 0.75)),
                        "mean": float(np.mean(values)),
                        "fraction_positive": float(np.mean(values > 0.0)),
                        "case_count": int(values.size),
                        "training_seed_count": 1,
                    }
                )
    return result
