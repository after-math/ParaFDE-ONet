"""Run the three-system fixed-checkpoint sensitivity-Jacobian benchmark."""

from __future__ import annotations

import argparse
import gc
import json
import logging
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import torch

from evaluation import (
    candidate_parameters,
    cosine_rows,
    evaluate_model_jacobian,
    gradient_summary_rows,
    jacobian_metric_rows,
)
from reference import ReferenceSolver, run_convergence_study, write_rows
from benchmark_reporting import make_all_figures, read_csv
from runtime import (
    checkpoint_identity,
    load_json,
    load_model_from_source,
    source_environment,
    validate_checkpoint_pair,
    write_environment,
)
from systems import (
    balanced_subset_positions,
    load_selected_system,
    save_selection,
    subset_system,
)


LOGGER = logging.getLogger("jacobian_benchmark")


def validate_inputs(config: dict[str, Any], requested: list[str]) -> None:
    """Check every external source, dataset, configuration, and checkpoint first."""
    missing: list[str] = []
    for name in requested:
        specification = config["systems"][name]
        for key in ("primary_code_dir", "baseline_code_dir"):
            code_dir = Path(specification[key])
            for filename in ("model.py", "data.py", "equation.py"):
                path = code_dir / filename
                if not path.is_file():
                    missing.append(f"{name}.{key}: {path}")
        for key in ("resolved_config", "test_dataset"):
            path = Path(specification[key])
            if not path.is_file():
                missing.append(f"{name}.{key}: {path}")
        for variant, value in specification["checkpoints"].items():
            path = Path(value)
            if not path.is_file():
                missing.append(f"{name}.checkpoints.{variant}: {path}")
    if missing:
        details = "\n".join(f"  - {item}" for item in missing)
        raise FileNotFoundError(f"benchmark input preflight failed:\n{details}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "benchmark.json",
    )
    parser.add_argument(
        "--systems",
        default="competition,sei,nicholson",
        help="Comma-separated subset of competition,sei,nicholson",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--reference-workers", type=int, default=112)
    parser.add_argument("--network-batch-size", type=int, default=64)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")
    return parser.parse_args()


def _save_reference_archive(
    path: Path, data: Any, sensitivities: np.ndarray, specification: dict[str, Any]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        histories=data.histories,
        parameters=data.parameters,
        output_times=data.output_times,
        history_times=data.history_times,
        test_indices=data.indices,
        reference_sensitivities=sensitivities.astype(np.float32),
        finite_difference_steps=np.asarray(
            [levels[-1] for levels in specification["finite_difference_levels"]],
            dtype=np.float64,
        ),
        solver_steps=np.asarray(
            [levels[-1] for levels in specification["solver_step_levels"]],
            dtype=np.float64,
        ),
    )


def _load_models(
    primary_modules: dict[str, Any],
    specification: dict[str, Any],
    device: torch.device,
) -> tuple[dict[str, torch.nn.Module], dict[str, dict[str, Any]]]:
    checkpoint_paths = {
        key: Path(value) for key, value in specification["checkpoints"].items()
    }
    improved, improved_checkpoint = primary_modules["model"].load_operator_checkpoint(
        checkpoint_paths["with_sensitivity"], device
    )
    improved.eval()
    baseline, baseline_checkpoint = load_model_from_source(
        Path(specification["baseline_code_dir"]),
        checkpoint_paths["without_sensitivity"],
        device,
    )
    validate_checkpoint_pair(
        baseline_checkpoint,
        improved_checkpoint,
        int(specification["training_seed"]),
    )
    return (
        {
            "without_sensitivity": baseline,
            "with_sensitivity": improved,
        },
        {
            "without_sensitivity": baseline_checkpoint,
            "with_sensitivity": improved_checkpoint,
        },
    )


def run_system(
    name: str,
    specification: dict[str, Any],
    common: dict[str, Any],
    output_dir: Path,
    device: torch.device,
    reference_workers: int,
    network_batch_size: int,
    smoke: bool,
) -> None:
    system_dir = output_dir / name
    system_dir.mkdir(parents=True, exist_ok=True)
    case_override = None
    if smoke:
        case_override = 6 if specification.get("balanced_strata") else 8
    data = load_selected_system(
        name,
        specification,
        int(common["selection_seed"]),
        case_override,
    )
    save_selection(system_dir / "selected_test_indices.csv", data)

    convergence_count = int(specification["convergence_case_count"])
    gradient_count = int(specification["gradient_case_count"])
    if smoke:
        convergence_count = 3 if data.strata is not None else 4
        gradient_count = 3 if data.strata is not None else 4
    convergence_positions = balanced_subset_positions(
        data, min(convergence_count, data.parameters.shape[0]), int(common["selection_seed"]) + 11
    )
    gradient_positions = balanced_subset_positions(
        data, min(gradient_count, data.parameters.shape[0]), int(common["selection_seed"]) + 23
    )
    convergence_data = subset_system(data, convergence_positions)
    gradient_data = subset_system(data, gradient_positions)

    epsilon = float(common["relative_error_epsilon"])
    with source_environment(Path(specification["primary_code_dir"])) as primary:
        models, checkpoints = _load_models(primary, specification, device)
        identities = {
            key: checkpoint_identity(checkpoint) for key, checkpoint in checkpoints.items()
        }
        (system_dir / "checkpoint_identities.json").write_text(
            json.dumps(identities, indent=2), encoding="utf-8"
        )

        solver = ReferenceSolver(
            primary["data"],
            system_dir / "reference" / "cache",
            reference_workers,
            int(specification["reference_chunk_size"]),
        )
        convergence_rows, _ = run_convergence_study(
            convergence_data, solver, specification, epsilon
        )
        convergence_threshold = float(common["convergence_warning_threshold"])
        for row in convergence_rows:
            row["warning_threshold"] = convergence_threshold
            row["passes_threshold"] = (
                float(row["aggregate_relative_change"]) <= convergence_threshold
            )
        write_rows(system_dir / "convergence.csv", convergence_rows)
        final_checks = [
            row for row in convergence_rows if bool(row["is_final_refinement"])
        ]
        failed_checks = [
            row for row in final_checks if not bool(row["passes_threshold"])
        ]
        convergence_status = {
            "threshold": convergence_threshold,
            "status": "warning" if failed_checks else "passed",
            "final_refinement_checks": final_checks,
            "failed_final_refinement_check_count": len(failed_checks),
        }
        (system_dir / "convergence_status.json").write_text(
            json.dumps(convergence_status, indent=2), encoding="utf-8"
        )
        if failed_checks:
            LOGGER.warning(
                "%s has %d final refinement checks above %.3g; inspect convergence_status.json",
                name,
                len(failed_checks),
                convergence_threshold,
            )

        reference_directions = []
        for parameter_index in range(len(data.parameter_names)):
            final_h = float(specification["finite_difference_levels"][parameter_index][-1])
            final_dt = float(specification["solver_step_levels"][parameter_index][-1])
            reference_directions.append(
                solver.sensitivity(
                    data,
                    data.parameters,
                    parameter_index,
                    final_h,
                    final_dt,
                    "formal_reference",
                )
            )
        reference = np.stack(reference_directions, axis=-1)
        _save_reference_archive(
            system_dir / "reference" / "reference_sensitivities.npz",
            data,
            reference,
            specification,
        )

        estimates: dict[str, np.ndarray] = {}
        for variant, model in models.items():
            _, estimate = evaluate_model_jacobian(
                model,
                data.histories,
                data.parameters,
                data.output_times,
                data.parameter_spans,
                device,
                network_batch_size,
            )
            estimates[variant] = estimate
        per_case_rows, summary_rows = jacobian_metric_rows(
            data,
            reference,
            estimates,
            epsilon,
            int(common["bootstrap_repeats"]),
            int(common["bootstrap_seed"]),
        )
        write_rows(system_dir / "per_case_jacobian_metrics.csv", per_case_rows)
        write_rows(system_dir / "jacobian_summary.csv", summary_rows)

        candidates = candidate_parameters(
            gradient_data.parameters,
            gradient_data.parameter_bounds,
            float(common["gradient_parameter_offset_fraction"]),
            int(common["selection_seed"]) + 37,
        )
        finest_dt = min(
            float(levels[-1]) for levels in specification["solver_step_levels"]
        )
        truth_solution = solver.solve(
            gradient_data,
            gradient_data.parameters,
            finest_dt,
            "gradient_truth",
        )
        candidate_solution = solver.solve(
            gradient_data,
            candidates,
            finest_dt,
            "gradient_candidate",
        )
        candidate_reference_directions = []
        for parameter_index in range(len(gradient_data.parameter_names)):
            final_h = float(specification["finite_difference_levels"][parameter_index][-1])
            final_dt = float(specification["solver_step_levels"][parameter_index][-1])
            candidate_reference_directions.append(
                solver.sensitivity(
                    gradient_data,
                    candidates,
                    parameter_index,
                    final_h,
                    final_dt,
                    "gradient_reference",
                )
            )
        candidate_reference = np.stack(candidate_reference_directions, axis=-1)
        candidate_predictions: dict[str, np.ndarray] = {}
        candidate_sensitivities: dict[str, np.ndarray] = {}
        for variant, model in models.items():
            prediction, sensitivity = evaluate_model_jacobian(
                model,
                gradient_data.histories,
                candidates,
                gradient_data.output_times,
                gradient_data.parameter_spans,
                device,
                network_batch_size,
            )
            candidate_predictions[variant] = prediction
            candidate_sensitivities[variant] = sensitivity
        gradient_rows = cosine_rows(
            gradient_data,
            truth_solution,
            candidate_solution,
            candidate_reference,
            candidate_predictions,
            candidate_sensitivities,
            int(common["gradient_observation_count"]),
            epsilon,
        )
        write_rows(system_dir / "gradient_direction.csv", gradient_rows)
        write_rows(
            system_dir / "gradient_direction_summary.csv",
            gradient_summary_rows(gradient_rows),
        )
        (system_dir / "COMPLETED.json").write_text(
            json.dumps(
                {
                    "system": name,
                    "case_count": int(data.parameters.shape[0]),
                    "convergence_case_count": int(convergence_data.parameters.shape[0]),
                    "gradient_case_count": int(gradient_data.parameters.shape[0]),
                    "status": "completed",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        del models
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def main() -> None:
    arguments = parse_arguments()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(processName)s | %(message)s",
    )
    config = load_json(arguments.config)
    repository_root = Path(__file__).resolve().parents[3]
    for specification in config["systems"].values():
        for key in ("primary_code_dir", "baseline_code_dir", "resolved_config", "test_dataset"):
            value = Path(specification[key]).expanduser()
            specification[key] = str(value if value.is_absolute() else repository_root / value)
        for variant, text in specification["checkpoints"].items():
            value = Path(text).expanduser()
            specification["checkpoints"][variant] = str(
                value if value.is_absolute() else repository_root / value
            )
    requested = [item.strip() for item in arguments.systems.split(",") if item.strip()]
    unknown = set(requested) - set(config["systems"])
    if unknown:
        raise ValueError(f"unknown systems: {sorted(unknown)}")
    validate_inputs(config, requested)
    output_dir = arguments.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(arguments.config, output_dir / "config.json")
    write_environment(output_dir / "environment.json")
    device = torch.device(arguments.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    for name in requested:
        system_marker = output_dir / name / "COMPLETED.json"
        if system_marker.is_file():
            marker_values = load_json(system_marker)
            if marker_values.get("status") == "completed":
                LOGGER.info(
                    "Skipping completed system: %s (remove its COMPLETED.json only if a rerun is intended)",
                    name,
                )
                continue
        LOGGER.info("Starting system: %s", name)
        run_system(
            name,
            dict(config["systems"][name]),
            config,
            output_dir,
            device,
            arguments.reference_workers,
            arguments.network_batch_size,
            arguments.smoke,
        )
        LOGGER.info("Completed system: %s", name)
    combined_outputs = {
        "all_systems_jacobian_summary.csv": "jacobian_summary.csv",
        "all_systems_gradient_direction_summary.csv": "gradient_direction_summary.csv",
        "all_systems_convergence.csv": "convergence.csv",
    }
    for combined_name, system_name in combined_outputs.items():
        combined_rows = []
        for name in requested:
            combined_rows.extend(read_csv(output_dir / name / system_name))
        write_rows(output_dir / combined_name, combined_rows)
    if not arguments.skip_figures:
        make_all_figures(output_dir, requested)
    (output_dir / "COMPLETED.json").write_text(
        json.dumps({"systems": requested, "status": "completed"}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
