"""Common adapter contract for untouched legacy projects."""

from __future__ import annotations

import gc
import importlib
import json
import math
from pathlib import Path
import sys
import time
from types import ModuleType
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from common.io_utils import resolve_from, sha256_file


LEGACY_MODULE_NAMES = ("data", "equation", "model", "training", "_source_patch")


class BaseSystemAdapter:
    """Translate one legacy system into the common panel protocol."""

    config_class_name = ""

    def __init__(
        self,
        config: Mapping[str, Any],
        benchmark_root: Path,
        source_root_override: Path | None = None,
    ) -> None:
        self.config = dict(config)
        self.benchmark_root = benchmark_root.resolve()
        configured_source = resolve_from(
            self.benchmark_root, str(self.config["source_project"])
        )
        self.source_project = (
            source_root_override.expanduser().resolve()
            if source_root_override is not None
            else configured_source
        )
        if not self.source_project.is_dir():
            raise FileNotFoundError(f"source project is missing: {self.source_project}")
        self.source_code = self.source_project / "code"
        if not self.source_code.is_dir():
            raise FileNotFoundError(f"source code is missing: {self.source_code}")

        configured_training = resolve_from(
            self.benchmark_root, str(self.config["training_resolved_config"])
        )
        if source_root_override is None:
            self.training_config_path = configured_training
        else:
            relative = configured_training.relative_to(configured_source)
            self.training_config_path = self.source_project / relative
        if not self.training_config_path.is_file():
            raise FileNotFoundError(
                f"resolved training config is missing: {self.training_config_path}"
            )
        self.training_config = json.loads(
            self.training_config_path.read_text(encoding="utf-8")
        )
        self.data_config = dict(self.training_config["data"])
        self.equation_mapping = dict(self.training_config["equation"])
        self.data_module, self.equation_module, self.model_module = (
            self._import_legacy_modules()
        )
        config_class = getattr(self.equation_module, self.config_class_name)
        self.equation = config_class.from_mapping(self.equation_mapping)
        self._validate_science_config()

    @property
    def system_key(self) -> str:
        return str(self.config["system_key"])

    @property
    def state_dim(self) -> int:
        return int(self.config["state_dim"])

    @property
    def parameter_names(self) -> tuple[str, str]:
        values = tuple(str(value) for value in self.config["parameter_names"])
        if len(values) != 2:
            raise ValueError("exactly two parameter names are required")
        return values  # type: ignore[return-value]

    @property
    def parameter_bounds(self) -> np.ndarray:
        return np.asarray(self.config["parameter_bounds"], dtype=np.float64)

    @property
    def horizon(self) -> float:
        return float(self.config["horizon"])

    @property
    def history_sensors(self) -> int:
        return int(self.config["history_sensors"])

    @property
    def observation_count(self) -> int:
        return int(self.config["observation_count"])

    @property
    def grid_points(self) -> tuple[int, int]:
        values = tuple(int(value) for value in self.config["grid_points"])
        if len(values) != 2:
            raise ValueError("grid_points must have length two")
        return values  # type: ignore[return-value]

    def _import_legacy_modules(self) -> tuple[ModuleType, ModuleType, ModuleType]:
        # A benchmark process handles one system only. Removing these generic
        # names prevents accidental reuse if tests construct adapters serially.
        for name in LEGACY_MODULE_NAMES:
            sys.modules.pop(name, None)
        source_text = str(self.source_code)
        project_text = str(self.source_project)
        for value in (source_text, project_text):
            while value in sys.path:
                sys.path.remove(value)
        sys.path.insert(0, source_text)
        sys.path.insert(1, project_text)
        data_module = importlib.import_module("data")
        equation_module = importlib.import_module("equation")
        model_module = importlib.import_module("model")
        return data_module, equation_module, model_module

    def _validate_science_config(self) -> None:
        if self.parameter_bounds.shape != (2, 2):
            raise ValueError("parameter bounds must have shape [2,2]")
        equation_bounds = np.asarray(self.equation.parameter_bounds, dtype=np.float64)
        if not np.allclose(equation_bounds, self.parameter_bounds, rtol=0.0, atol=1e-12):
            raise RuntimeError("benchmark parameter bounds differ from training equation")
        if int(self.data_config["history_sensors"]) != self.history_sensors:
            raise RuntimeError("history sensor count differs from training distribution")
        if not math.isclose(
            float(self.data_config["horizon"]), self.horizon, rel_tol=0.0, abs_tol=1e-12
        ):
            raise RuntimeError("horizon differs from training distribution")

    def resolve_checkpoint(self, override: Path | None = None) -> Path:
        if override is not None:
            path = override.expanduser().resolve()
        else:
            configured_source = resolve_from(
                self.benchmark_root, str(self.config["source_project"])
            )
            configured_checkpoint = resolve_from(
                self.benchmark_root, str(self.config["default_checkpoint"])
            )
            if self.source_project == configured_source or not configured_checkpoint.is_relative_to(configured_source):
                path = configured_checkpoint
            else:
                path = self.source_project / configured_checkpoint.relative_to(
                    configured_source
                )
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint is missing: {path}")
        return path

    def history_grid(self) -> np.ndarray:
        return np.linspace(
            -float(self.equation.maximum_history),
            0.0,
            self.history_sensors,
            dtype=np.float64,
        )

    def output_times(self, output_points: int) -> np.ndarray:
        return np.linspace(0.0, self.horizon, output_points, dtype=np.float64)

    def observation_indices(
        self, output_points: int, observation_count: int | None = None
    ) -> np.ndarray:
        count = self.observation_count if observation_count is None else observation_count
        indices = np.rint(np.linspace(0, output_points - 1, count)).astype(np.int64)
        if np.unique(indices).size != count:
            raise ValueError("observation indices are not unique")
        return indices

    def sample_panel_histories(
        self, count: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, list[str]]:
        raise NotImplementedError

    def solve_batch(
        self,
        histories: np.ndarray,
        parameters: np.ndarray,
        history_grid: np.ndarray,
        output_times: np.ndarray,
        internal_step: float,
    ) -> np.ndarray:
        return self.equation_module.solve_batch(
            histories,
            parameters,
            history_grid,
            output_times,
            self.equation,
            float(internal_step),
        )

    def unit_to_physical_numpy(self, unit: np.ndarray) -> np.ndarray:
        unit = np.asarray(unit, dtype=np.float64)
        return self.parameter_bounds[:, 0] + unit * (
            self.parameter_bounds[:, 1] - self.parameter_bounds[:, 0]
        )

    def normalized_to_physical_tensor(self, normalized: torch.Tensor) -> torch.Tensor:
        bounds = torch.as_tensor(
            self.parameter_bounds, dtype=normalized.dtype, device=normalized.device
        )
        return bounds[:, 0] + 0.5 * (normalized + 1.0) * (
            bounds[:, 1] - bounds[:, 0]
        )

    def physical_to_unit_numpy(self, physical: np.ndarray) -> np.ndarray:
        return (np.asarray(physical) - self.parameter_bounds[:, 0]) / (
            self.parameter_bounds[:, 1] - self.parameter_bounds[:, 0]
        )

    def parameter_grid(self, device: torch.device, points: tuple[int, int]) -> torch.Tensor:
        first = torch.linspace(-1.0, 1.0, points[0], device=device)
        second = torch.linspace(-1.0, 1.0, points[1], device=device)
        return torch.cartesian_prod(first, second).to(torch.float32)

    def load_model(
        self, checkpoint_path: Path, device: torch.device
    ) -> tuple[Any, dict[str, Any], dict[str, Any]]:
        started = time.perf_counter()
        model, checkpoint = self.model_module.load_operator_checkpoint(
            checkpoint_path, device
        )
        load_seconds = time.perf_counter() - started
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        if str(model.model_type) != "separate_parameter_shared":
            raise RuntimeError("panel caching requires separate_parameter_shared")
        if int(model.history_sensors) != self.history_sensors:
            raise RuntimeError(
                "checkpoint history sensor count differs from panel archive: "
                f"checkpoint={model.history_sensors}, archive={self.history_sensors}"
            )
        if not math.isclose(
            float(model.horizon), self.horizon, rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise RuntimeError(
                "checkpoint horizon differs from panel archive: "
                f"checkpoint={model.horizon}, archive={self.horizon}"
            )
        config_class = getattr(self.equation_module, self.config_class_name)
        checkpoint_equation = config_class.from_mapping(
            checkpoint["resolved_config"]["equation"]
        )
        checkpoint_bounds = np.asarray(
            checkpoint_equation.parameter_bounds, dtype=np.float64
        )
        if not np.allclose(
            checkpoint_bounds, self.parameter_bounds, rtol=0.0, atol=1e-12
        ):
            raise RuntimeError("checkpoint parameter bounds differ from panel archive")
        if not math.isclose(
            float(checkpoint_equation.maximum_history),
            float(self.equation.maximum_history),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise RuntimeError("checkpoint history horizon differs from panel archive")
        if hasattr(model, "physical_parameter_lower"):
            lower = model.physical_parameter_lower.detach().cpu().numpy()
            span = model.physical_parameter_span.detach().cpu().numpy()
            model_bounds = np.column_stack((lower, lower + span))
            if not np.allclose(
                model_bounds, self.parameter_bounds, rtol=0.0, atol=1e-7
            ):
                raise RuntimeError(
                    "checkpoint model normalization bounds differ from panel archive"
                )
        identity = {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "size_bytes": checkpoint_path.stat().st_size,
            "model_type": str(model.model_type),
            "training_seed": checkpoint.get("training_seed"),
            "best_iteration": checkpoint.get("best_iteration"),
            "best_validation": checkpoint.get("best_validation"),
            "load_seconds": float(load_seconds),
        }
        # The model owns the weights; retaining another GPU state dict is wasteful.
        lightweight = dict(checkpoint)
        lightweight.pop("model_state_dict", None)
        gc.collect()
        return model, lightweight, identity

    def encode_history(self, model: Any, history: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            encoded = [
                1.0
                + branch(history[:, component, :]).reshape(
                    history.shape[0], self.state_dim, model.latent_dim
                )
                for component, branch in enumerate(model.history_branches)
            ]
            result = encoded[0]
            for value in encoded[1:]:
                result = result * value
        return result.detach()

    def encode_trunk(self, model: Any, times: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = model.time_features(times.reshape(1, -1))
            result = model.time_trunk(features)[0]
        return result.detach()

    def parameter_branch_input(self, model: Any, physical: torch.Tensor) -> torch.Tensor:
        if str(self.config["parameter_branch_input"]) == "normalized":
            return model.normalize_parameters(physical)
        return physical

    def encode_parameter(
        self, model: Any, normalized: torch.Tensor, detach: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        physical = self.normalized_to_physical_tensor(normalized)
        features = 1.0 + model.parameter_branch(
            self.parameter_branch_input(model, physical)
        )
        return (features.detach() if detach else features), physical

    def predict_from_features(
        self,
        model: Any,
        history_features: torch.Tensor,
        parameter_features: torch.Tensor,
        trunk_features: torch.Tensor,
    ) -> torch.Tensor:
        fusion = history_features.unsqueeze(0) * parameter_features[:, None, None, :]
        return (
            torch.einsum("chsl,ql->chqs", fusion, trunk_features)
            * float(model.latent_scale)
            + model.output_bias.reshape(1, 1, 1, self.state_dim)
        )

    def full_model_prediction(
        self,
        model: Any,
        histories: torch.Tensor,
        normalized: torch.Tensor,
        times: torch.Tensor,
    ) -> torch.Tensor:
        physical = self.normalized_to_physical_tensor(normalized)
        candidate_count = normalized.shape[0]
        history_count = histories.shape[0]
        repeated_histories = histories.unsqueeze(0).expand(
            candidate_count, -1, -1, -1
        ).reshape(candidate_count * history_count, self.state_dim, self.history_sensors)
        repeated_parameters = physical[:, None, :].expand(
            -1, history_count, -1
        ).reshape(candidate_count * history_count, 2)
        prediction = model(repeated_histories, repeated_parameters, times)
        return prediction.reshape(
            candidate_count, history_count, times.numel(), self.state_dim
        )


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
