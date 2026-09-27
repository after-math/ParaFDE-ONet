"""Two-history ParaFDEONet with one shared-P condition branch for the CSTR."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from .equation import CONDITION_DIM, STATE_DIM


NETWORK_FORMAT = "cstr_parafdeonet_2state_4condition_shared_p_v2"


def activation_layer(name: str) -> nn.Module:
    choices = {"gelu": nn.GELU, "silu": nn.SiLU, "tanh": nn.Tanh}
    normalized = str(name).lower()
    if normalized not in choices:
        raise ValueError(f"unsupported activation: {name}")
    return choices[normalized]()


class ResidualBlock(nn.Module):
    def __init__(self, width: int, activation: str, scale: float) -> None:
        super().__init__()
        self.scale = float(scale)
        self.activation_1 = activation_layer(activation)
        self.linear_1 = nn.Linear(width, width)
        self.activation_2 = activation_layer(activation)
        self.linear_2 = nn.Linear(width, width)
        for layer in (self.linear_1, self.linear_2):
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = self.linear_2(
            self.activation_2(self.linear_1(self.activation_1(values)))
        )
        return values + self.scale * residual


class ResidualMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        width: int,
        depth: int,
        activation: str,
    ) -> None:
        super().__init__()
        self.input = nn.Linear(input_dim, width)
        self.input_activation = activation_layer(activation)
        self.blocks = nn.ModuleList(
            ResidualBlock(width, activation, 1.0 / math.sqrt(depth))
            for _ in range(depth)
        )
        self.output = nn.Linear(width, output_dim)
        nn.init.xavier_normal_(self.input.weight)
        nn.init.zeros_(self.input.bias)
        nn.init.xavier_normal_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        hidden = self.input_activation(self.input(values))
        for block in self.blocks:
            hidden = block(hidden)
        return self.output(hidden)


def seeded_mlp(seed: int, *arguments: Any) -> ResidualMLP:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        return ResidualMLP(*arguments)


class CSTRParaFDEONet(nn.Module):
    def __init__(
        self,
        history_sensors: int,
        horizon: float,
        latent_dim: int,
        history_width: int,
        history_depth: int,
        condition_width: int,
        condition_depth: int,
        trunk_width: int,
        trunk_depth: int,
        activation: str,
        fourier_modes: int,
        initialization_seed: int,
        condition_bounds: list[list[float]],
        history_mean: list[float],
        history_std: list[float],
        output_mean: list[float],
        output_std: list[float],
    ) -> None:
        super().__init__()
        self.history_sensors = int(history_sensors)
        self.horizon = float(horizon)
        self.latent_dim = int(latent_dim)
        self.fourier_modes = int(fourier_modes)
        self.initialization_seed = int(initialization_seed)
        self.latent_scale = 1.0 / math.sqrt(self.latent_dim)
        bounds = torch.as_tensor(condition_bounds, dtype=torch.float32)
        if bounds.shape != (CONDITION_DIM, 2):
            raise ValueError("condition bounds must have shape [4,2]")
        self.register_buffer("condition_lower", bounds[:, 0].clone())
        self.register_buffer("condition_span", bounds[:, 1] - bounds[:, 0])
        self.register_buffer("history_mean", torch.as_tensor(history_mean, dtype=torch.float32))
        self.register_buffer("history_std", torch.as_tensor(history_std, dtype=torch.float32))
        self.register_buffer("output_mean", torch.as_tensor(output_mean, dtype=torch.float32))
        self.register_buffer("output_std", torch.as_tensor(output_std, dtype=torch.float32))
        if torch.any(self.history_std <= 0.0) or torch.any(self.output_std <= 0.0):
            raise ValueError("normalization standard deviations must be positive")

        self.history_branches = nn.ModuleList(
            seeded_mlp(
                initialization_seed + 110 + component,
                self.history_sensors,
                STATE_DIM * self.latent_dim,
                history_width,
                history_depth,
                activation,
            )
            for component in range(STATE_DIM)
        )
        self.condition_branch = seeded_mlp(
            initialization_seed + 200,
            CONDITION_DIM,
            self.latent_dim,
            condition_width,
            condition_depth,
            activation,
        )
        self.time_trunk = seeded_mlp(
            initialization_seed + 300,
            1 + 2 * self.fourier_modes,
            self.latent_dim,
            trunk_width,
            trunk_depth,
            activation,
        )
        self.normalized_output_bias = nn.Parameter(torch.zeros(STATE_DIM))
        self._model_config = {
            "history_sensors": self.history_sensors,
            "horizon": self.horizon,
            "latent_dim": self.latent_dim,
            "history_width": int(history_width),
            "history_depth": int(history_depth),
            "condition_width": int(condition_width),
            "condition_depth": int(condition_depth),
            "trunk_width": int(trunk_width),
            "trunk_depth": int(trunk_depth),
            "activation": str(activation),
            "fourier_modes": self.fourier_modes,
            "initialization_seed": self.initialization_seed,
            "condition_bounds": bounds.tolist(),
            "history_mean": self.history_mean.tolist(),
            "history_std": self.history_std.tolist(),
            "output_mean": self.output_mean.tolist(),
            "output_std": self.output_std.tolist(),
            "condition_feature_mode": "shared_p",
        }

    def normalize_histories(self, histories: torch.Tensor) -> torch.Tensor:
        return (histories - self.history_mean.view(1, STATE_DIM, 1)) / self.history_std.view(
            1, STATE_DIM, 1
        )

    def normalize_conditions(self, conditions: torch.Tensor) -> torch.Tensor:
        return 2.0 * (conditions - self.condition_lower) / self.condition_span - 1.0

    def normalize_outputs(self, outputs: torch.Tensor) -> torch.Tensor:
        return (outputs - self.output_mean) / self.output_std

    def time_features(self, times: torch.Tensor) -> torch.Tensor:
        normalized = times / self.horizon
        features = [2.0 * normalized - 1.0]
        for mode in range(1, self.fourier_modes + 1):
            phase = 2.0 * math.pi * mode * normalized
            features.extend((torch.sin(phase), torch.cos(phase)))
        return torch.stack(features, dim=-1)

    def encode_histories(self, histories: torch.Tensor) -> torch.Tensor:
        """Encode two scalar histories into state-specific features ``[B,2,P]``."""
        if histories.ndim != 3 or histories.shape[1:] != (
            STATE_DIM,
            self.history_sensors,
        ):
            raise ValueError("histories must have shape [B,2,M]")
        batch = histories.shape[0]
        normalized_histories = self.normalize_histories(histories)
        encoded = [
            1.0
            + branch(normalized_histories[:, component, :]).reshape(
                batch, STATE_DIM, self.latent_dim
            )
            for component, branch in enumerate(self.history_branches)
        ]
        return encoded[0] * encoded[1]

    def encode_conditions(self, conditions: torch.Tensor) -> torch.Tensor:
        """Encode ``(k,kappa,D,Tc)`` once into one shared feature ``[B,P]``."""
        if conditions.ndim != 2 or conditions.shape[1] != CONDITION_DIM:
            raise ValueError("conditions must have shape [B,4]")
        return 1.0 + self.condition_branch(self.normalize_conditions(conditions))

    def forward(
        self,
        histories: torch.Tensor,
        conditions: torch.Tensor,
        times: torch.Tensor,
    ) -> torch.Tensor:
        if histories.ndim != 3 or histories.shape[1:] != (
            STATE_DIM,
            self.history_sensors,
        ):
            raise ValueError("histories must have shape [B,2,M]")
        batch = histories.shape[0]
        if conditions.shape != (batch, CONDITION_DIM):
            raise ValueError("conditions must have shape [B,4]")
        if times.ndim == 1:
            times = times.unsqueeze(0).expand(batch, -1)
        if times.ndim != 2 or times.shape[0] != batch:
            raise ValueError("times must have shape [Q] or [B,Q]")
        state_features = self.encode_histories(histories)
        condition_features = self.encode_conditions(conditions)
        state_features = state_features * condition_features.unsqueeze(1)
        trunk = self.time_trunk(self.time_features(times))
        normalized_output = (
            torch.einsum("bsp,bqp->bqs", state_features, trunk) * self.latent_scale
            + self.normalized_output_bias
        )
        return normalized_output * self.output_std + self.output_mean

    def model_config(self) -> dict[str, Any]:
        return dict(self._model_config)


def build_operator(
    data_config: Mapping[str, Any],
    operator_config: Mapping[str, Any],
    equation_config: Mapping[str, Any],
    normalization: Mapping[str, Any],
    device: torch.device,
    initialization_seed: int,
) -> CSTRParaFDEONet:
    if str(operator_config.get("condition_feature_mode", "shared_p")) != "shared_p":
        raise ValueError("this experiment requires a shared-P condition branch")
    return CSTRParaFDEONet(
        history_sensors=int(data_config["history_sensors"]),
        horizon=float(data_config["prediction_horizon"]),
        latent_dim=int(operator_config["latent_dim"]),
        history_width=int(operator_config["history_width"]),
        history_depth=int(operator_config["history_depth"]),
        condition_width=int(operator_config["condition_width"]),
        condition_depth=int(operator_config["condition_depth"]),
        trunk_width=int(operator_config["trunk_width"]),
        trunk_depth=int(operator_config["trunk_depth"]),
        activation=str(operator_config["activation"]),
        fourier_modes=int(operator_config["fourier_modes"]),
        initialization_seed=int(initialization_seed),
        condition_bounds=equation_config["condition_bounds"],
        history_mean=list(normalization["history_mean"]),
        history_std=list(normalization["history_std"]),
        output_mean=list(normalization["output_mean"]),
        output_std=list(normalization["output_std"]),
    ).to(device)


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(value.numel() for value in model.parameters() if value.requires_grad)


def load_operator_checkpoint(
    path: Path | str, device: torch.device
) -> tuple[CSTRParaFDEONet, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("network_format") != NETWORK_FORMAT:
        raise RuntimeError("incompatible CSTR operator checkpoint")
    model_config = dict(payload["model_config"])
    if model_config.pop("condition_feature_mode", None) != "shared_p":
        raise RuntimeError("checkpoint does not use the shared-P condition branch")
    model = CSTRParaFDEONet(**model_config).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model, payload
