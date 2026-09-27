"""Single-model ParaFDEONet for the eight-state delayed smart grid."""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch
from torch import nn

from equation import PARAMETER_DIM, STATE_DIM


MODEL_TYPE = "parafdeonet"
MODEL_DISPLAY_NAME = "ParaFDEONet"
OPERATOR_NETWORK_FORMAT = "delayed_smart_grid4node8d_parafdeonet_v1"


def activation_layer(name: str) -> nn.Module:
    choices = {"gelu": nn.GELU, "silu": nn.SiLU, "tanh": nn.Tanh}
    key = str(name).lower()
    if key not in choices:
        raise ValueError(f"unsupported activation: {name}")
    return choices[key]()


class ResidualBlock(nn.Module):
    def __init__(self, width: int, activation: str, residual_scale: float) -> None:
        super().__init__()
        if width < 1 or residual_scale <= 0.0:
            raise ValueError("invalid residual block")
        self.residual_scale = float(residual_scale)
        self.activation_1 = activation_layer(activation)
        self.linear_1 = nn.Linear(width, width)
        self.activation_2 = activation_layer(activation)
        self.linear_2 = nn.Linear(width, width)
        for layer in (self.linear_1, self.linear_2):
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = self.linear_2(self.activation_2(self.linear_1(self.activation_1(values))))
        return values + self.residual_scale * residual


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
        if min(input_dim, output_dim, width, depth) < 1:
            raise ValueError("MLP dimensions and depth must be positive")
        self.input = nn.Linear(input_dim, width)
        self.input_activation = activation_layer(activation)
        residual_scale = 1.0 / math.sqrt(depth)
        self.blocks = nn.ModuleList(
            ResidualBlock(width, activation, residual_scale) for _ in range(depth)
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
    """Initialize a subnetwork reproducibly without advancing the global RNG."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        return ResidualMLP(*arguments)


class ParaFDEONet(nn.Module):
    """Encode eight scalar histories, three parameters and query time separately."""

    def __init__(
        self,
        history_sensors: int,
        horizon: float,
        latent_dim: int,
        history_width: int,
        history_depth: int,
        parameter_width: int,
        parameter_depth: int,
        trunk_width: int,
        trunk_depth: int,
        activation: str,
        fourier_modes: int,
        initialization_seed: int,
        parameter_bounds: list[list[float]],
        state_mean: list[float],
        state_std: list[float],
    ) -> None:
        super().__init__()
        if history_sensors < 2 or horizon <= 0.0 or latent_dim < 1 or fourier_modes < 0:
            raise ValueError("invalid operator dimensions")
        bounds = torch.as_tensor(parameter_bounds, dtype=torch.float32)
        means = torch.as_tensor(state_mean, dtype=torch.float32)
        standard_deviations = torch.as_tensor(state_std, dtype=torch.float32)
        if bounds.shape != (PARAMETER_DIM, 2) or not torch.all(bounds[:, 1] > bounds[:, 0]):
            raise ValueError("parameter_bounds must have shape [3,2]")
        if means.shape != (STATE_DIM,) or standard_deviations.shape != (STATE_DIM,):
            raise ValueError("state normalization must contain eight entries")
        if not torch.all(standard_deviations > 0.0):
            raise ValueError("state standard deviations must be positive")

        self.history_sensors = int(history_sensors)
        self.horizon = float(horizon)
        self.latent_dim = int(latent_dim)
        self.fourier_modes = int(fourier_modes)
        self.initialization_seed = int(initialization_seed)
        self.latent_scale = 1.0 / math.sqrt(latent_dim)
        self.register_buffer("physical_parameter_lower", bounds[:, 0].clone())
        self.register_buffer("physical_parameter_span", bounds[:, 1] - bounds[:, 0])
        self.register_buffer("state_mean", means.clone())
        self.register_buffer("state_std", standard_deviations.clone())

        self.history_branches = nn.ModuleList(
            seeded_mlp(
                initialization_seed + 110 + component,
                history_sensors,
                STATE_DIM * latent_dim,
                history_width,
                history_depth,
                activation,
            )
            for component in range(STATE_DIM)
        )
        self.parameter_branch = seeded_mlp(
            initialization_seed + 200,
            PARAMETER_DIM,
            latent_dim,
            parameter_width,
            parameter_depth,
            activation,
        )
        self.time_trunk = seeded_mlp(
            initialization_seed + 300,
            1 + 2 * fourier_modes,
            latent_dim,
            trunk_width,
            trunk_depth,
            activation,
        )
        self.output_bias = nn.Parameter(torch.zeros(STATE_DIM))
        self.model_config = {
            "network_format": OPERATOR_NETWORK_FORMAT,
            "history_sensors": history_sensors,
            "horizon": horizon,
            "latent_dim": latent_dim,
            "history_width": history_width,
            "history_depth": history_depth,
            "parameter_width": parameter_width,
            "parameter_depth": parameter_depth,
            "trunk_width": trunk_width,
            "trunk_depth": trunk_depth,
            "activation": activation,
            "fourier_modes": fourier_modes,
            "initialization_seed": initialization_seed,
            "parameter_bounds": bounds.tolist(),
            "state_mean": means.tolist(),
            "state_std": standard_deviations.tolist(),
        }

    def normalize_parameters(self, parameters: torch.Tensor) -> torch.Tensor:
        if parameters.shape[-1] != PARAMETER_DIM:
            raise ValueError("parameters must end in dimension three")
        return 2.0 * (parameters - self.physical_parameter_lower) / self.physical_parameter_span - 1.0

    def normalize_histories(self, histories: torch.Tensor) -> torch.Tensor:
        if histories.ndim != 3 or histories.shape[1] != STATE_DIM:
            raise ValueError("histories must have shape [B,8,M]")
        return (histories - self.state_mean[None, :, None]) / self.state_std[None, :, None]

    def prepare_times(self, times: torch.Tensor, batch: int) -> torch.Tensor:
        if times.ndim == 1:
            return times.unsqueeze(0).expand(batch, -1)
        if times.ndim == 2 and times.shape[0] == batch:
            return times
        raise ValueError("times must have shape [Q] or [B,Q]")

    def time_features(self, times: torch.Tensor) -> torch.Tensor:
        normalized = times / self.horizon
        features = [2.0 * normalized - 1.0]
        for mode in range(1, self.fourier_modes + 1):
            phase = 2.0 * math.pi * mode * normalized
            features.extend((torch.sin(phase), torch.cos(phase)))
        return torch.stack(features, dim=-1)

    def encode_histories(self, histories: torch.Tensor) -> torch.Tensor:
        normalized = self.normalize_histories(histories)
        batch = histories.shape[0]
        product = torch.ones(
            batch, STATE_DIM, self.latent_dim,
            dtype=histories.dtype, device=histories.device,
        )
        for component, branch in enumerate(self.history_branches):
            feature = branch(normalized[:, component, :]).reshape(
                batch, STATE_DIM, self.latent_dim
            )
            product = product * (1.0 + feature)
        return product

    def encode_parameters(self, parameters: torch.Tensor) -> torch.Tensor:
        return 1.0 + self.parameter_branch(self.normalize_parameters(parameters))

    def encode_times(self, times: torch.Tensor, batch: int) -> torch.Tensor:
        if times.ndim == 1:
            shared = self.time_trunk(self.time_features(times.unsqueeze(0)))
            return shared.expand(batch, -1, -1)
        prepared = self.prepare_times(times, batch)
        return self.time_trunk(self.time_features(prepared))

    def decode_features(
        self,
        history_features: torch.Tensor,
        parameter_features: torch.Tensor,
        trunk_features: torch.Tensor,
    ) -> torch.Tensor:
        fused = history_features * parameter_features.unsqueeze(1)
        normalized_output = self.latent_scale * torch.einsum(
            "bsp,bqp->bqs", fused, trunk_features
        ) + self.output_bias
        return normalized_output * self.state_std[None, None, :] + self.state_mean[None, None, :]

    def forward(
        self,
        histories: torch.Tensor,
        parameters: torch.Tensor,
        times: torch.Tensor,
    ) -> torch.Tensor:
        if parameters.shape != (histories.shape[0], PARAMETER_DIM):
            raise ValueError("parameters must have shape [B,3]")
        history_features = self.encode_histories(histories)
        parameter_features = self.encode_parameters(parameters)
        trunk_features = self.encode_times(times, histories.shape[0])
        return self.decode_features(history_features, parameter_features, trunk_features)


def build_operator(
    operator_config: Mapping[str, Any],
    data_config: Mapping[str, Any],
    state_mean: list[float],
    state_std: list[float],
    initialization_seed: int,
) -> ParaFDEONet:
    if str(operator_config.get("architecture")) != "parafdeonet":
        raise ValueError("this project only supports ParaFDEONet")
    model = ParaFDEONet(
        history_sensors=int(data_config["history_sensors"]),
        horizon=float(data_config["horizon"]),
        latent_dim=int(operator_config["latent_dim"]),
        history_width=int(operator_config["history_width"]),
        history_depth=int(operator_config["history_depth"]),
        parameter_width=int(operator_config["parameter_width"]),
        parameter_depth=int(operator_config["parameter_depth"]),
        trunk_width=int(operator_config["trunk_width"]),
        trunk_depth=int(operator_config["trunk_depth"]),
        activation=str(operator_config["activation"]),
        fourier_modes=int(operator_config["fourier_modes"]),
        initialization_seed=int(initialization_seed),
        parameter_bounds=[list(row) for row in operator_config["parameter_bounds"]],
        state_mean=state_mean,
        state_std=state_std,
    )
    return model


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
