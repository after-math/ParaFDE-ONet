"""Four normalized neural-operator structures for the delayed SEI system."""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch
from torch import nn


STATE_DIM = 3
PARAMETER_DIM = 2
MODEL_TYPES = (
    "single_branch_deeponet",
    "three_branch_mionet",
    "separate_parameter_shared",
    "separate_parameter_state_specific",
)
MODEL_DISPLAY_NAMES = {
    "single_branch_deeponet": "JI-DeepONet",
    "three_branch_mionet": "RP-MIONet",
    "separate_parameter_shared": "PFDEONet",
    "separate_parameter_state_specific": "SSP-MIONet",
}
EXPECTED_FORMAL_PARAMETER_COUNTS = {
    "single_branch_deeponet": 25_453_571,
    "three_branch_mionet": 53_806_595,
    "separate_parameter_shared": 58_793_987,
    "separate_parameter_state_specific": 59_319_299,
}
OPERATOR_NETWORK_FORMAT = "delayed_sei3d_normalized_sensitivity_v1"


def activation_layer(name: str) -> nn.Module:
    """Create one supported activation module from a case-insensitive name.

    ``name='gelu'`` returns a new ``nn.GELU``; SiLU and Tanh are also accepted.
    Unsupported names raise ``ValueError``. The function has no side effect and
    ``ResidualMLP`` calls it for every nonlinear location.
    """
    choices = {"gelu": nn.GELU, "silu": nn.SiLU, "tanh": nn.Tanh}
    normalized = str(name).lower()
    if normalized not in choices:
        raise ValueError(f"unsupported activation: {name}")
    return choices[normalized]()


class ResidualBlock(nn.Module):
    """Apply two equal-width linear layers plus a scaled identity path.

    Input and output share the final dimension ``width``. For example a block of
    width 8 maps ``[2,8]`` to ``[2,8]``. ``ResidualMLP`` owns and calls these blocks.
    """

    def __init__(self, width: int, activation: str, residual_scale: float) -> None:
        """Initialize one residual block with Xavier weights and zero biases.

        ``width`` and ``residual_scale`` must be positive. Construction returns
        ``None`` and allocates two linear layers; width 8 creates two ``[8,8]``
        weights. ``ResidualMLP`` calls this once per configured depth.
        """
        super().__init__()
        if width < 1 or not math.isfinite(residual_scale) or residual_scale <= 0.0:
            raise ValueError("invalid residual block dimensions")
        self.residual_scale = float(residual_scale)
        self.activation_1 = activation_layer(activation)
        self.linear_1 = nn.Linear(width, width)
        self.activation_2 = activation_layer(activation)
        self.linear_2 = nn.Linear(width, width)
        for layer in (self.linear_1, self.linear_2):
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Return ``values + scale*F(values)`` with an ordinary autograd graph.

        ``values`` may have any leading dimensions and final dimension ``width``.
        Output shape is identical, e.g. ``[16,768]``. Inputs are not modified and
        ``ResidualMLP.forward`` calls blocks sequentially.
        """
        residual = self.linear_2(
            self.activation_2(self.linear_1(self.activation_1(values)))
        )
        return values + self.residual_scale * residual


class ResidualMLP(nn.Module):
    """Map an input feature vector through one projection and residual stack.

    The final dimension changes from ``input_dim`` to ``output_dim``. For example
    ``ResidualMLP(102,1536,768,11,'gelu')`` maps ``[B,102]`` to ``[B,1536]``.
    Operator branches and the time trunk use this class.
    """

    def __init__(
        self, input_dim: int, output_dim: int, width: int, depth: int, activation: str
    ) -> None:
        """Create and Xavier-initialize input, residual and output projections.

        Dimensions and depth must be positive. The constructor returns ``None`` and
        consumes the isolated RNG set by ``seeded_mlp``. A depth-two example creates
        two residual blocks. ``Operator3D`` constructs every subnetwork through it.
        """
        super().__init__()
        if min(input_dim, output_dim, width, depth) < 1:
            raise ValueError("all MLP dimensions must be positive")
        self.input = nn.Linear(input_dim, width)
        self.input_activation = activation_layer(activation)
        scale = 1.0 / math.sqrt(depth)
        self.blocks = nn.ModuleList(
            ResidualBlock(width, activation, scale) for _ in range(depth)
        )
        self.output = nn.Linear(width, output_dim)
        nn.init.xavier_normal_(self.input.weight)
        nn.init.zeros_(self.input.bias)
        nn.init.xavier_normal_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Map the final input dimension to the configured output dimension.

        ``[16,102]`` becomes ``[16,1536]`` in a formal three-branch network. Leading
        dimensions and gradients are retained. ``Operator3D.forward`` calls it.
        """
        hidden = self.input_activation(self.input(values))
        for block in self.blocks:
            hidden = block(hidden)
        return self.output(hidden)


def seeded_mlp(seed: int, *arguments: Any) -> ResidualMLP:
    """Build one reproducibly initialized MLP without advancing global Torch RNG.

    ``seed`` and the five ResidualMLP arguments determine the result. Identical calls
    yield identical weights, allowing common modules to match across methods.
    Initialization is isolated by ``fork_rng``. ``Operator3D`` calls this helper.
    """
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        return ResidualMLP(*arguments)


class Operator3D(nn.Module):
    """Implement four three-history/two-parameter factorization structures.

    Common inputs are histories ``[B,3,M]``, physical ``[b,a]`` in ``[B,2]`` and
    times ``[Q]`` or ``[B,Q]``; output is ``[B,Q,3]``. Each learned parameter is
    affinely normalized to ``[-1,1]`` inside a Branch, while physics uses physical
    values. Single Branch concatenates everything; Three Branch appends both values
    to every history; separate models use shared-p or state-specific 3p features.
    """

    def __init__(
        self,
        model_type: str,
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
        parameter_bounds: list[list[float]] | tuple[tuple[float, float], ...],
    ) -> None:
        """Construct exactly the subnetworks required by one registered method.

        Formal history input has 101 sensors, parameter dimension two and latent width
        p=512. ``parameter_bounds`` must have shape ``[2,2]``. The constructor returns
        ``None``; isolated seeded allocation is its only side effect. ``build_operator``
        and checkpoint loading call it.
        """
        super().__init__()
        if model_type not in MODEL_TYPES:
            raise ValueError(f"unsupported model_type: {model_type}")
        if history_sensors < 2 or horizon <= 0.0 or latent_dim < 1:
            raise ValueError("invalid operator dimensions")
        self.model_type = model_type
        self.history_sensors = int(history_sensors)
        self.horizon = float(horizon)
        self.latent_dim = int(latent_dim)
        self.fourier_modes = int(fourier_modes)
        self.initialization_seed = int(initialization_seed)
        self.latent_scale = 1.0 / math.sqrt(latent_dim)
        bounds = torch.as_tensor(parameter_bounds, dtype=torch.float32)
        if bounds.shape != (PARAMETER_DIM, 2) or not torch.all(bounds[:, 1] > bounds[:, 0]):
            raise ValueError("parameter_bounds must have shape [2,2]")
        self.register_buffer("physical_parameter_lower", bounds[:, 0].clone())
        self.register_buffer("physical_parameter_span", bounds[:, 1] - bounds[:, 0])

        if model_type == "single_branch_deeponet":
            self.joint_branch = seeded_mlp(
                initialization_seed + 100,
                STATE_DIM * history_sensors + PARAMETER_DIM,
                STATE_DIM * latent_dim,
                history_width, history_depth, activation,
            )
        elif model_type == "three_branch_mionet":
            self.history_parameter_branches = nn.ModuleList(
                seeded_mlp(
                    initialization_seed + 110 + component,
                    history_sensors + PARAMETER_DIM,
                    STATE_DIM * latent_dim,
                    history_width, history_depth, activation,
                )
                for component in range(STATE_DIM)
            )
        else:
            self.history_branches = nn.ModuleList(
                seeded_mlp(
                    initialization_seed + 110 + component,
                    history_sensors,
                    STATE_DIM * latent_dim,
                    history_width, history_depth, activation,
                )
                for component in range(STATE_DIM)
            )
            parameter_output = (
                latent_dim
                if model_type == "separate_parameter_shared"
                else STATE_DIM * latent_dim
            )
            self.parameter_branch = seeded_mlp(
                initialization_seed + 200, PARAMETER_DIM, parameter_output,
                parameter_width, parameter_depth, activation,
            )

        self.time_trunk = seeded_mlp(
            initialization_seed + 300, 1 + 2 * fourier_modes, latent_dim,
            trunk_width, trunk_depth, activation,
        )
        self.output_bias = nn.Parameter(torch.zeros(STATE_DIM))
        self._model_config = {
            "model_type": model_type,
            "history_sensors": int(history_sensors),
            "horizon": float(horizon),
            "latent_dim": int(latent_dim),
            "history_width": int(history_width),
            "history_depth": int(history_depth),
            "parameter_width": int(parameter_width),
            "parameter_depth": int(parameter_depth),
            "trunk_width": int(trunk_width),
            "trunk_depth": int(trunk_depth),
            "activation": str(activation),
            "fourier_modes": int(fourier_modes),
            "initialization_seed": int(initialization_seed),
            "parameter_bounds": bounds.tolist(),
        }
        self._diagnostics_enabled = False
        self._last_diagnostics: dict[str, float] = {}

    def normalize_parameters(self, parameters: torch.Tensor) -> torch.Tensor:
        """Map physical ``b`` and ``a`` independently into ``[-1,1]``.

        ``parameters`` ends in dimension two. Formally each lower bound maps to -1,
        midpoint to 0 and upper bound to 1. The affine transformation remains
        differentiable and changes no input. Only network Branches use it; the SEI
        residual receives physical ``b`` and ``a``.
        """
        if parameters.shape[-1] != PARAMETER_DIM:
            raise ValueError("parameters must end in dimension 2")
        return 2.0 * (
            parameters - self.physical_parameter_lower
        ) / self.physical_parameter_span - 1.0

    def time_features(self, times: torch.Tensor) -> torch.Tensor:
        """Embed physical time using one coordinate and configured Fourier modes.

        Times are ``[B,Q]`` in ``[0,horizon]`` and output is ``[B,Q,1+2K]``; K=12
        gives 25 features. Time gradients remain available for physics residuals.
        ``forward`` is the sole caller.
        """
        normalized = times / self.horizon
        features = [2.0 * normalized - 1.0]
        for mode in range(1, self.fourier_modes + 1):
            phase = 2.0 * math.pi * mode * normalized
            features.extend((torch.sin(phase), torch.cos(phase)))
        return torch.stack(features, dim=-1)

    def set_numerical_diagnostics(self, enabled: bool) -> None:
        """Enable or disable detached RMS diagnostics for the next forward pass.

        ``enabled=True`` clears previous values. The return is ``None`` and no model
        mathematics changes. Training enables it on representative batches.
        """
        self._diagnostics_enabled = bool(enabled)
        if enabled:
            self._last_diagnostics = {}

    def numerical_diagnostics(self) -> dict[str, float]:
        """Return a copy of the latest fusion, trunk and output RMS diagnostics.

        Before collection the dictionary is empty. Modifying the returned copy has
        no side effect. The training loop reads it for logs and saved history.
        """
        return dict(self._last_diagnostics)

    def record_diagnostics(
        self, fusion: torch.Tensor, trunk: torch.Tensor, contribution: torch.Tensor
    ) -> None:
        """Store detached RMS values only when diagnostics are enabled.

        Fusion is ``[B,3,p]``, trunk ``[B,Q,p]`` and contribution ``[B,Q,3]``.
        The return is ``None`` and three scalar device reads are the only side effect.
        ``forward`` calls it before returning.
        """
        if not self._diagnostics_enabled:
            return

        def rms(values: torch.Tensor) -> float:
            """Return detached float32 root-mean-square as a Python float."""
            return float(values.detach().float().square().mean().sqrt().cpu())

        self._last_diagnostics = {
            "fusion_feature_rms": rms(fusion),
            "trunk_feature_rms": rms(trunk),
            "operator_contribution_rms": rms(contribution),
        }

    def forward(
        self, histories: torch.Tensor, parameters: torch.Tensor, times: torch.Tensor
    ) -> torch.Tensor:
        """Predict all three gene concentrations for every requested time.

        Histories are ``[B,3,M]``, physical ``[b,a]`` is ``[B,2]`` and times are ``[Q]``
        or ``[B,Q]``. Output is ``[B,Q,3]``. Product factors use ``1+feature`` to
        avoid multiplicative gradient collapse. Gradients to weights, ``b``, ``a`` and time
        are retained. Training, evaluation, JVP sensitivity and physics call it.
        """
        if histories.ndim != 3 or histories.shape[1:] != (STATE_DIM, self.history_sensors):
            raise ValueError("histories must have shape [B,3,M]")
        batch = histories.shape[0]
        if parameters.shape != (batch, PARAMETER_DIM):
            raise ValueError("parameters must have shape [B,2]")
        if times.ndim == 1:
            times = times.unsqueeze(0).expand(batch, -1)
        if times.ndim != 2 or times.shape[0] != batch:
            raise ValueError("times must have shape [Q] or [B,Q]")
        trunk = self.time_trunk(self.time_features(times))
        normalized_parameters = self.normalize_parameters(parameters)

        if self.model_type == "single_branch_deeponet":
            joint = torch.cat(
                (histories.reshape(batch, -1), normalized_parameters), dim=-1
            )
            state_features = self.joint_branch(joint).reshape(
                batch, STATE_DIM, self.latent_dim
            )
        elif self.model_type == "three_branch_mionet":
            encoded = [
                1.0 + branch(torch.cat(
                    (histories[:, component, :], normalized_parameters), dim=-1
                )).reshape(batch, STATE_DIM, self.latent_dim)
                for component, branch in enumerate(self.history_parameter_branches)
            ]
            state_features = encoded[0]
            for values in encoded[1:]:
                state_features = state_features * values
        else:
            encoded = [
                1.0 + branch(histories[:, component, :]).reshape(
                    batch, STATE_DIM, self.latent_dim
                )
                for component, branch in enumerate(self.history_branches)
            ]
            state_features = encoded[0]
            for values in encoded[1:]:
                state_features = state_features * values
            parameter_features = self.parameter_branch(normalized_parameters)
            if self.model_type == "separate_parameter_shared":
                state_features = state_features * (1.0 + parameter_features.unsqueeze(1))
            else:
                state_features = state_features * (
                    1.0 + parameter_features.reshape(batch, STATE_DIM, self.latent_dim)
                )

        raw = torch.einsum("bsp,bqp->bqs", state_features, trunk)
        contribution = raw * self.latent_scale
        self.record_diagnostics(state_features, trunk, contribution)
        return contribution + self.output_bias

    def model_config(self) -> dict[str, Any]:
        """Return reconstruction-safe architecture arguments as a new dictionary.

        The result contains method, dimensions, bounds and initialization seed; for
        example ``result['model_type']`` is the registered key. No state changes.
        Checkpoint creation uses this to prevent incompatible reloads.
        """
        return dict(self._model_config)


def build_operator(
    model_type: str,
    history_sensors: int,
    horizon: float,
    config: Mapping[str, Any],
    device: torch.device,
    initialization_seed: int,
) -> Operator3D:
    """Construct one configured operator directly on the selected device.

    Inputs are method, sensor count, horizon, operator config, device and seed. The
    return is ``Operator3D``; formal input history shape is ``[B,3,101]``. Device
    allocation is its only side effect. Training and checkpoint reconstruction call it.
    """
    return Operator3D(
        model_type=model_type,
        history_sensors=history_sensors,
        horizon=horizon,
        latent_dim=int(config["latent_dim"]),
        history_width=int(config["history_width"]),
        history_depth=int(config["history_depth"]),
        parameter_width=int(config["parameter_width"]),
        parameter_depth=int(config["parameter_depth"]),
        trunk_width=int(config["trunk_width"]),
        trunk_depth=int(config["trunk_depth"]),
        activation=str(config["activation"]),
        fourier_modes=int(config["fourier_modes"]),
        initialization_seed=int(initialization_seed),
        parameter_bounds=config["parameter_bounds"],
    ).to(device)


def count_trainable_parameters(model: nn.Module) -> int:
    """Count scalar parameters whose ``requires_grad`` flag is true.

    ``model`` may be any registered operator. The integer return is used in logs and
    drift checks; ``nn.Linear(3,2)`` would return 8. No tensors change. Training calls it.
    """
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def load_operator_checkpoint(path: Any, device: torch.device) -> tuple[Operator3D, dict[str, Any]]:
    """Rebuild and strictly load one versioned delayed-SEI checkpoint.

    ``path`` is ``best_model.pt`` and ``device`` is the target. The return is
    ``(model,payload)``. Wrong network format, method or state raises before use.
    Model allocation is its only side effect. Final test and resume verification call it.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("network_format") != OPERATOR_NETWORK_FORMAT:
        raise RuntimeError(f"incompatible operator checkpoint: {path}")
    model_type = str(checkpoint.get("model_type"))
    if model_type not in MODEL_TYPES:
        raise RuntimeError("checkpoint has an unknown model type")
    model = Operator3D(**dict(checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model, checkpoint


if set(MODEL_DISPLAY_NAMES) != set(MODEL_TYPES):
    raise RuntimeError("model display-name registry is incomplete")
if set(EXPECTED_FORMAL_PARAMETER_COUNTS) != set(MODEL_TYPES):
    raise RuntimeError("formal parameter-count registry is incomplete")
