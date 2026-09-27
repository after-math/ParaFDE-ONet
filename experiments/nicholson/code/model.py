"""Four forward neural-operator structures for the four-patch Nicholson DDE."""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch
from torch import nn


MODEL_TYPES = (
    "single_branch_deeponet",
    "four_branch_mionet",
    "separate_parameter_shared",
    "separate_parameter_state_specific",
)
MODEL_DISPLAY_NAMES = {
    "single_branch_deeponet": "Single-Branch DeepONet",
    "four_branch_mionet": "Four-Branch MIONet",
    "separate_parameter_shared": "Separate-Parameter MIONet",
    "separate_parameter_state_specific": "Separate Parameter (State-Specific 4p)",
}
EXPECTED_FORMAL_PARAMETER_COUNTS = {
    "single_branch_deeponet": 25_924_868,
    "four_branch_mionet": 69_635_588,
    "separate_parameter_shared": 74_621_444,
    "separate_parameter_state_specific": 75_409_412,
}
OPERATOR_NETWORK_FORMAT = "nicholson_patch_operator_4d_v1"


def activation_layer(name: str) -> nn.Module:
    """Create one stateless activation from a validated name.

    ``name`` is normally ``gelu``; the return is a new GELU, SiLU or Tanh module.
    For example ``activation_layer('gelu')`` returns ``nn.GELU``.  No RNG or file is
    touched.  ``ResidualMLP`` calls the function for every nonlinear location.
    """
    choices = {"gelu": nn.GELU, "silu": nn.SiLU, "tanh": nn.Tanh}
    normalized = str(name).lower()
    if normalized not in choices:
        raise ValueError(f"unsupported activation: {name}")
    return choices[normalized]()


class ResidualBlock(nn.Module):
    """Two equal-width layers with an unattenuated identity path.

    The mapping is ``x + scale*F(x)``.  ``width`` and ``residual_scale`` determine
    its parameters; input and output have the same final dimension.  A minimal
    example ``ResidualBlock(8,'gelu',0.5)(zeros(2,8))`` returns shape ``[2,8]``.
    ``ResidualMLP`` is the only caller.
    """

    def __init__(self, width: int, activation: str, residual_scale: float) -> None:
        """Initialize the two linear layers and scaled residual path.

        ``width`` must be positive and ``residual_scale`` finite.  Construction
        returns ``None`` and consumes only the surrounding local Torch RNG.  For
        example width 640 creates two ``[640,640]`` weights.  ``ResidualMLP`` calls
        this once per configured depth.
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
        """Apply the residual mapping without modifying ``values``.

        ``values`` may have any leading dimensions and final dimension ``width``;
        output shape is identical, e.g. ``[48,640]``.  A normal autograd graph is
        produced.  ``ResidualMLP.forward`` calls this method in sequence.
        """
        residual = self.linear_2(
            self.activation_2(self.linear_1(self.activation_1(values)))
        )
        return values + self.residual_scale * residual


class ResidualMLP(nn.Module):
    """Input projection, residual stack and output projection.

    The module maps an arbitrary leading shape from ``input_dim`` to ``output_dim``.
    For example ``ResidualMLP(103,768,640,5,'gelu')`` maps ``[B,103]`` to
    ``[B,768]``.  Branches and the common time trunk are all instances of this class.
    """

    def __init__(
        self, input_dim: int, output_dim: int, width: int, depth: int, activation: str
    ) -> None:
        """Create and Xavier-initialize the residual MLP.

        Inputs specify dimensions, block count and activation.  All dimensions must
        be positive.  The constructor returns ``None`` and consumes only the local
        seeded RNG established by ``seeded_mlp``.  A width-8 example has input and
        output projections plus ``depth`` residual blocks.
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

        For example ``[16,103]`` becomes ``[16,768]``.  The return retains all
        leading dimensions and standard parameter gradients. ``Operator4D`` calls
        this for its joint/history/parameter/trunk features.
        """
        hidden = self.input_activation(self.input(values))
        for block in self.blocks:
            hidden = block(hidden)
        return self.output(hidden)


def seeded_mlp(seed: int, *arguments: Any) -> ResidualMLP:
    """Build one reproducibly initialized MLP without advancing global RNG.

    ``seed`` and the five ``ResidualMLP`` arguments define the return.  Identical
    calls produce identical weights, e.g. common trunks match across methods for one
    training seed.  Initialization is isolated with ``fork_rng`` and has no external
    RNG side effect. ``Operator4D.__init__`` calls it for every subnetwork.
    """
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        return ResidualMLP(*arguments)


class Operator4D(nn.Module):
    """Implement the four requested history/parameter factorization structures.

    Common input is histories ``[B,4,M]``, physical ``(beta,tau0)`` values ``[B,2]``
    and query times ``[Q]`` or ``[B,Q]``; output is ``[B,Q,4]``. Single Branch
    concatenates everything; Four Branch appends parameters to each history;
    Separate Shared uses four history branches plus one shared-p parameter branch;
    State-Specific uses a 4p parameter branch. B=2,Q=5 returns ``[2,5,4]``.
    Training, validation, testing and physics residuals all call this class.
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
    ) -> None:
        """Construct exactly the subnetworks required by ``model_type``.

        All architecture values come unchanged from the parent HPO configuration.
        The formal joint input has dimension 406, a Four-Branch input 103, and the
        history-only input 101.  Construction returns ``None``; local seeded module
        initialization is its only side effect.  ``build_operator`` and checkpoint
        loading are the callers.
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

        if model_type == "single_branch_deeponet":
            self.joint_branch = seeded_mlp(
                initialization_seed + 100, 4 * history_sensors + 2,
                4 * latent_dim, history_width, history_depth, activation,
            )
        elif model_type == "four_branch_mionet":
            self.history_parameter_branches = nn.ModuleList(
                seeded_mlp(
                    initialization_seed + 110 + component, history_sensors + 2,
                    4 * latent_dim, history_width, history_depth, activation,
                )
                for component in range(4)
            )
        else:
            self.history_branches = nn.ModuleList(
                seeded_mlp(
                    initialization_seed + 110 + component, history_sensors,
                    4 * latent_dim, history_width, history_depth, activation,
                )
                for component in range(4)
            )
            parameter_output = (
                latent_dim if model_type == "separate_parameter_shared" else 4 * latent_dim
            )
            self.parameter_branch = seeded_mlp(
                initialization_seed + 200, 2, parameter_output,
                parameter_width, parameter_depth, activation,
            )

        self.time_trunk = seeded_mlp(
            initialization_seed + 300, 1 + 2 * fourier_modes, latent_dim,
            trunk_width, trunk_depth, activation,
        )
        self.output_bias = nn.Parameter(torch.zeros(4))
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
        }
        self._diagnostics_enabled = False
        self._last_diagnostics: dict[str, float] = {}

    def time_features(self, times: torch.Tensor) -> torch.Tensor:
        """Embed physical time with a linear coordinate and Fourier modes.

        ``times`` has shape ``[B,Q]`` and values in ``[0,horizon]``.  The output is
        ``[B,Q,1+2K]``; with K=12 the last dimension is 25.  Time gradients are
        retained for the DDE residual.  ``forward`` is the only caller.
        """
        normalized = times / self.horizon
        features = [2.0 * normalized - 1.0]
        for mode in range(1, self.fourier_modes + 1):
            phase = 2.0 * math.pi * mode * normalized
            features.extend((torch.sin(phase), torch.cos(phase)))
        return torch.stack(features, dim=-1)

    def set_numerical_diagnostics(self, enabled: bool) -> None:
        """Enable RMS diagnostics for the next forward pass.

        ``enabled=True`` clears the previous dictionary; the method returns ``None``.
        Diagnostics are detached scalars and do not alter loss or gradients.  For
        example the training loop enables them on its first and last epoch batches.
        ``training.train_operator`` is the caller.
        """
        self._diagnostics_enabled = bool(enabled)
        if enabled:
            self._last_diagnostics = {}

    def numerical_diagnostics(self) -> dict[str, float]:
        """Return a copy of the most recently collected RMS diagnostics.

        The dictionary contains fusion, trunk and operator-contribution RMS; before
        collection it is empty.  Modifying the returned dictionary has no model side
        effect.  ``training.train_operator`` reads it for logs and history.
        """
        return dict(self._last_diagnostics)

    def record_diagnostics(
        self, fusion: torch.Tensor, trunk: torch.Tensor, contribution: torch.Tensor
    ) -> None:
        """Store detached RMS values only when explicitly enabled.

        Inputs are branch fusion ``[B,4,p]``, trunk ``[B,Q,p]`` and output
        contribution ``[B,Q,4]``.  The return is ``None``; three device-to-host
        scalar reads are the only side effect.  ``forward`` calls this before
        returning a prediction.
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
        """Predict all four patch populations for all requested times.

        Histories are ``[B,4,M]``, ``(beta,tau0)`` is ``[B,2]``, and times are
        ``[Q]`` or ``[B,Q]``. The return is ``[B,Q,4]``. All paths use one trunk and
        ``1+feature`` centering for products, preventing multiplicative gradient
        collapse.  The method retains gradients to weights, parameters and time and
        is called by every train/validation/test/physics computation.
        """
        if histories.ndim != 3 or histories.shape[1:] != (4, self.history_sensors):
            raise ValueError("histories must have shape [B,4,M]")
        batch = histories.shape[0]
        if parameters.shape != (batch, 2):
            raise ValueError("parameters must have shape [B,2]")
        if times.ndim == 1:
            times = times.unsqueeze(0).expand(batch, -1)
        if times.ndim != 2 or times.shape[0] != batch:
            raise ValueError("times must have shape [Q] or [B,Q]")
        trunk = self.time_trunk(self.time_features(times))

        if self.model_type == "single_branch_deeponet":
            joint = torch.cat((histories.reshape(batch, -1), parameters), dim=-1)
            state_features = self.joint_branch(joint).reshape(batch, 4, self.latent_dim)
        elif self.model_type == "four_branch_mionet":
            encoded = [
                1.0 + branch(
                    torch.cat((histories[:, component, :], parameters), dim=-1)
                ).reshape(batch, 4, self.latent_dim)
                for component, branch in enumerate(self.history_parameter_branches)
            ]
            state_features = encoded[0]
            for values in encoded[1:]:
                state_features = state_features * values
        else:
            histories_encoded = [
                1.0 + branch(histories[:, component, :]).reshape(
                    batch, 4, self.latent_dim
                )
                for component, branch in enumerate(self.history_branches)
            ]
            state_features = histories_encoded[0]
            for values in histories_encoded[1:]:
                state_features = state_features * values
            parameter_features = self.parameter_branch(parameters)
            if self.model_type == "separate_parameter_shared":
                state_features = state_features * (1.0 + parameter_features.unsqueeze(1))
            else:
                state_features = state_features * (
                    1.0 + parameter_features.reshape(batch, 4, self.latent_dim)
                )

        raw = torch.einsum("bsp,bqp->bqs", state_features, trunk)
        contribution = raw * self.latent_scale
        self.record_diagnostics(state_features, trunk, contribution)
        return contribution + self.output_bias

    def model_config(self) -> dict[str, Any]:
        """Return a reconstruction-safe copy of all architecture arguments.

        The dictionary includes method, dimensions, activation and initialization
        seed; for example ``result['model_type']`` is the current registry key.
        It has no side effect and is saved in every checkpoint by ``training``.
        """
        return dict(self._model_config)


def build_operator(
    model_type: str,
    history_sensors: int,
    horizon: float,
    config: Mapping[str, Any],
    device: torch.device,
    initialization_seed: int,
) -> Operator4D:
    """Construct one configured operator directly on the selected device.

    Inputs are the method key, sensor count, horizon, common operator config, device
    and seed. The return is ``Operator4D``; formal input shape is ``[B,4,101]``.
    Parameter allocation on ``device`` is the only side effect.
    ``training.train_operator`` and checkpoint loading call this function.
    """
    return Operator4D(
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
    ).to(device)


def count_trainable_parameters(model: nn.Module) -> int:
    """Count all scalar parameters with ``requires_grad=True``.

    ``model`` may be any registered operator; the returned Python integer is used
    for logs and structural drift checks.  For example ``nn.Linear(3,2)`` returns 8.
    No tensor or gradient is changed.  ``training.train_operator`` is the caller.
    """
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def load_operator_checkpoint(path: Any, device: torch.device) -> tuple[Operator4D, dict[str, Any]]:
    """Rebuild and load a strictly versioned operator checkpoint.

    ``path`` is a ``best_model.pt`` and ``device`` is the target.  The return is
    ``(model,payload)``; e.g. ``model.model_type`` matches payload ``model_type``.
    Incompatible format, method or state raises before evaluation.  Model allocation
    is the only side effect.  Final test and resume verification call this helper.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("network_format") != OPERATOR_NETWORK_FORMAT:
        raise RuntimeError(f"incompatible operator checkpoint: {path}")
    model_type = str(checkpoint.get("model_type"))
    if model_type not in MODEL_TYPES:
        raise RuntimeError("checkpoint has an unknown model type")
    model = Operator4D(**dict(checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model, checkpoint


if set(MODEL_DISPLAY_NAMES) != set(MODEL_TYPES):
    raise RuntimeError("model display-name registry is incomplete")
if set(EXPECTED_FORMAL_PARAMETER_COUNTS) != set(MODEL_TYPES):
    raise RuntimeError("formal parameter-count registry is incomplete")
