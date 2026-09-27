"""Parent trainer extended with physical ``r1``/``r2`` sensitivity supervision."""

from _source_patch import execute_source, parent_code_path, patched_source

_SOURCE = parent_code_path(__file__, "training.py")

_SENSITIVITY_FUNCTION = r'''

def model_parameter_sensitivities(
    model: Operator2D,
    histories: torch.Tensor,
    parameters: torch.Tensor,
    times: torch.Tensor,
) -> torch.Tensor:
    """Differentiate predictions with respect to physical ``r1`` and ``r2``.

    Two forward-mode JVPs return ``[B,Q,2,2]``.  The final dimension contains
    ``dy/dr1`` and ``dy/dr2`` in physical units.  No parameter normalization or
    derivative-span scaling is applied.  The tensor remains differentiable with
    respect to model weights and can therefore enter the training objective.
    """
    if parameters.ndim != 2 or parameters.shape[1] != 2:
        raise ValueError("parameters must have shape [B,2]")

    def evaluate(parameter_values: torch.Tensor) -> torch.Tensor:
        return model(histories, parameter_values, times)

    r1_tangent = torch.zeros_like(parameters)
    r1_tangent[:, 0] = 1.0
    r2_tangent = torch.zeros_like(parameters)
    r2_tangent[:, 1] = 1.0
    _, r1_derivative = torch.func.jvp(
        evaluate, (parameters,), (r1_tangent,), strict=True
    )
    _, r2_derivative = torch.func.jvp(
        evaluate, (parameters,), (r2_tangent,), strict=True
    )
    return torch.stack((r1_derivative, r2_derivative), dim=-1)
'''

_TRAIN_OLD = """    rng = np.random.default_rng(training_seed + 1000)
    train = splits["train"]
    train_size = train.histories.shape[0]"""
_TRAIN_NEW = """    rng = np.random.default_rng(training_seed + 1000)
    train = splits["train"]
    if train.parameter_sensitivities is None:
        raise RuntimeError("training split is missing physical parameter sensitivities")
    expected_sensitivity_shape = (*train.solutions.shape, 2)
    if train.parameter_sensitivities.shape != expected_sensitivity_shape:
        raise RuntimeError(
            "training sensitivity shape mismatch: "
            f"{train.parameter_sensitivities.shape} != {expected_sensitivity_shape}"
        )
    train_size = train.histories.shape[0]"""

_SETTINGS_OLD = """    pool_batch = int(operator_config["physics_pool_batch"])
    residual_points = int(operator_config["residual_points"])
    validation_every = int(operator_config["validation_every"])
    steps_per_epoch = int(operator_config["steps_per_epoch"])
    evaluation_batch = int(operator_config["evaluation_batch_size"])
    running = {"total": 0.0, "data": 0.0, "initial": 0.0, "physics": 0.0, "count": 0}"""
_SETTINGS_NEW = """    pool_batch = int(operator_config["physics_pool_batch"])
    residual_points = int(operator_config["residual_points"])
    sensitivity_points = min(
        int(operator_config["sensitivity_points"]), int(train.output_times.size)
    )
    if sensitivity_points < 1:
        raise ValueError("sensitivity_points must be positive")
    validation_every = int(operator_config["validation_every"])
    steps_per_epoch = int(operator_config["steps_per_epoch"])
    evaluation_batch = int(operator_config["evaluation_batch_size"])
    running = {
        "total": 0.0,
        "data": 0.0,
        "initial": 0.0,
        "r1_sensitivity": 0.0,
        "r2_sensitivity": 0.0,
        "physics": 0.0,
        "count": 0,
    }"""

_INITIAL_OLD = """            initial_loss = torch.mean((initial_prediction - histories_batch[:, :, -1]) ** 2)
            active_physics_weight = physics_weight_at(iteration, operator_config)"""
_INITIAL_NEW = """            initial_loss = torch.mean((initial_prediction - histories_batch[:, :, -1]) ** 2)
            sensitivity_indices = np.sort(
                rng.choice(train.output_times.size, size=sensitivity_points, replace=False)
            )
            sensitivity_index_tensor = torch.as_tensor(
                sensitivity_indices, dtype=torch.long, device=device
            )
            sensitivity_times = full_times[sensitivity_index_tensor]
            sensitivity_targets = torch.as_tensor(
                train.parameter_sensitivities[indices][:, sensitivity_indices],
                dtype=torch.float32,
                device=device,
            )
            sensitivity_prediction = model_parameter_sensitivities(
                model, histories_batch, parameters_batch, sensitivity_times
            )
            r1_sensitivity_loss = torch.mean(
                (sensitivity_prediction[..., 0] - sensitivity_targets[..., 0]) ** 2
            )
            r2_sensitivity_loss = torch.mean(
                (sensitivity_prediction[..., 1] - sensitivity_targets[..., 1]) ** 2
            )
            active_physics_weight = physics_weight_at(iteration, operator_config)"""

_TOTAL_OLD = """            total_loss = (
                float(operator_config["data_weight"]) * data_loss
                + float(operator_config["initial_weight"]) * initial_loss
                + active_physics_weight * physics_loss
            )"""
_TOTAL_NEW = """            total_loss = (
                float(operator_config["data_weight"]) * data_loss
                + float(operator_config["initial_weight"]) * initial_loss
                + float(operator_config["r1_sensitivity_weight"]) * r1_sensitivity_loss
                + float(operator_config["r2_sensitivity_weight"]) * r2_sensitivity_loss
                + active_physics_weight * physics_loss
            )"""

_ACCUMULATE_OLD = """            running["data"] += float(data_loss.detach().cpu())
            running["initial"] += float(initial_loss.detach().cpu())
            running["physics"] += float(physics_loss.detach().cpu())"""
_ACCUMULATE_NEW = """            running["data"] += float(data_loss.detach().cpu())
            running["initial"] += float(initial_loss.detach().cpu())
            running["r1_sensitivity"] += float(r1_sensitivity_loss.detach().cpu())
            running["r2_sensitivity"] += float(r2_sensitivity_loss.detach().cpu())
            running["physics"] += float(physics_loss.detach().cpu())"""

_ROW_OLD = """                    "train_data_loss": running["data"] / denominator,
                    "train_initial_loss": running["initial"] / denominator,
                    "train_physics_loss": running["physics"] / denominator,"""
_ROW_NEW = """                    "train_data_loss": running["data"] / denominator,
                    "train_initial_loss": running["initial"] / denominator,
                    "train_r1_sensitivity_loss": running["r1_sensitivity"] / denominator,
                    "train_r2_sensitivity_loss": running["r2_sensitivity"] / denominator,
                    "train_physics_loss": running["physics"] / denominator,"""

_WANDB_OLD = """                        "train/data_loss": row["train_data_loss"],
                        "train/initial_loss": row["train_initial_loss"],
                        "train/physics_loss": row["train_physics_loss"],"""
_WANDB_NEW = """                        "train/data_loss": row["train_data_loss"],
                        "train/initial_loss": row["train_initial_loss"],
                        "train/r1_sensitivity_loss": row["train_r1_sensitivity_loss"],
                        "train/r2_sensitivity_loss": row["train_r2_sensitivity_loss"],
                        "train/physics_loss": row["train_physics_loss"],"""

_RESET_OLD = """                running = {"total": 0.0, "data": 0.0, "initial": 0.0, "physics": 0.0, "count": 0}"""
_RESET_NEW = """                running = {
                    "total": 0.0,
                    "data": 0.0,
                    "initial": 0.0,
                    "r1_sensitivity": 0.0,
                    "r2_sensitivity": 0.0,
                    "physics": 0.0,
                    "count": 0,
                }"""

_TEXT = patched_source(
    _SOURCE,
    [
        ("\n\n@torch.no_grad()\ndef evaluate_operator", _SENSITIVITY_FUNCTION + "\n\n@torch.no_grad()\ndef evaluate_operator"),
        (_TRAIN_OLD, _TRAIN_NEW),
        (_SETTINGS_OLD, _SETTINGS_NEW),
        (_INITIAL_OLD, _INITIAL_NEW),
        (_TOTAL_OLD, _TOTAL_NEW),
        (_ACCUMULATE_OLD, _ACCUMULATE_NEW),
        (_ROW_OLD, _ROW_NEW),
        (_WANDB_OLD, _WANDB_NEW),
        (_RESET_OLD, _RESET_NEW),
    ],
)
execute_source(_TEXT, globals(), __file__)
