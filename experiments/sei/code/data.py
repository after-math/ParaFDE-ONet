"""Leakage-safe GRF data and two-parameter normalized sensitivity labels."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import logging
import multiprocessing as mp
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import Dataset

from equation import DelayedSEIConfig, PARAMETER_DIM, STATE_DIM, solve_batch


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatasetSplit:
    """Hold one immutable forward split and optional training sensitivities.

    Histories are ``[N,3,M]``, physical ``[b,a]`` values ``[N,2]`` and solutions
    ``[N,Q,3]``. Training additionally stores width-scaled derivatives
    ``[N,Q,3,2]``;
    validation and test store ``None``. ``load_dataset`` and ``generate_dataset``
    create these objects for training/evaluation without later mutation.
    """

    histories: np.ndarray
    parameters: np.ndarray
    solutions: np.ndarray
    history_times: np.ndarray
    output_times: np.ndarray
    normalized_parameter_sensitivities: np.ndarray | None = None


class OperatorDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Expose histories, physical parameters and trajectories through Dataset.

    A row returns history ``[3,M]``, parameters ``[2]`` and target ``[Q,3]``. For a
    formal training split length is 24576. Training currently samples NumPy arrays
    directly, but this class provides a transparent verified Dataset interface.
    """

    def __init__(self, split: DatasetSplit) -> None:
        """Validate equal row counts and convert arrays to CPU float32 tensors.

        ``split`` is one generated/loaded split. Construction returns ``None`` and
        allocates tensor views/copies; for N=8, ``len(self)==8``. External callers and
        smoke checks use it to verify the dataset contract.
        """
        if not (
            split.histories.shape[0]
            == split.parameters.shape[0]
            == split.solutions.shape[0]
        ):
            raise ValueError("dataset arrays have unequal sample counts")
        self.histories = torch.from_numpy(np.asarray(split.histories, dtype=np.float32))
        self.parameters = torch.from_numpy(np.asarray(split.parameters, dtype=np.float32))
        self.solutions = torch.from_numpy(np.asarray(split.solutions, dtype=np.float32))

    def __len__(self) -> int:
        """Return the number of paired operator samples without changing state.

        Formal training returns 24576 and a smoke example returns 16. PyTorch
        DataLoader and validation checks call this method.
        """
        return int(self.histories.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return one history, two physical parameters and one trajectory tuple.

        ``index`` lies in ``[0,len(self))``. Index zero returns shapes ``[3,M]``,
        ``[2]`` and ``[Q,3]``. Returned tensors reference stored CPU tensors and no
        state changes. PyTorch data loading calls this method.
        """
        return self.histories[index], self.parameters[index], self.solutions[index]


def latin_hypercube(count: int, dimension: int, rng: np.random.Generator) -> np.ndarray:
    """Draw a reproducible Latin-hypercube design on the unit cube.

    ``count`` and ``dimension`` are positive and ``rng`` owns all randomness. Output
    is ``[count,dimension]``; for ``(8,1)`` every one-dimensional stratum is occupied
    exactly once. RNG state advances and no file is written. Parameter sampling calls it.
    """
    if count < 1 or dimension < 1:
        raise ValueError("Latin-hypercube dimensions must be positive")
    result = np.empty((count, dimension), dtype=np.float64)
    for column in range(dimension):
        permutation = rng.permutation(count)
        result[:, column] = (permutation + rng.random(count)) / count
    return result


def sample_parameters(
    count: int, config: DelayedSEIConfig, rng: np.random.Generator
) -> np.ndarray:
    """Map a two-dimensional LHS into the physical ``(b,a)`` rectangle.

    ``count`` is the pool size; config supplies ``b in [0.9,1.3]`` and
    ``a in [0.5,2]``; ``rng`` is reproducible. The float32 return is ``[count,2]``
    and lies strictly inside both intervals. RNG state advances. ``generate_dataset``
    calls it for independent training pools and validation/test samples.
    """
    unit = latin_hypercube(count, PARAMETER_DIM, rng)
    lower = np.asarray([bounds[0] for bounds in config.parameter_bounds])
    upper = np.asarray([bounds[1] for bounds in config.parameter_bounds])
    return (lower + unit * (upper - lower)).astype(np.float32)


def sample_histories(
    count: int,
    history_times: np.ndarray,
    level_ranges: tuple[tuple[float, float], ...],
    sigmas: tuple[float, ...],
    length_scale: float,
    bounds: tuple[tuple[float, float], ...],
    total_upper: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw smooth positive SEI histories and project them into the unit simplex.

    ``count`` histories are sampled on ``history_times``. Each of S/E/I has its own
    level range, GRF amplitude and positive bounds, while all share one length scale.
    If a point has ``S+E+I>total_upper``, all three values are proportionally scaled;
    this preserves their composition and enforces the normalized SEI domain. Output
    is float32 ``[count,3,M]``; formally ``[4096,3,101]``. RNG state advances and
    ``generate_dataset`` calls it for three leakage-independent history pools.
    """
    if count < 1 or length_scale <= 0.0 or total_upper <= 0.0:
        raise ValueError("invalid history sampling configuration")
    if len(level_ranges) != STATE_DIM or len(sigmas) != STATE_DIM or len(bounds) != STATE_DIM:
        raise ValueError("history ranges, sigmas and bounds must have one row per state")
    if any(
        sigma <= 0.0 or level[0] >= level[1] or limit[0] <= 0.0 or limit[0] >= limit[1]
        for level, sigma, limit in zip(level_ranges, sigmas, bounds)
    ):
        raise ValueError("invalid state-specific history values")
    distances = history_times[:, None] - history_times[None, :]
    covariance = np.exp(-0.5 * (distances / length_scale) ** 2)
    covariance.flat[:: covariance.shape[0] + 1] += 1e-10
    cholesky = np.linalg.cholesky(covariance)
    levels = np.stack(
        [rng.uniform(low, high, size=count) for low, high in level_ranges], axis=1
    )[..., None]
    standard_normal = rng.normal(size=(count, STATE_DIM, history_times.size))
    perturbations = (standard_normal @ cholesky.T) * np.asarray(sigmas)[None, :, None]
    histories = levels + perturbations
    lower = np.asarray([limit[0] for limit in bounds])[None, :, None]
    upper = np.asarray([limit[1] for limit in bounds])[None, :, None]
    histories = np.clip(histories, lower, upper)
    total = np.sum(histories, axis=1, keepdims=True)
    histories *= np.minimum(1.0, total_upper / np.maximum(total, 1e-15))
    if np.any(histories <= 0.0) or np.any(np.sum(histories, axis=1) > total_upper + 1e-12):
        raise RuntimeError("history simplex projection failed")
    return histories.astype(np.float32)


def balanced_pair_indices(
    history_count: int,
    parameter_count: int,
    pairs_per_history: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Create balanced unique history-parameter edges for supervised training.

    Each history receives ``pairs_per_history`` distinct ``(b,a)`` values, while cyclic
    offsets balance pool usage. Returns two int64 arrays of length
    ``history_count*pairs_per_history``; formally 24576. RNG advances through two
    permutations. ``generate_dataset`` uses the indices before reference solving.
    """
    if min(history_count, parameter_count, pairs_per_history) < 1:
        raise ValueError("pair dimensions must be positive")
    if pairs_per_history > parameter_count:
        raise ValueError("pairs_per_history cannot exceed parameter_count")
    history_order = rng.permutation(history_count)
    parameter_order = rng.permutation(parameter_count)
    history_indices: list[int] = []
    parameter_indices: list[int] = []
    for rank, history_index in enumerate(history_order):
        for offset in range(pairs_per_history):
            history_indices.append(int(history_index))
            parameter_indices.append(
                int(parameter_order[(rank + offset) % parameter_count])
            )
    return np.asarray(history_indices), np.asarray(parameter_indices)


def _solve_worker(arguments: tuple[Any, ...]) -> tuple[int, np.ndarray]:
    """Solve one trajectory chunk inside a spawned CPU process.

    ``arguments`` contains start index, arrays, equation mapping and integration
    step. Output is ``(start,solutions)`` with ``solutions [chunk,Q,3]``. A formal
    chunk has at most eight cases. Worker memory is its only side effect and
    ``solve_parallel`` submits this top-level function.
    """
    start, histories, parameters, history_times, output_times, equation_values, step = arguments
    equation = DelayedSEIConfig.from_mapping(equation_values)
    return int(start), solve_batch(
        histories, parameters, history_times, output_times, equation, float(step)
    )


def solve_parallel(
    histories: np.ndarray,
    parameters: np.ndarray,
    history_times: np.ndarray,
    output_times: np.ndarray,
    equation_values: Mapping[str, Any],
    internal_step: float,
    workers: int,
    chunk_size: int,
) -> np.ndarray:
    """Solve all paired delayed-SEI trajectories with bounded multiprocessing.

    Inputs define common grids and N cases; workers and chunk size bound CPU use.
    Output is float32 ``[N,Q,3]``; formal training returns ``[24576,401,3]``.
    Spawned workers and progress logs are its side effects. Data and sensitivity
    generation call it, while invalid/nonpositive solutions stop the experiment.
    """
    count = histories.shape[0]
    if parameters.shape != (count, PARAMETER_DIM) or workers < 1 or chunk_size < 1:
        raise ValueError("invalid parallel solver inputs")
    tasks = [
        (
            start,
            histories[start : start + chunk_size],
            parameters[start : start + chunk_size],
            history_times,
            output_times,
            dict(equation_values),
            internal_step,
        )
        for start in range(0, count, chunk_size)
    ]
    result = np.empty((count, output_times.size, STATE_DIM), dtype=np.float32)
    context = mp.get_context("spawn")
    completed = 0
    with context.Pool(processes=min(workers, len(tasks))) as pool:
        for start, values in pool.imap_unordered(_solve_worker, tasks):
            result[start : start + values.shape[0]] = values
            completed += values.shape[0]
            if completed == count or completed % max(chunk_size, count // 10) < chunk_size:
                LOGGER.info("Reference trajectories completed: %d/%d", completed, count)
    if not np.isfinite(result).all() or np.any(result <= 0.0):
        raise RuntimeError("reference solver produced invalid concentrations")
    return result


def generate_normalized_parameter_sensitivities(
    histories: np.ndarray,
    parameters: np.ndarray,
    history_times: np.ndarray,
    output_times: np.ndarray,
    equation_values: Mapping[str, Any],
    internal_step: float,
    workers: int,
    chunk_size: int,
    finite_difference_steps: tuple[float, float],
) -> np.ndarray:
    """Generate width-scaled ``b`` and ``a`` sensitivity labels by differences.

    Output is float32 ``[N,Q,3,2]``. Channel zero stores ``0.4*d y/d b`` and channel
    one stores ``1.5*d y/d a``, which are derivatives with respect to unit-interval
    parameter coordinates. Each sample/parameter uses the largest symmetric step no
    greater than its configured value that stays inside its bounds. Four additional
    reference-solver passes are performed only for training. ``generate_dataset``
    calls this function and validation/test remain sensitivity-label free.
    """
    equation = DelayedSEIConfig.from_mapping(equation_values)
    physical = np.asarray(parameters, dtype=np.float64)
    if physical.shape != (histories.shape[0], PARAMETER_DIM):
        raise ValueError("invalid sensitivity finite-difference configuration")
    if len(finite_difference_steps) != PARAMETER_DIM or any(
        step <= 0.0 for step in finite_difference_steps
    ):
        raise ValueError("finite_difference_steps must contain two positive values")
    derivatives: list[np.ndarray] = []
    for column, ((lower, upper), requested_step) in enumerate(
        zip(equation.parameter_bounds, finite_difference_steps)
    ):
        distance = np.minimum(physical[:, column] - lower, upper - physical[:, column])
        effective = np.minimum(float(requested_step), distance)
        if np.any(effective <= 1e-7):
            raise RuntimeError(
                f"parameter column {column} is too close to a bound for central difference"
            )
        plus = physical.copy()
        minus = physical.copy()
        plus[:, column] += effective
        minus[:, column] -= effective
        LOGGER.info(
            "Solving +/- sensitivity trajectories for parameter %d (step %.6g)",
            column,
            requested_step,
        )
        plus_solutions = solve_parallel(
            histories, plus.astype(np.float32), history_times, output_times,
            equation_values, internal_step, workers, chunk_size,
        )
        minus_solutions = solve_parallel(
            histories, minus.astype(np.float32), history_times, output_times,
            equation_values, internal_step, workers, chunk_size,
        )
        derivatives.append(
            (upper - lower)
            * (plus_solutions.astype(np.float64) - minus_solutions.astype(np.float64))
            / (2.0 * effective)[:, None, None]
        )
    result = np.stack(derivatives, axis=-1).astype(np.float32)
    if not np.isfinite(result).all():
        raise RuntimeError("sensitivity generation produced NaN or Inf")
    return result


def dataset_signature(config: Mapping[str, Any], seed: int, smoke: bool) -> str:
    """Hash every scientific value that determines cached data arrays.

    ``config``, data ``seed`` and smoke flag form a canonical JSON payload. Output is
    a 64-character SHA256; equal inputs yield equal signatures. No state changes.
    ``generate_or_load_dataset`` uses it to reject stale caches.
    """
    payload = {
        "seed": int(seed),
        "smoke": bool(smoke),
        "data": config["data"],
        "equation": config["equation"],
        "smoke_config": config.get("smoke", {}) if smoke else None,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _split_values(config: Mapping[str, Any], smoke: bool) -> dict[str, int]:
    """Resolve formal or smoke counts without modifying the configuration.

    Output contains history/parameter/pair/validation/test counts. Formally history
    count is 4096; smoke is 8. ``generate_dataset`` calls it exactly once.
    """
    source = config["smoke"] if smoke else config["data"]
    return {name: int(source[name]) for name in (
        "history_count", "parameter_count", "pairs_per_history",
        "validation_count", "test_count",
    )}


def generate_dataset(
    config: Mapping[str, Any], seed: int, workers: int, smoke: bool
) -> dict[str, DatasetSplit]:
    """Generate strict train/validation/test splits and training sensitivities.

    Config supplies GRF, LHS, equation and grids; seed controls local NumPy draws;
    workers bounds reference solving and smoke changes only sizes. Output contains
    three independent ``DatasetSplit`` objects. Formal train uses 4096 histories,
    1024 two-parameter values and six edges per history; validation/test use entirely
    new histories and parameter samples. CPU solver processes and logs are its side effects.
    ``generate_or_load_dataset`` calls it on a cache miss.
    """
    data = config["data"]
    sizes = _split_values(config, smoke)
    equation = DelayedSEIConfig.from_mapping(config["equation"])
    rng = np.random.default_rng(int(seed))
    history_times = np.linspace(
        -equation.maximum_history, 0.0, int(data["history_sensors"]), dtype=np.float64
    )
    output_times = np.linspace(
        0.0, float(data["horizon"]), int(data["output_points"]), dtype=np.float64
    )
    history_arguments = (
        history_times,
        tuple(tuple(float(value) for value in row) for row in data["history_level_ranges"]),
        tuple(float(value) for value in data["history_sigmas"]),
        float(data["history_length_scale"]),
        tuple(tuple(float(value) for value in row) for row in data["history_bounds"]),
        float(data["history_total_upper"]),
    )
    train_history_pool = sample_histories(sizes["history_count"], *history_arguments, rng)
    train_parameter_pool = sample_parameters(sizes["parameter_count"], equation, rng)
    history_indices, parameter_indices = balanced_pair_indices(
        sizes["history_count"], sizes["parameter_count"], sizes["pairs_per_history"], rng
    )
    train_histories = train_history_pool[history_indices]
    train_parameters = train_parameter_pool[parameter_indices]

    splits: dict[str, DatasetSplit] = {}
    for name, histories, parameters in (
        ("train", train_histories, train_parameters),
        (
            "validation",
            sample_histories(sizes["validation_count"], *history_arguments, rng),
            sample_parameters(sizes["validation_count"], equation, rng),
        ),
        (
            "test",
            sample_histories(sizes["test_count"], *history_arguments, rng),
            sample_parameters(sizes["test_count"], equation, rng),
        ),
    ):
        LOGGER.info("Solving %s split with %d cases", name, histories.shape[0])
        solutions = solve_parallel(
            histories, parameters, history_times, output_times, config["equation"],
            float(data["internal_step"]), workers, int(data["solver_chunk_size"]),
        )
        sensitivities = None
        if name == "train":
            sensitivities = generate_normalized_parameter_sensitivities(
                histories, parameters, history_times, output_times,
                config["equation"], float(data["internal_step"]), workers,
                int(data["solver_chunk_size"]),
                tuple(float(value) for value in data["sensitivity_finite_difference_steps"]),
            )
        splits[name] = DatasetSplit(
            histories=histories,
            parameters=parameters,
            solutions=solutions,
            history_times=history_times.astype(np.float32),
            output_times=output_times.astype(np.float32),
            normalized_parameter_sensitivities=sensitivities,
        )
    return splits


def save_dataset(
    directory: Path, splits: Mapping[str, DatasetSplit], signature: str
) -> None:
    """Atomically persist every split array and the scientific signature.

    ``directory`` is a stage dataset folder, splits contains train/validation/test
    and signature identifies their config. The return is ``None``; it writes one
    compressed NPZ per split and ``signature.json`` through temporary files.
    ``generate_or_load_dataset`` calls it after successful generation.
    """
    directory.mkdir(parents=True, exist_ok=True)
    for name, split in splits.items():
        temporary = directory / f".{name}.npz"
        payload: dict[str, np.ndarray] = {
            "histories": split.histories,
            "parameters": split.parameters,
            "solutions": split.solutions,
            "history_times": split.history_times,
            "output_times": split.output_times,
        }
        if split.normalized_parameter_sensitivities is not None:
            payload["normalized_parameter_sensitivities"] = (
                split.normalized_parameter_sensitivities
            )
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **payload)
        temporary.replace(directory / f"{name}.npz")
    temporary_signature = directory / ".signature.json"
    temporary_signature.write_text(
        json.dumps({"signature": signature}, indent=2), encoding="utf-8"
    )
    temporary_signature.replace(directory / "signature.json")


def load_dataset(directory: Path, expected_signature: str) -> dict[str, DatasetSplit]:
    """Load and verify one cached train/validation/test dataset.

    ``directory`` must contain a matching signature and three NPZ files. Output maps
    split names to ``DatasetSplit``; formal test solution shape is ``[2048,401,3]``.
    Files are read only. Missing training sensitivity labels raise. The pipeline calls
    it when resuming or sharing one dataset across all methods.
    """
    signature_path = directory / "signature.json"
    if not signature_path.exists():
        raise FileNotFoundError(signature_path)
    actual = json.loads(signature_path.read_text(encoding="utf-8")).get("signature")
    if actual != expected_signature:
        raise RuntimeError("cached dataset signature does not match resolved config")
    result: dict[str, DatasetSplit] = {}
    for name in ("train", "validation", "test"):
        with np.load(directory / f"{name}.npz", allow_pickle=False) as values:
            sensitivity = (
                values["normalized_parameter_sensitivities"]
                if "normalized_parameter_sensitivities" in values.files
                else None
            )
            result[name] = DatasetSplit(
                histories=values["histories"],
                parameters=values["parameters"],
                solutions=values["solutions"],
                history_times=values["history_times"],
                output_times=values["output_times"],
                normalized_parameter_sensitivities=sensitivity,
            )
    if result["train"].normalized_parameter_sensitivities is None:
        raise RuntimeError("cached training split lacks sensitivity labels")
    return result


def generate_or_load_dataset(
    directory: Path,
    config: Mapping[str, Any],
    seed: int,
    workers: int,
    smoke: bool,
) -> dict[str, DatasetSplit]:
    """Reuse a signature-compatible dataset or generate and persist it once.

    Inputs identify cache directory, resolved config, data seed, CPU budget and stage.
    Output contains three verified splits. Cache hits only read files; misses solve and
    write them. ``pipeline.run_stage`` calls it before GPU jobs so every method/seed
    receives identical scientific data.
    """
    signature = dataset_signature(config, seed, smoke)
    if (directory / "signature.json").exists():
        LOGGER.info("Reusing dataset: %s", directory)
        return load_dataset(directory, signature)
    splits = generate_dataset(config, seed, workers, smoke)
    save_dataset(directory, splits, signature)
    return splits
