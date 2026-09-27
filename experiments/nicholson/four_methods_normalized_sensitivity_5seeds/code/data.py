"""Leakage-safe data generation for the four-patch Nicholson operator."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import logging
import math
import multiprocessing as mp
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import Dataset

from equation import NicholsonConfig, STATE_DIM, solve_batch


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatasetSplit:
    """Hold one immutable forward-operator split and its common grids.

    Histories are ``[N,4,M]``, parameters ``[N,2]`` and solutions ``[N,Q,4]``.
    For example the formal validation split contains 2048 rows.  ``load_dataset``
    creates instances used by training and reporting; the object has no side effect.
    """

    histories: np.ndarray
    parameters: np.ndarray
    solutions: np.ndarray
    history_times: np.ndarray
    output_times: np.ndarray
    normalized_parameter_sensitivities: np.ndarray | None = None


class OperatorDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Expose one ``DatasetSplit`` to the common PyTorch training loop.

    Construction receives a split and stores float32 tensors.  ``dataset[i]``
    returns history ``[4,M]``, parameters ``[2]`` and target ``[Q,4]``; e.g. a
    DataLoader batch is ``[82,4,101]``, ``[82,2]``, ``[82,401,4]``. It is called
    only by ``training.train_operator`` and does not mutate source arrays.
    """

    def __init__(self, split: DatasetSplit) -> None:
        """Convert a split to CPU float32 tensors.

        ``split`` must contain equal sample counts and matching grids.  For a
        2048-case split, ``len(result)==2048``.  The constructor returns ``None``
        and allocates CPU tensors; ``training.train_operator`` is the caller.
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
        """Return the number of paired operator samples.

        The result is an integer, e.g. 24576 for the formal training set.  There is
        no side effect.  PyTorch DataLoader calls this method.
        """
        return int(self.histories.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return one paired history, parameter vector and trajectory.

        ``index`` is in ``[0,len(self))``; e.g. zero returns shapes ``[4,101]``,
        ``[2]`` and ``[401,4]``. Tensors are views of stored CPU data. DataLoader
        calls this method while constructing supervised batches.
        """
        return self.histories[index], self.parameters[index], self.solutions[index]


def latin_hypercube(count: int, dimension: int, rng: np.random.Generator) -> np.ndarray:
    """Draw a reproducible Latin-hypercube design on the unit cube.

    ``count`` and ``dimension`` are positive and ``rng`` is an independent NumPy
    generator.  The return is ``[count,dimension]``; for example ``(8,2)`` has one
    point in every one-dimensional stratum.  The generator state advances.  Data
    generation uses it for growth-rate pools and independent validation/test rows.
    """
    if count < 1 or dimension < 1:
        raise ValueError("Latin-hypercube dimensions must be positive")
    result = np.empty((count, dimension), dtype=np.float64)
    for column in range(dimension):
        permutation = rng.permutation(count)
        result[:, column] = (permutation + rng.random(count)) / count
    return result


def sample_parameters(
    count: int, config: NicholsonConfig, rng: np.random.Generator
) -> np.ndarray:
    """Map a Latin-hypercube design to the physical ``(beta,tau0)`` box.

    ``count`` is the requested pool size, ``config`` supplies bounds and ``rng``
    supplies reproducibility.  The float32 return is ``[count,2]``; for example all
    formal rows lie in ``[3,7] x [0.8,1.4]``. The RNG advances. generate_dataset is
    the caller.
    """
    unit = latin_hypercube(count, 2, rng)
    bounds = np.asarray(config.parameter_bounds, dtype=np.float64)
    return (bounds[:, 0] + unit * (bounds[:, 1] - bounds[:, 0])).astype(np.float32)


def sample_histories(
    count: int,
    history_times: np.ndarray,
    means: tuple[float, float, float, float],
    sigma: float,
    length_scale: float,
    bounds: tuple[float, float],
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw smooth positive two-component Gaussian-process histories.

    ``count`` histories are sampled on ``history_times`` using a squared-exponential
    covariance, component means, standard deviation and length scale, then clipped
    to bounds. The float32 result is ``[count,4,M]``; e.g. formal data returns
    ``[4096,4,101]`` within ``[0.2,2.4]``. RNG state advances. generate_dataset
    calls this once for each leakage-independent split.
    """
    if count < 1 or sigma <= 0.0 or length_scale <= 0.0 or bounds[0] >= bounds[1]:
        raise ValueError("invalid history sampling values")
    distances = history_times[:, None] - history_times[None, :]
    covariance = sigma**2 * np.exp(-0.5 * (distances / length_scale) ** 2)
    covariance.flat[:: covariance.shape[0] + 1] += 1e-10
    cholesky = np.linalg.cholesky(covariance)
    if len(means) != STATE_DIM:
        raise ValueError("history means must contain four patch values")
    histories = np.empty((count, STATE_DIM, history_times.size), dtype=np.float64)
    for component, mean in enumerate(means):
        standard_normal = rng.normal(size=(count, history_times.size))
        histories[:, component, :] = mean + standard_normal @ cholesky.T
    return np.clip(histories, bounds[0], bounds[1]).astype(np.float32)


def balanced_pair_indices(
    history_count: int,
    parameter_count: int,
    pairs_per_history: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Create connected, balanced history--parameter training pairs.

    Every history receives exactly ``pairs_per_history`` different parameters while
    cyclic offsets distribute parameter use nearly uniformly.  Returns two int64
    arrays of length ``history_count*pairs_per_history``; e.g. formal length is
    24576.  RNG state advances through permutations.  ``generate_dataset`` uses the
    indices before solving reference trajectories.
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
            parameter_indices.append(int(parameter_order[(rank + offset) % parameter_count]))
    return np.asarray(history_indices), np.asarray(parameter_indices)


def _solve_worker(arguments: tuple[Any, ...]) -> tuple[int, np.ndarray]:
    """Solve one trajectory chunk in a spawned CPU worker.

    ``arguments`` contains chunk start, arrays, equation mapping and step.  The
    return is ``(start,solutions)`` with solutions ``[chunk,Q,4]``; e.g. a formal
    chunk has at most 64 cases.  It allocates worker-local solver memory.  Only
    ``solve_parallel`` submits this top-level spawn-safe helper.
    """
    start, histories, parameters, history_times, output_times, equation_values, step = arguments
    equation = NicholsonConfig.from_mapping(equation_values)
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
    """Solve all reference trajectories with bounded spawn multiprocessing.

    Inputs describe paired cases and common grids; ``workers`` is capped by chunks.
    The float32 return is ``[N,Q,4]``; for example formal training produces
    ``[24576,401,4]``. It starts and joins CPU workers and logs coarse progress.
    ``generate_dataset`` calls it independently for train, validation and test.
    """
    count = histories.shape[0]
    if parameters.shape != (count, 2) or workers < 1 or chunk_size < 1:
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
        raise RuntimeError("reference solver produced non-finite or nonpositive states")
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
    """Generate width-normalized solver sensitivities by central differences.

    The returned float32 array has shape ``[N,Q,4,2]``.  Its final index stores
    ``4*d y/d beta`` and ``0.6*d y/d tau0`` for the formal bounds.  Each row uses
    the largest symmetric perturbation no greater than the requested step that stays
    inside the physical box, so every label is centered at the original parameter.
    Four additional reference-solver passes are performed; only the training split
    calls this function.
    """
    bounds = np.asarray(equation_values["parameter_bounds"], dtype=np.float64)
    steps = np.asarray(finite_difference_steps, dtype=np.float64)
    physical_parameters = np.asarray(parameters, dtype=np.float64)
    if bounds.shape != (2, 2) or steps.shape != (2,) or np.any(steps <= 0.0):
        raise ValueError("invalid sensitivity finite-difference configuration")
    if np.any(physical_parameters < bounds[:, 0]) or np.any(
        physical_parameters > bounds[:, 1]
    ):
        raise ValueError("parameter row lies outside the physical bounds")
    spans = bounds[:, 1] - bounds[:, 0]
    sensitivities = np.empty(
        (histories.shape[0], output_times.size, STATE_DIM, 2), dtype=np.float32
    )
    for parameter_index, parameter_name in enumerate(("beta", "tau0")):
        distance_to_bounds = np.minimum(
            physical_parameters[:, parameter_index] - bounds[parameter_index, 0],
            bounds[parameter_index, 1] - physical_parameters[:, parameter_index],
        )
        effective_steps = np.minimum(steps[parameter_index], distance_to_bounds)
        if np.any(effective_steps <= 1e-7):
            raise RuntimeError(
                f"a {parameter_name} sample is too close to its bound for a central difference"
            )
        plus_parameters = physical_parameters.copy()
        minus_parameters = physical_parameters.copy()
        plus_parameters[:, parameter_index] += effective_steps
        minus_parameters[:, parameter_index] -= effective_steps
        LOGGER.info(
            "Solving +/-%s sensitivity trajectories (requested step %.6g)",
            parameter_name,
            steps[parameter_index],
        )
        plus_solutions = solve_parallel(
            histories,
            plus_parameters.astype(np.float32),
            history_times,
            output_times,
            equation_values,
            internal_step,
            workers,
            chunk_size,
        )
        minus_solutions = solve_parallel(
            histories,
            minus_parameters.astype(np.float32),
            history_times,
            output_times,
            equation_values,
            internal_step,
            workers,
            chunk_size,
        )
        denominator = (2.0 * effective_steps)[:, None, None]
        sensitivities[..., parameter_index] = (
            spans[parameter_index]
            * (plus_solutions.astype(np.float64) - minus_solutions.astype(np.float64))
            / denominator
        ).astype(np.float32)
        del plus_solutions, minus_solutions
    if not np.isfinite(sensitivities).all():
        raise RuntimeError("reference sensitivity generation produced non-finite values")
    return sensitivities


def dataset_signature(config: Mapping[str, Any], seed: int, smoke: bool) -> str:
    """Hash every scientific input that determines cached arrays.

    ``config`` is resolved JSON, ``seed`` the data seed and ``smoke`` selects sizes.
    The return is a 64-character SHA256 string; equal scientific inputs give equal
    signatures.  No state changes.  ``generate_or_load_dataset`` prevents accidental
    reuse of incompatible caches with it.
    """
    payload = {
        "seed": int(seed),
        "smoke": bool(smoke),
        "data": config["data"],
        "equation": config["equation"],
        "smoke_config": config.get("smoke", {}) if smoke else None,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _split_values(config: Mapping[str, Any], smoke: bool) -> dict[str, int]:
    """Resolve formal or smoke sample sizes without changing the config.

    ``config`` contains ``data`` and optional ``smoke`` mappings.  The return has
    history, parameter, pair, validation and test counts; e.g. formal history count
    is 4096.  ``generate_dataset`` is the only caller.
    """
    data = config["data"]
    if smoke:
        small = config["smoke"]
        return {
            "history_count": int(small["history_count"]),
            "parameter_count": int(small["parameter_count"]),
            "pairs_per_history": int(small["pairs_per_history"]),
            "validation_count": int(small["validation_count"]),
            "test_count": int(small["test_count"]),
        }
    return {name: int(data[name]) for name in (
        "history_count", "parameter_count", "pairs_per_history",
        "validation_count", "test_count",
    )}


def generate_dataset(
    config: Mapping[str, Any], seed: int, workers: int, smoke: bool
) -> dict[str, DatasetSplit]:
    """Generate fixed, leakage-independent train/validation/test operator data.

    ``config`` supplies the exact equation and sampling protocol, ``seed`` controls
    all NumPy draws, ``workers`` bounds reference solves and ``smoke`` only changes
    counts.  Returns the three ``DatasetSplit`` objects.  Train uses a 4096-by-1024
    pool with six pairs per history; validation/test draw wholly new histories and
    parameters, preventing group leakage.  CPU workers are its only side effect.
    ``generate_or_load_dataset`` calls it on a cache miss.
    """
    data = config["data"]
    sizes = _split_values(config, smoke)
    equation = NicholsonConfig.from_mapping(config["equation"])
    rng = np.random.default_rng(int(seed))
    history_times = np.linspace(
        -equation.maximum_history, 0.0, int(data["history_sensors"]), dtype=np.float64
    )
    output_times = np.linspace(
        0.0, float(data["horizon"]), int(data["output_points"]), dtype=np.float64
    )
    means = tuple(float(item) for item in data["history_mean"])
    bounds = tuple(float(item) for item in data["history_bounds"])
    history_arguments = (
        history_times, means, float(data["history_sigma"]),
        float(data["history_length_scale"]), bounds,
    )
    train_history_pool = sample_histories(sizes["history_count"], *history_arguments, rng)
    train_parameter_pool = sample_parameters(sizes["parameter_count"], equation, rng)
    history_indices, parameter_indices = balanced_pair_indices(
        sizes["history_count"], sizes["parameter_count"], sizes["pairs_per_history"], rng
    )
    train_histories = train_history_pool[history_indices]
    train_parameters = train_parameter_pool[parameter_indices]

    splits: dict[str, DatasetSplit] = {}
    for name, count, histories, parameters in (
        ("train", train_histories.shape[0], train_histories, train_parameters),
        (
            "validation", sizes["validation_count"],
            sample_histories(sizes["validation_count"], *history_arguments, rng),
            sample_parameters(sizes["validation_count"], equation, rng),
        ),
        (
            "test", sizes["test_count"],
            sample_histories(sizes["test_count"], *history_arguments, rng),
            sample_parameters(sizes["test_count"], equation, rng),
        ),
    ):
        LOGGER.info("Solving %s split with %d cases", name, count)
        solutions = solve_parallel(
            histories, parameters, history_times, output_times, config["equation"],
            float(data["internal_step"]), workers, int(data["solver_chunk_size"]),
        )
        normalized_parameter_sensitivities = None
        if name == "train":
            sensitivity_steps = tuple(
                float(value) for value in data["sensitivity_finite_difference_steps"]
            )
            if len(sensitivity_steps) != 2:
                raise ValueError("sensitivity_finite_difference_steps must contain two values")
            normalized_parameter_sensitivities = generate_normalized_parameter_sensitivities(
                histories,
                parameters,
                history_times,
                output_times,
                config["equation"],
                float(data["internal_step"]),
                workers,
                int(data["solver_chunk_size"]),
                sensitivity_steps,
            )
        splits[name] = DatasetSplit(
            histories=histories, parameters=parameters, solutions=solutions,
            history_times=history_times.astype(np.float32),
            output_times=output_times.astype(np.float32),
            normalized_parameter_sensitivities=normalized_parameter_sensitivities,
        )
    return splits


def save_dataset(
    directory: Path, splits: Mapping[str, DatasetSplit], signature: str
) -> None:
    """Atomically persist generated arrays and their scientific signature.

    ``directory`` is the stage dataset folder, ``splits`` contains train/validation/
    test and ``signature`` identifies the config.  Returns ``None``; it creates one
    compressed NPZ per split plus ``signature.json``.  A failed write never replaces
    a valid final file.  ``generate_or_load_dataset`` calls it after generation.
    """
    directory.mkdir(parents=True, exist_ok=True)
    for name, split in splits.items():
        temporary = directory / f".{name}.npz"
        arrays = {
            "histories": split.histories,
            "parameters": split.parameters,
            "solutions": split.solutions,
            "history_times": split.history_times,
            "output_times": split.output_times,
        }
        if split.normalized_parameter_sensitivities is not None:
            arrays["normalized_parameter_sensitivities"] = (
                split.normalized_parameter_sensitivities
            )
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        temporary.replace(directory / f"{name}.npz")
    temporary_signature = directory / ".signature.json"
    temporary_signature.write_text(json.dumps({"signature": signature}, indent=2), encoding="utf-8")
    temporary_signature.replace(directory / "signature.json")


def load_dataset(directory: Path, expected_signature: str) -> dict[str, DatasetSplit]:
    """Load and verify one cached three-way dataset.

    ``directory`` must contain signature and three NPZ files matching
    ``expected_signature``.  The return maps split names to ``DatasetSplit``; e.g.
    ``result['test'].solutions`` is ``[2048,401,4]`` formally. Files are read only.
    ``generate_or_load_dataset`` calls it when resuming or sharing data across methods.
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
    return result


def generate_or_load_dataset(
    directory: Path,
    config: Mapping[str, Any],
    seed: int,
    workers: int,
    smoke: bool,
) -> dict[str, DatasetSplit]:
    """Reuse a compatible dataset or generate it exactly once.

    Inputs identify the stage, resolved config, sole data seed, CPU worker budget and
    smoke status.  Returns three verified splits.  On cache miss it writes NPZ files;
    on hit it only reads.  ``scripts.pipeline.run_stage`` calls this before launching
    the four GPU model jobs so all structures see identical cases.
    """
    signature = dataset_signature(config, seed, smoke)
    if (directory / "signature.json").exists():
        LOGGER.info("Reusing dataset: %s", directory)
        return load_dataset(directory, signature)
    splits = generate_dataset(config, seed, workers, smoke)
    save_dataset(directory, splits, signature)
    return splits

COMBINATION_SPLIT_NAMES = (
    "seen_history_seen_parameter",
    "seen_history_unseen_parameter",
    "unseen_history_seen_parameter",
    "unseen_history_unseen_parameter",
)


def _sample_parameters_in_bounds(
    count: int, bounds: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Sample ``(beta,tau0)`` vectors inside an explicit two-parameter box.

    ``count`` is positive, ``bounds`` has shape ``[2,2]`` and ``rng`` controls an
    independent Latin-hypercube design.  The float32 return has shape ``[count,2]``;
    for example bounds ``[[3.5,6.5],[0.875,1.325]]`` reserve the outer physical
    parameter region for a separate extrapolation study.  RNG state advances but no
    file is written.  ``generate_or_load_combination_dataset`` calls this for the
    training, validation and unseen-exact-parameter pools.
    """
    values = np.asarray(bounds, dtype=np.float64)
    if values.shape != (2, 2) or np.any(values[:, 0] >= values[:, 1]):
        raise ValueError("parameter bounds must have shape [2,2] with lower < upper")
    unit = latin_hypercube(count, 2, rng)
    return (values[:, 0] + unit * (values[:, 1] - values[:, 0])).astype(np.float32)


def _held_out_seen_edges(
    training_history_indices: np.ndarray,
    training_parameter_indices: np.ndarray,
    selected_histories: np.ndarray,
    parameter_count: int,
    pairs_per_history: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Choose unseen edges between histories and parameters already seen in training.

    Training edge arrays describe the observed bipartite graph; ``selected_histories``
    identifies existing history nodes and ``pairs_per_history`` requests new edges for
    each.  Returns two int64 arrays of equal length.  For example 1024 histories and
    two held-out edges produce 2048 SH--SP cases, none present in training.  RNG state
    advances and invalid graph sizes raise.  The combination dataset generator uses
    it to prevent random-pair leakage in the seen-history/seen-parameter split.
    """
    train_h = np.asarray(training_history_indices, dtype=np.int64)
    train_p = np.asarray(training_parameter_indices, dtype=np.int64)
    if train_h.ndim != 1 or train_h.shape != train_p.shape:
        raise ValueError("training edge arrays must be equal one-dimensional arrays")
    output_h: list[int] = []
    output_p: list[int] = []
    for history_index in np.asarray(selected_histories, dtype=np.int64):
        used = set(train_p[train_h == history_index].tolist())
        candidates = np.asarray(
            [index for index in range(parameter_count) if index not in used],
            dtype=np.int64,
        )
        if candidates.size < pairs_per_history:
            raise ValueError("insufficient held-out parameter edges for a seen history")
        chosen = rng.choice(candidates, size=pairs_per_history, replace=False)
        output_h.extend([int(history_index)] * pairs_per_history)
        output_p.extend(int(index) for index in chosen)
    held_h = np.asarray(output_h, dtype=np.int64)
    held_p = np.asarray(output_p, dtype=np.int64)
    training_edges = set(zip(train_h.tolist(), train_p.tolist()))
    if any(edge in training_edges for edge in zip(held_h.tolist(), held_p.tolist())):
        raise RuntimeError("held-out seen edge overlaps the training graph")
    return held_h, held_p


def combination_dataset_signature(
    config: Mapping[str, Any], seed: int, smoke: bool
) -> str:
    """Hash every scientific input defining a combination-generalization dataset.

    ``config`` supplies resolved data, equation and holdout protocol, ``seed`` is the
    fixed data seed and ``smoke`` distinguishes small/formal grids.  The return is a
    64-character SHA256 string.  For example changing central parameter bounds makes
    an old cache incompatible.  It has no side effect and is called before loading or
    generating combination arrays.
    """
    payload = {
        "seed": int(seed),
        "smoke": bool(smoke),
        "data": config["data"],
        "equation": config["equation"],
        "combination_generalization": config["combination_generalization"],
        "physics_pairing_mode": config["operator"].get("physics_pairing_mode"),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _save_combination_split(path: Path, split: DatasetSplit) -> None:
    """Atomically save one named combination split as a compressed NPZ file.

    ``path`` is the final file and ``split`` contains histories ``[N,4,M]``, physical
    parameters ``[N,2]``, solutions ``[N,Q,4]`` and grids.  Returns ``None``; for
    example ``test_unseen_history_unseen_parameter.npz`` is created through a hidden
    temporary file.  The dataset generator is the only caller and interrupted writes
    cannot masquerade as complete data.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            histories=split.histories,
            parameters=split.parameters,
            solutions=split.solutions,
            history_times=split.history_times,
            output_times=split.output_times,
        )
    temporary.replace(path)


def load_combination_dataset(
    directory: Path, expected_signature: str
) -> tuple[dict[str, DatasetSplit], dict[str, DatasetSplit], dict[str, Any]]:
    """Load and audit one cached combination-generalization dataset.

    ``directory`` must contain a matching manifest, train/validation and four named
    test NPZ files; ``expected_signature`` prevents scientific drift.  Returns the
    ordinary training mapping ``{train,validation,test}``, all four semantic test
    splits and the manifest.  Here ``test`` is the predeclared UH--UP split.  Files
    are read only.  ``generate_or_load_combination_dataset`` and GPU workers call it.
    """
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("complete") is not True
        or manifest.get("signature") != expected_signature
        or tuple(manifest.get("split_order", ())) != COMBINATION_SPLIT_NAMES
        or int(manifest.get("leakage_audit", {}).get("training_edge_overlap", -1)) != 0
        or not bool(manifest.get("leakage_audit", {}).get("physics_uses_training_edges"))
    ):
        raise RuntimeError("combination dataset manifest failed compatibility/leakage audit")

    def read_split(name: str) -> DatasetSplit:
        """Read one verified NPZ into an immutable ``DatasetSplit``.

        ``name`` is ``train``, ``validation`` or a prefixed test name.  The return
        contains five NumPy arrays and has no write side effect.  For example
        ``read_split('train')`` returns 24576 formal pairs.  Only this loader calls it.
        """
        with np.load(directory / f"{name}.npz", allow_pickle=False) as values:
            return DatasetSplit(**{key: values[key] for key in (
                "histories", "parameters", "solutions", "history_times", "output_times"
            )})

    train = read_split("train")
    validation = read_split("validation")
    named = {
        split_name: read_split(f"test_{split_name}")
        for split_name in COMBINATION_SPLIT_NAMES
    }
    primary = str(manifest["primary_split"])
    return {"train": train, "validation": validation, "test": named[primary]}, named, manifest


def generate_or_load_combination_dataset(
    directory: Path,
    config: Mapping[str, Any],
    seed: int,
    workers: int,
    smoke: bool,
) -> tuple[dict[str, DatasetSplit], dict[str, DatasetSplit], dict[str, Any]]:
    """Create or reuse a strict four-way history--parameter holdout dataset.

    ``directory`` is stage-local, ``config`` is already resolved for smoke/formal
    sizes, ``seed`` controls every NumPy draw, ``workers`` bounds CPU solvers and
    ``smoke`` participates in the cache signature.  Returns training/validation/
    primary-test mappings, four named test splits and an audit manifest.  Formally it
    builds 24576 training edges, 2048 validation cases and four test sets of 2048.
    It writes cache files and runs reference solvers only on a cache miss.
    ``combination_generalization.run_stage`` calls it before any GPU training.
    """
    signature = combination_dataset_signature(config, seed, smoke)
    if (directory / "manifest.json").exists():
        LOGGER.info("Reusing combination dataset: %s", directory)
        return load_combination_dataset(directory, signature)

    protocol = config["combination_generalization"]
    if tuple(protocol["split_order"]) != COMBINATION_SPLIT_NAMES:
        raise ValueError("combination split order must match the fixed four-way protocol")
    if str(protocol["primary_split"]) != "unseen_history_unseen_parameter":
        raise ValueError("UH--UP must remain the predeclared primary split")
    if str(config["operator"].get("physics_pairing_mode")) != "training_edges":
        raise ValueError("combination training requires physics_pairing_mode=training_edges")

    data = config["data"]
    equation = NicholsonConfig.from_mapping(config["equation"])
    training_bounds = np.asarray(protocol["training_parameter_bounds"], dtype=np.float64)
    physical_bounds = np.asarray(equation.parameter_bounds, dtype=np.float64)
    if (
        training_bounds.shape != (2, 2)
        or np.any(training_bounds[:, 0] < physical_bounds[:, 0])
        or np.any(training_bounds[:, 1] > physical_bounds[:, 1])
    ):
        raise ValueError("training parameter bounds must lie inside the physical box")
    rng = np.random.default_rng(int(seed))
    history_times = np.linspace(
        -equation.maximum_history, 0.0, int(data["history_sensors"]), dtype=np.float64
    )
    output_times = np.linspace(
        0.0, float(data["horizon"]), int(data["output_points"]), dtype=np.float64
    )
    history_arguments = (
        history_times,
        tuple(float(item) for item in data["history_mean"]),
        float(data["history_sigma"]),
        float(data["history_length_scale"]),
        tuple(float(item) for item in data["history_bounds"]),
    )
    train_history_count = int(data["history_count"])
    train_parameter_count = int(data["parameter_count"])
    validation_history_count = int(protocol["validation_history_count"])
    validation_parameter_count = int(protocol["validation_parameter_count"])
    seen_history_count = int(protocol["seen_test_history_count"])
    unseen_history_count = int(protocol["unseen_test_history_count"])
    unseen_parameter_count = int(protocol["unseen_parameter_count"])
    evaluation_pairs = int(protocol["evaluation_pairs_per_history"])
    if seen_history_count > train_history_count:
        raise ValueError("seen test history count exceeds training history count")

    train_history_pool = sample_histories(train_history_count, *history_arguments, rng)
    validation_history_pool = sample_histories(
        validation_history_count, *history_arguments, rng
    )
    unseen_history_pool = sample_histories(unseen_history_count, *history_arguments, rng)
    train_parameter_pool = _sample_parameters_in_bounds(
        train_parameter_count, training_bounds, rng
    )
    validation_parameter_pool = _sample_parameters_in_bounds(
        validation_parameter_count, training_bounds, rng
    )
    unseen_parameter_pool = _sample_parameters_in_bounds(
        unseen_parameter_count, training_bounds, rng
    )
    history_sets = [
        {row.tobytes() for row in pool}
        for pool in (train_history_pool, validation_history_pool, unseen_history_pool)
    ]
    parameter_sets = [
        {row.tobytes() for row in pool}
        for pool in (train_parameter_pool, validation_parameter_pool, unseen_parameter_pool)
    ]
    if any(
        history_sets[left] & history_sets[right]
        for left in range(3) for right in range(left + 1, 3)
    ):
        raise RuntimeError("train/validation/unseen history pools overlap")
    if any(
        parameter_sets[left] & parameter_sets[right]
        for left in range(3) for right in range(left + 1, 3)
    ):
        raise RuntimeError("train/validation/unseen parameter pools overlap")
    train_h, train_p = balanced_pair_indices(
        train_history_count, train_parameter_count, int(data["pairs_per_history"]), rng
    )
    validation_h, validation_p = balanced_pair_indices(
        validation_history_count, validation_parameter_count, evaluation_pairs, rng
    )
    selected_histories = rng.permutation(train_history_count)[:seen_history_count]
    shsp_h, shsp_p = _held_out_seen_edges(
        train_h, train_p, selected_histories, train_parameter_count, evaluation_pairs, rng
    )
    shup_local_h, shup_p = balanced_pair_indices(
        seen_history_count, unseen_parameter_count, evaluation_pairs, rng
    )
    uhsp_h, uhsp_p = balanced_pair_indices(
        unseen_history_count, train_parameter_count, evaluation_pairs, rng
    )
    uhup_h, uhup_p = balanced_pair_indices(
        unseen_history_count, unseen_parameter_count, evaluation_pairs, rng
    )
    split_inputs = {
        "seen_history_seen_parameter": (
            train_history_pool[shsp_h], train_parameter_pool[shsp_p]
        ),
        "seen_history_unseen_parameter": (
            train_history_pool[selected_histories[shup_local_h]],
            unseen_parameter_pool[shup_p],
        ),
        "unseen_history_seen_parameter": (
            unseen_history_pool[uhsp_h], train_parameter_pool[uhsp_p]
        ),
        "unseen_history_unseen_parameter": (
            unseen_history_pool[uhup_h], unseen_parameter_pool[uhup_p]
        ),
    }

    common_solver = (
        history_times, output_times, config["equation"], float(data["internal_step"]),
        workers, int(data["solver_chunk_size"]),
    )
    LOGGER.info("Solving combination training split with %d cases", train_h.size)
    train_split = DatasetSplit(
        train_history_pool[train_h], train_parameter_pool[train_p],
        solve_parallel(
            train_history_pool[train_h], train_parameter_pool[train_p], *common_solver
        ),
        history_times.astype(np.float32), output_times.astype(np.float32),
    )
    LOGGER.info("Solving combination validation split with %d cases", validation_h.size)
    validation_split = DatasetSplit(
        validation_history_pool[validation_h], validation_parameter_pool[validation_p],
        solve_parallel(
            validation_history_pool[validation_h],
            validation_parameter_pool[validation_p],
            *common_solver,
        ),
        history_times.astype(np.float32), output_times.astype(np.float32),
    )
    named_splits: dict[str, DatasetSplit] = {}
    for split_name in COMBINATION_SPLIT_NAMES:
        histories, parameters = split_inputs[split_name]
        LOGGER.info("Solving %s with %d cases", split_name, parameters.shape[0])
        named_splits[split_name] = DatasetSplit(
            histories, parameters,
            solve_parallel(histories, parameters, *common_solver),
            history_times.astype(np.float32), output_times.astype(np.float32),
        )

    directory.mkdir(parents=True, exist_ok=True)
    _save_combination_split(directory / "train.npz", train_split)
    _save_combination_split(directory / "validation.npz", validation_split)
    for split_name, split in named_splits.items():
        _save_combination_split(directory / f"test_{split_name}.npz", split)
    training_edges = set(zip(train_h.tolist(), train_p.tolist()))
    held_out_edges = set(zip(shsp_h.tolist(), shsp_p.tolist()))
    manifest: dict[str, Any] = {
        "complete": True,
        "protocol": "history_parameter_combinatorial_holdout_v1",
        "signature": signature,
        "data_seed": int(seed),
        "smoke": bool(smoke),
        "split_order": list(COMBINATION_SPLIT_NAMES),
        "primary_split": str(protocol["primary_split"]),
        "training_parameter_bounds": training_bounds.tolist(),
        "physical_parameter_bounds": physical_bounds.tolist(),
        "train_pair_count": int(train_h.size),
        "validation_count": int(validation_h.size),
        "test_counts": {
            name: int(split.parameters.shape[0]) for name, split in named_splits.items()
        },
        "leakage_audit": {
            "training_edge_overlap": int(len(training_edges & held_out_edges)),
            "history_pools_disjoint": True,
            "parameter_pools_disjoint": True,
            "primary_uses_unseen_histories_and_parameters": True,
            "physics_uses_training_edges": True,
        },
    }
    temporary_manifest = directory / ".manifest.json"
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary_manifest.replace(directory / "manifest.json")
    return load_combination_dataset(directory, signature)
