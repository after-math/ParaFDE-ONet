"""Optional, failure-isolated offline Weights & Biases tracking."""

from __future__ import annotations

import logging
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping


class WandBTracker:
    """Isolate every optional W&B operation from the scientific computation.

    The constructor receives the resolved W&B mapping, run directory, complete
    config, run name and logger.  When disabled or unavailable all public methods
    are safe no-ops.  A typical formal run records epoch scalars and one gradient
    histogram set per logical epoch into an offline directory.  ``train_operator``
    owns one tracker per method and calls ``finish`` after local results are saved.
    """

    def __init__(
        self,
        config: Mapping[str, Any],
        output_dir: Path,
        resolved_config: Mapping[str, Any],
        run_name: str,
        logger: logging.Logger,
    ) -> None:
        """Attempt a local W&B initialization without making training depend on it.

        ``config`` contains enabled/mode/project/entity, ``output_dir`` receives the
        offline run, and the remaining arguments supply run metadata and logs.  The
        constructor returns ``None``.  On import/init failure it emits one warning
        and disables itself; otherwise it creates a local W&B run.  ``train_operator``
        calls it before optimization.
        """
        self.logger = logger
        self.run: Any | None = None
        self.wandb: Any | None = None
        self.enabled = bool(config.get("enabled", False)) and str(config.get("mode")) != "disabled"
        if not self.enabled:
            return
        try:
            import wandb

            output_dir.mkdir(parents=True, exist_ok=True)
            self.wandb = wandb
            self.run = wandb.init(
                project=str(config.get("project", "delayed-sei3d-operator")),
                entity=config.get("entity"),
                name=run_name,
                mode=str(config.get("mode", "offline")),
                dir=str(output_dir),
                config=dict(resolved_config),
                reinit=True,
                settings=wandb.Settings(start_method="thread"),
            )
            self.logger.info("W&B initialized in %s mode: %s", config.get("mode"), run_name)
        except Exception:
            self.logger.warning("W&B initialization failed; core training continues", exc_info=True)
            self.enabled = False
            self.run = None

    def log_epoch(self, metrics: Mapping[str, float], epoch: int) -> None:
        """Record one logical epoch of scalar metrics with ``step=epoch``.

        ``metrics`` normally contains train/data/physics/validation losses and LR;
        ``epoch`` is a positive integer.  Returns ``None``.  A W&B failure produces
        a warning and disables later logging without affecting optimization.
        ``training.train_operator`` calls it after every 500 training iterations.
        """
        if not self.enabled or self.run is None:
            return
        try:
            self.run.log(dict(metrics), step=int(epoch), commit=True)
        except Exception:
            self.logger.warning("W&B scalar logging failed; disabling tracker", exc_info=True)
            self.enabled = False

    def log_gradients(self, model: Any, epoch: int) -> None:
        """Log existing parameter gradients before clipping and optimizer update.

        ``model`` is the current operator and ``epoch`` the common W&B step.  Each
        non-None gradient becomes one histogram; no gradient is changed.  Returns
        ``None``.  Failures disable W&B only.  The training loop calls this on the
        last effective batch of each logical epoch, immediately after ``backward``.
        """
        if not self.enabled or self.run is None or self.wandb is None:
            return
        try:
            values = {
                f"gradients/{name}": self.wandb.Histogram(parameter.grad.detach().float().cpu().numpy())
                for name, parameter in model.named_parameters()
                if parameter.grad is not None
            }
            if values:
                # Gradients and the later validation scalars share one epoch step.
                # Keeping this call uncommitted prevents W&B from advancing the
                # step before validation has been computed.
                self.run.log(values, step=int(epoch), commit=False)
        except Exception:
            self.logger.warning("W&B gradient logging failed; disabling tracker", exc_info=True)
            self.enabled = False

    def log_figure(self, name: str, path: Path, step: int) -> None:
        """Attach one already-saved local result figure to the optional run.

        ``name`` is its dashboard key, ``path`` points to PNG/SVG and ``step`` is an
        integer.  Returns ``None`` and never deletes the local source.  Failures only
        warn and disable tracking.  Reporting or training may call it after plotting.
        """
        if not self.enabled or self.run is None or self.wandb is None:
            return
        try:
            self.run.log({name: self.wandb.Image(str(path))}, step=int(step))
        except Exception:
            self.logger.warning("W&B figure logging failed; disabling tracker", exc_info=True)
            self.enabled = False

    def finish(self, success: bool) -> None:
        """Close the local run and record whether core computation succeeded.

        ``success`` is true only after checkpoints, metrics and figures are saved.
        The method returns ``None``.  It writes one final scalar and closes W&B;
        failure is a warning and does not change scientific success.  The owning
        pipeline job calls it exactly once in ``finally``.
        """
        if self.run is None:
            return
        try:
            self.run.log({"experiment/success": int(bool(success))})
            self.run.finish(exit_code=0 if success else 1)
        except Exception:
            self.logger.warning("W&B finish failed; local scientific outputs remain valid", exc_info=True)
        finally:
            self.run = None


def sync_offline_runs(output_root: Path, logger: logging.Logger) -> tuple[bool, str]:
    """Synchronize every completed local offline run after full experiment success.

    ``output_root`` is the complete experiment directory and ``logger`` records each
    subprocess result. The return is ``(success,message)``; for example no local run
    returns true with an explanatory message. It recursively finds directories named
    ``offline-run-*`` and executes the installed W&B module once per directory.
    Sync failures are returned, never raised, so checkpoints and metrics stay valid.
    The ordinary and combination pipeline entries call it only after full evaluation
    and all local plots succeed.
    """
    run_paths = sorted(
        path for path in output_root.rglob("offline-run-*") if path.is_dir()
    )
    if not run_paths:
        message = "No W&B offline runs were found; nothing to sync"
        logger.info(message)
        return True, message
    failures: list[str] = []
    for run_path in run_paths:
        try:
            completed = subprocess.run(
                [sys.executable, "-m", "wandb", "sync", str(run_path)],
                check=False,
                capture_output=True,
                text=True,
                timeout=600,
            )
            if completed.returncode != 0:
                failures.append(f"{run_path}: {completed.stderr.strip()}")
            else:
                logger.info("W&B sync succeeded: %s", run_path)
        except Exception as error:
            failures.append(f"{run_path}: {error!r}")
    if failures:
        message = "W&B sync failed for " + "; ".join(failures)
        logger.warning(message)
        return False, message
    message = f"W&B sync completed for {len(run_paths)} offline run(s)"
    logger.info(message)
    return True, message
