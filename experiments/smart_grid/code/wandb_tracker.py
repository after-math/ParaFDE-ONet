"""Failure-isolated optional Weights & Biases logging."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping


class WandBTracker:
    def __init__(
        self,
        config: Mapping[str, Any],
        output_dir: Path,
        resolved_config: Mapping[str, Any],
        run_name: str,
        logger: logging.Logger,
    ) -> None:
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
                project=str(config.get("project", "delayed-smart-grid4node8d-parafdeonet")),
                entity=config.get("entity"),
                name=run_name,
                mode=str(config.get("mode", "offline")),
                dir=str(output_dir),
                config=dict(resolved_config),
                reinit=True,
                settings=wandb.Settings(start_method="thread"),
            )
        except Exception:
            self.logger.warning("W&B initialization failed; training continues", exc_info=True)
            self.enabled = False
            self.run = None

    def log(self, values: Mapping[str, float], step: int, commit: bool = True) -> None:
        if not self.enabled or self.run is None:
            return
        try:
            self.run.log(dict(values), step=int(step), commit=commit)
        except Exception:
            self.logger.warning("W&B scalar logging failed; disabling tracker", exc_info=True)
            self.enabled = False

    def log_gradients(self, model: Any, step: int) -> None:
        if not self.enabled or self.run is None or self.wandb is None:
            return
        try:
            values = {
                f"gradients/{name}": self.wandb.Histogram(parameter.grad.detach().float().cpu().numpy())
                for name, parameter in model.named_parameters()
                if parameter.grad is not None
            }
            if values:
                self.run.log(values, step=int(step), commit=False)
        except Exception:
            self.logger.warning("W&B gradient logging failed; disabling tracker", exc_info=True)
            self.enabled = False

    def finish(self, success: bool) -> None:
        if self.run is None:
            return
        try:
            self.run.log({"experiment/success": int(bool(success))})
            self.run.finish(exit_code=0 if success else 1)
        except Exception:
            self.logger.warning("W&B finish failed; local outputs remain valid", exc_info=True)
        finally:
            self.run = None
