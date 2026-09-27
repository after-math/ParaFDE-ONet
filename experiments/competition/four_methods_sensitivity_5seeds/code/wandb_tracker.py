"""Unchanged optional W&B integration reused from the parent project."""

from _source_patch import execute_source, parent_code_path

_SOURCE = parent_code_path(__file__, "wandb_tracker.py")
execute_source(_SOURCE.read_text(encoding="utf-8"), globals(), __file__)
