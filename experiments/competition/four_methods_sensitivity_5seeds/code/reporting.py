"""Unchanged reporting implementation reused from the parent project."""

from _source_patch import execute_source, parent_code_path

_SOURCE = parent_code_path(__file__, "reporting.py")
execute_source(_SOURCE.read_text(encoding="utf-8"), globals(), __file__)
