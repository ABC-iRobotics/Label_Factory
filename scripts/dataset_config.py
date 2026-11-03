"""Shared helpers for resolving numeric or named dataset directories."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def normalize_dataset_name(value: Any) -> str:
    """Return a safe directory name while retaining legacy zero-padded IDs."""
    if isinstance(value, bool) or value is None:
        raise ValueError("dataset_id must be a number or a non-empty name")
    raw = str(value).strip()
    if not raw:
        raise ValueError("dataset_id cannot be empty")
    if raw.isdecimal():
        return raw.zfill(4)
    if raw in {".", ".."} or Path(raw).name != raw or "/" in raw or "\\" in raw:
        raise ValueError("dataset_id must be a single directory name, not a path")
    if any(ord(character) < 32 for character in raw):
        raise ValueError("dataset_id cannot contain control characters")
    return raw


def dataset_path(project: str | Path, dataset_id: Any) -> Path:
    return (Path(project).expanduser().resolve() / "data" /
            normalize_dataset_name(dataset_id))
