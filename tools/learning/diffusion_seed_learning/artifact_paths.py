"""Helpers for resolving standalone diffusion artifact pointers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ARTIFACT_POINTER = REPO_ROOT / "artifacts/current_artifacts.json"


def load_artifacts(pointer: str | Path = DEFAULT_ARTIFACT_POINTER) -> dict[str, Any]:
    path = Path(pointer)
    return json.loads(path.read_text(encoding="utf-8"))


def artifact_path(*keys: str, pointer: str | Path = DEFAULT_ARTIFACT_POINTER, default: str | None = None) -> Path:
    value: Any = load_artifacts(pointer)
    for key in keys:
        if not isinstance(value, dict):
            return Path(default or "")
        value = value.get(key)
    if isinstance(value, dict) and "path" in value:
        value = value.get("path")
    return Path(str(value or default or ""))


def public_root(pointer: str | Path = DEFAULT_ARTIFACT_POINTER) -> Path:
    artifacts = load_artifacts(pointer)
    dataset_path = artifacts.get("dataset", {}).get("training_dataset", {}).get("path")
    if dataset_path:
        return Path(dataset_path).parents[2]
    return Path("/pub/data/caohy/tashan_Manipulation/diffusionSeedLearning")

