from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

CLASS_NAMES = ("healthy", "early_blight", "late_blight")
CLASSIFICATION_SPLITS = ("train", "val", "test")
YOLO_SPLITS = ("train", "val")


class ConfigPathError(FileNotFoundError):
    """Raised when a required project or dataset path cannot be resolved."""


def find_project_root(start: Path | None = None) -> Path:
    """Walk upward from *start* until a repository marker is found."""
    cursor = (start or Path(__file__).resolve()).parent
    markers = ("pyproject.toml", "notebooks", "data")
    for candidate in [cursor, *cursor.parents]:
        if (candidate / "pyproject.toml").exists() and (candidate / "data").exists():
            return candidate
        if all((candidate / name).exists() for name in markers):
            return candidate
    raise ConfigPathError(
        "Could not locate the Potato Leaf Vision project root from "
        f"{start or Path(__file__).resolve()}. Expected a directory containing "
        "data/ and pyproject.toml."
    )


@dataclass(frozen=True)
class ProjectPaths:
    root: Path

    @classmethod
    def from_root(cls, root: Path | None = None) -> "ProjectPaths":
        resolved = Path(root).resolve() if root is not None else find_project_root()
        return cls(root=resolved)

    @property
    def data_raw(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def data_yolo(self) -> Path:
        return self.root / "data" / "yolo"

    @property
    def yolo_yaml(self) -> Path:
        return self.data_yolo / "data.yaml"

    def classification_split(self, split: str) -> Path:
        if split not in CLASSIFICATION_SPLITS:
            raise ValueError(f"Unknown classification split: {split!r}")
        return self.root / "data" / split

    def require_exists(self, path: Path, what: str) -> Path:
        if not path.exists():
            raise ConfigPathError(
                f"{what} was not found at {path}. "
                "No fallback location was guessed."
            )
        return path


def _as_path(value: Any) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ConfigPathError(
            f"YOLO data.yaml field must be a non-empty string, got {value!r}."
        )
    return Path(value)


def load_yolo_data_yaml(yaml_path: Path) -> dict[str, Any]:
    """Load Ultralytics data.yaml and resolve dataset paths relative to the file.

    Absolute paths are accepted only if they exist. Missing absolute or relative
    paths raise ConfigPathError instead of searching other machines or folders.
    """
    yaml_path = Path(yaml_path).resolve()
    if not yaml_path.is_file():
        raise ConfigPathError(f"YOLO config was not found at {yaml_path}.")

    with yaml_path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    if not isinstance(payload, dict):
        raise ConfigPathError(f"YOLO config at {yaml_path} is not a mapping.")

    dataset_root = _as_path(payload.get("path", "."))
    if not dataset_root.is_absolute():
        dataset_root = (yaml_path.parent / dataset_root).resolve()
    elif not dataset_root.exists():
        raise ConfigPathError(
            f"YOLO dataset path {dataset_root} from {yaml_path} does not exist. "
            "Update data.yaml to a path relative to this repository "
            "(path: .) instead of a machine-specific absolute location."
        )

    if not dataset_root.exists():
        raise ConfigPathError(
            f"Resolved YOLO dataset root {dataset_root} does not exist. "
            "No alternate dataset location was guessed."
        )

    resolved_splits: dict[str, Path] = {}
    for split in YOLO_SPLITS:
        if split not in payload:
            raise ConfigPathError(
                f"YOLO config {yaml_path} is missing required key {split!r}."
            )
        split_path = _as_path(payload[split])
        if not split_path.is_absolute():
            split_path = (dataset_root / split_path).resolve()
        if not split_path.exists():
            raise ConfigPathError(
                f"YOLO {split} path {split_path} does not exist."
            )
        resolved_splits[split] = split_path

    names = payload.get("names", {})
    return {
        "yaml_path": yaml_path,
        "dataset_root": dataset_root,
        "train": resolved_splits["train"],
        "val": resolved_splits["val"],
        "nc": payload.get("nc"),
        "names": names,
        "raw": payload,
    }
