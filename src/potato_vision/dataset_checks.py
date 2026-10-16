
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def is_image_file(path: Path) -> bool:
    """Return True if the path is an existing image file."""
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def list_image_files(directory: Path) -> list[Path]:
    """List image files directly inside a directory, without recursion."""
    if not directory.exists():
        return []

    return sorted(
        path for path in directory.iterdir()
        if is_image_file(path)
    )


def list_split_stems(
    split_dir: Path,
    classes: Sequence[str] | None = None,
) -> set[str]:
    """
    Return image filename stems from a classification split.

    Only direct image files inside each class directory are included.
    Nested processing outputs, masks, and segmentation artifacts are ignored.
    """
    stems: set[str] = set()

    if classes:
        search_roots = [split_dir / class_name for class_name in classes]
    else:
        search_roots = [split_dir]

    for root in search_roots:
        if not root.is_dir():
            continue

        for path in root.iterdir():
            if is_image_file(path):
                stems.add(path.stem.lower())

    return stems


def split_filename_overlap(
    splits: Mapping[str, Iterable[str]],
) -> dict[tuple[str, str], set[str]]:
    """Return filename stem overlaps between every pair of named splits."""
    names = list(splits)
    overlap: dict[tuple[str, str], set[str]] = {}

    for i, left in enumerate(names):
        left_set = set(splits[left])

        for right in names[i + 1:]:
            shared = left_set & set(splits[right])

            if shared:
                overlap[(left, right)] = shared

    return overlap


def missing_paths(paths: Iterable[Path]) -> list[Path]:
    """Return paths that do not exist."""
    return [path for path in paths if not path.exists()]
