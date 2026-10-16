from pathlib import Path

import pytest

from potato_vision.dataset_checks import (
    list_split_stems,
    missing_paths,
    split_filename_overlap,
)
from potato_vision.paths import CLASS_NAMES, CLASSIFICATION_SPLITS, ProjectPaths


def test_toy_splits_are_disjoint(toy_classification_root: Path) -> None:
    splits = {
        split: list_split_stems(
            toy_classification_root / "data" / split,
            CLASS_NAMES,
        )
        for split in CLASSIFICATION_SPLITS
    }
    assert split_filename_overlap(splits) == {}
    assert all(splits[name] for name in CLASSIFICATION_SPLITS)


def test_missing_file_detection(tmp_path: Path) -> None:
    present = tmp_path / "present.jpg"
    present.write_bytes(b"not-an-image-but-exists")
    absent = tmp_path / "missing.jpg"
    missing = missing_paths([present, absent])
    assert missing == [absent]


def test_overlapping_splits_are_reported() -> None:
    overlap = split_filename_overlap(
        {
            "train": {"shared", "only_train"},
            "test": {"shared", "only_test"},
        }
    )
    assert overlap[("train", "test")] == {"shared"}


def test_live_classification_splits_disjoint_if_present() -> None:
    """Uses the real tree only when all three split folders exist."""
    paths = ProjectPaths.from_root()
    split_dirs = {split: paths.classification_split(split) for split in CLASSIFICATION_SPLITS}
    if any(missing_paths(split_dirs.values())):
        pytest.skip("classification split directories are not present")

    stems = {
        split: list_split_stems(directory, CLASS_NAMES)
        for split, directory in split_dirs.items()
    }
    overlap = split_filename_overlap(stems)
    assert overlap == {}, f"filename overlap across splits: {overlap}"
