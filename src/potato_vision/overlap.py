from __future__ import annotations

from pathlib import Path

from .dataset_checks import list_split_stems
from .paths import CLASS_NAMES, ProjectPaths, load_yolo_data_yaml
from .yolo_checks import unpaired_stems


def yolo_image_stems(dataset_root: Path) -> set[str]:
    stems: set[str] = set()
    for split in ("train", "val"):
        image_dir = dataset_root / "images" / split
        label_dir = dataset_root / "labels" / split
        missing_images, missing_labels = unpaired_stems(image_dir, label_dir)
        if missing_images or missing_labels:
            raise ValueError(
                f"YOLO {split} is unpaired: images without labels={sorted(missing_images)}; "
                f"labels without images={sorted(missing_labels)}"
            )
        stems.update(path.stem.lower() for path in image_dir.iterdir() if path.is_file())
    return stems


def classification_yolo_overlap(root: Path | None = None) -> dict[str, set[str]]:
    """Stems that appear in both the YOLO subset and each classification split."""
    paths = ProjectPaths.from_root(root)
    yolo = yolo_image_stems(load_yolo_data_yaml(paths.yolo_yaml)["dataset_root"])
    return {
        split: yolo & list_split_stems(paths.classification_split(split), CLASS_NAMES)
        for split in ("train", "val", "test")
    }
