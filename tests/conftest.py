from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from PIL import Image


def _write_jpeg(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color=(20, 140, 40)).save(path, format="JPEG")


@pytest.fixture
def toy_classification_root(tmp_path: Path) -> Path:
    """Tiny disjoint train/val/test tree. Does not use the real dataset."""
    classes = ("healthy", "early_blight", "late_blight")
    layout = {
        "train": {
            "healthy": ["train_h1.jpg"],
            "early_blight": ["train_eb1.jpg"],
            "late_blight": ["train_lb1.jpg"],
        },
        "val": {
            "healthy": ["val_h1.jpg"],
            "early_blight": ["val_eb1.jpg"],
            "late_blight": ["val_lb1.jpg"],
        },
        "test": {
            "healthy": ["test_h1.jpg"],
            "early_blight": ["test_eb1.jpg"],
            "late_blight": ["test_lb1.jpg"],
        },
    }
    for split, by_class in layout.items():
        for class_name in classes:
            for filename in by_class[class_name]:
                _write_jpeg(tmp_path / "data" / split / class_name / filename)
    return tmp_path


@pytest.fixture
def toy_yolo_root(tmp_path: Path) -> Path:
    yolo = tmp_path / "data" / "yolo"
    for split, stem in (("train", "leaf_a"), ("val", "leaf_b")):
        image = yolo / "images" / split / f"{stem}.jpg"
        label = yolo / "labels" / split / f"{stem}.txt"
        _write_jpeg(image)
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text("0 0.50 0.50 0.20 0.25\n", encoding="utf-8")

    payload = {
        "path": ".",
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": {0: "late_blight"},
    }
    yaml_path = yolo / "data.yaml"
    yaml_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return tmp_path
