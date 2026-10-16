from pathlib import Path

import pytest

from potato_vision.paths import ProjectPaths, load_yolo_data_yaml
from potato_vision.yolo_checks import (
    YoloAnnotationError,
    parse_yolo_label_file,
    unpaired_stems,
    validate_boxes,
)


def test_valid_label_parses(tmp_path: Path) -> None:
    label = tmp_path / "ok.txt"
    label.write_text("0 0.4 0.6 0.2 0.3\n", encoding="utf-8")
    boxes = parse_yolo_label_file(label)
    assert len(boxes) == 1
    assert boxes[0].class_id == 0
    assert validate_boxes(boxes) == []


def test_invalid_field_count(tmp_path: Path) -> None:
    label = tmp_path / "bad.txt"
    label.write_text("0 0.1 0.2 0.3\n", encoding="utf-8")
    with pytest.raises(YoloAnnotationError, match="expected 5 fields"):
        parse_yolo_label_file(label)


def test_out_of_range_coordinates(tmp_path: Path) -> None:
    label = tmp_path / "oob.txt"
    label.write_text("0 1.5 0.5 0.2 0.2\n", encoding="utf-8")
    problems = validate_boxes(parse_yolo_label_file(label))
    assert any("x_center" in item for item in problems)


def test_disallowed_class_id(tmp_path: Path) -> None:
    label = tmp_path / "cls.txt"
    label.write_text("3 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    problems = validate_boxes(parse_yolo_label_file(label))
    assert any("class_id 3" in item for item in problems)


def test_empty_label_is_invalid(tmp_path: Path) -> None:
    label = tmp_path / "empty.txt"
    label.write_text("\n", encoding="utf-8")
    problems = validate_boxes(parse_yolo_label_file(label))
    assert problems == ["label file contains no boxes"]


def test_toy_yolo_pairs_and_config(toy_yolo_root: Path) -> None:
    yaml_path = toy_yolo_root / "data" / "yolo" / "data.yaml"
    config = load_yolo_data_yaml(yaml_path)
    missing_images, missing_labels = unpaired_stems(
        config["train"],
        config["dataset_root"] / "labels" / "train",
    )
    assert missing_images == set()
    assert missing_labels == set()
    assert config["nc"] == 1


def test_live_yolo_labels_if_present() -> None:
    paths = ProjectPaths.from_root()
    if not paths.yolo_yaml.exists():
        pytest.skip("data/yolo/data.yaml is not present")

    config = load_yolo_data_yaml(paths.yolo_yaml)
    for split in ("train", "val"):
        image_dir = config[split]
        label_dir = config["dataset_root"] / "labels" / split
        missing_images, missing_labels = unpaired_stems(image_dir, label_dir)
        assert missing_images == set(), f"{split} images without labels: {missing_images}"
        assert missing_labels == set(), f"{split} labels without images: {missing_labels}"

        for label in sorted(label_dir.glob("*.txt")):
            problems = validate_boxes(parse_yolo_label_file(label))
            assert problems == [], f"{label}: {problems}"
