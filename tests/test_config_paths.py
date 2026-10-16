from pathlib import Path

import pytest
import yaml

from potato_vision.paths import ConfigPathError, ProjectPaths, load_yolo_data_yaml


def test_relative_yolo_yaml_resolves(toy_yolo_root: Path) -> None:
    yaml_path = toy_yolo_root / "data" / "yolo" / "data.yaml"
    config = load_yolo_data_yaml(yaml_path)
    assert config["dataset_root"] == yaml_path.parent.resolve()
    assert config["train"].is_dir()
    assert config["val"].is_dir()


def test_missing_yaml_raises(tmp_path: Path) -> None:
    missing = tmp_path / "data.yaml"
    with pytest.raises(ConfigPathError, match="was not found"):
        load_yolo_data_yaml(missing)


def test_missing_absolute_path_is_not_guessed(tmp_path: Path) -> None:
    yaml_path = tmp_path / "data.yaml"
    payload = {
        "path": "C:/Users/user/potato-leaf-vision/data/yolo",
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": {0: "late_blight"},
    }
    yaml_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ConfigPathError, match="does not exist"):
        load_yolo_data_yaml(yaml_path)


def test_missing_split_directory_raises(toy_yolo_root: Path) -> None:
    yaml_path = toy_yolo_root / "data" / "yolo" / "data.yaml"
    payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    payload["val"] = "images/does-not-exist"
    yaml_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ConfigPathError, match="val path"):
        load_yolo_data_yaml(yaml_path)


def test_live_repo_yaml_is_portable() -> None:
    paths = ProjectPaths.from_root()
    if not paths.yolo_yaml.exists():
        pytest.skip("data/yolo/data.yaml is not present")
    text = paths.yolo_yaml.read_text(encoding="utf-8")
    assert "C:/Users/user/" not in text
    assert "C:\\Users\\user\\" not in text
    config = load_yolo_data_yaml(paths.yolo_yaml)
    assert config["dataset_root"] == paths.data_yolo.resolve()


def test_project_paths_require_exists(tmp_path: Path) -> None:
    paths = ProjectPaths.from_root(tmp_path)
    with pytest.raises(ConfigPathError, match="No fallback location was guessed"):
        paths.require_exists(tmp_path / "missing-dir", "dataset")
