from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .dataset_checks import IMAGE_EXTENSIONS, is_image_file

ALLOWED_CLASS_IDS = {0}


class YoloAnnotationError(ValueError):
    """Raised when a YOLO label file cannot be parsed or is out of range."""


@dataclass(frozen=True)
class YoloBox:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float
    source: Path
    line_number: int


def parse_yolo_label_file(path: Path) -> list[YoloBox]:
    text = path.read_text(encoding="utf-8")
    boxes: list[YoloBox] = []
    for line_number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise YoloAnnotationError(
                f"{path}:{line_number} expected 5 fields "
                f"(class x y w h), got {len(parts)}: {line!r}"
            )
        try:
            class_id = int(parts[0])
            x_center, y_center, width, height = (float(value) for value in parts[1:])
        except ValueError as exc:
            raise YoloAnnotationError(
                f"{path}:{line_number} could not parse numbers: {line!r}"
            ) from exc
        boxes.append(
            YoloBox(
                class_id=class_id,
                x_center=x_center,
                y_center=y_center,
                width=width,
                height=height,
                source=path,
                line_number=line_number,
            )
        )
    return boxes


def validate_boxes(
    boxes: list[YoloBox],
    allowed_class_ids: set[int] | None = None,
) -> list[str]:
    """Return human-readable problems; empty means the boxes are valid."""
    allowed = allowed_class_ids if allowed_class_ids is not None else ALLOWED_CLASS_IDS
    problems: list[str] = []
    if not boxes:
        problems.append("label file contains no boxes")
        return problems

    for box in boxes:
        loc = f"{box.source}:{box.line_number}"
        if box.class_id not in allowed:
            problems.append(f"{loc} class_id {box.class_id} is not in {sorted(allowed)}")
        for name, value in (
            ("x_center", box.x_center),
            ("y_center", box.y_center),
            ("width", box.width),
            ("height", box.height),
        ):
            if not 0.0 <= value <= 1.0:
                problems.append(f"{loc} {name}={value} is outside [0, 1]")
        if box.width <= 0 or box.height <= 0:
            problems.append(f"{loc} width/height must be positive")
        x_min = box.x_center - box.width / 2
        x_max = box.x_center + box.width / 2
        y_min = box.y_center - box.height / 2
        y_max = box.y_center + box.height / 2
        if x_min < -1e-6 or y_min < -1e-6 or x_max > 1 + 1e-6 or y_max > 1 + 1e-6:
            problems.append(
                f"{loc} box extent [{x_min:.4f}, {y_min:.4f}, {x_max:.4f}, {y_max:.4f}] "
                "leaves the image"
            )
    return problems


def _stems_in(directory: Path, suffixes: set[str]) -> set[str]:
    if not directory.exists():
        return set()
    return {
        path.stem
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in suffixes
    }


def unpaired_stems(image_dir: Path, label_dir: Path) -> tuple[set[str], set[str]]:
    images = _stems_in(image_dir, IMAGE_EXTENSIONS)
    labels = _stems_in(label_dir, {".txt"})
    return images - labels, labels - images


def image_label_pairs(image_dir: Path, label_dir: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    if not image_dir.exists():
        return pairs
    for image in sorted(path for path in image_dir.iterdir() if is_image_file(path)):
        label = label_dir / f"{image.stem}.txt"
        if label.exists():
            pairs.append((image, label))
    return pairs
