
from __future__ import annotations

from pathlib import Path
import math

import cv2
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Choose the dataset split: "train" or "val".
SPLIT = "train"

IMAGE_DIR = PROJECT_ROOT / "data" / "yolo" / "images" / SPLIT
LABEL_DIR = PROJECT_ROOT / "data" / "yolo" / "labels" / SPLIT
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "yolo_label_preview" / SPLIT

# Set to 0 to visualize every matching image.
MAX_IMAGES = 0

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
COLUMNS = 3
FIGURE_WIDTH = 15
FIGURE_HEIGHT_PER_ROW = 4


def draw_yolo_boxes(image_path: Path, label_path: Path):
    """Draw normalized YOLO bounding boxes on an image."""
    image = cv2.imread(str(image_path))

    if image is None:
        print(f"WARNING: Could not read image: {image_path.name}")
        return None

    height, width = image.shape[:2]

    try:
        with label_path.open("r", encoding="utf-8") as file:
            lines = [
                line.strip()
                for line in file
                if line.strip()
            ]
    except (OSError, UnicodeError) as exc:
        print(f"WARNING: Could not read label {label_path.name}: {exc}")
        return None

    box_count = 0

    for line_number, line in enumerate(lines, start=1):
        values = line.split()

        if len(values) != 5:
            print(
                f"WARNING: Unexpected label format in "
                f"{label_path.name}, line {line_number}"
            )
            continue

        try:
            class_value, x_center, y_center, box_width, box_height = (
                float(value) for value in values
            )
        except ValueError:
            print(
                f"WARNING: Invalid numbers in "
                f"{label_path.name}, line {line_number}"
            )
            continue

        if not class_value.is_integer():
            print(
                f"WARNING: Non-integer class ID in "
                f"{label_path.name}, line {line_number}"
            )
            continue

        if not all(
            math.isfinite(value)
            for value in (
                x_center,
                y_center,
                box_width,
                box_height,
            )
        ):
            print(
                f"WARNING: Non-finite coordinates in "
                f"{label_path.name}, line {line_number}"
            )
            continue

        # YOLO coordinates are normalized to image dimensions.
        x1 = round((x_center - box_width / 2) * width)
        y1 = round((y_center - box_height / 2) * height)
        x2 = round((x_center + box_width / 2) * width)
        y2 = round((y_center + box_height / 2) * height)

        # Clip coordinates to the image boundaries for display.
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(0, min(width - 1, x2))
        y2 = max(0, min(height - 1, y2))

        if x2 <= x1 or y2 <= y1:
            print(
                f"WARNING: Invalid box in "
                f"{label_path.name}, line {line_number}"
            )
            continue

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        label = f"class {int(class_value)}"

        # Keep the label text inside the image.
        text_y = max(20, y1 - 6)

        cv2.putText(
            image,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        box_count += 1

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_rgb, box_count


def main() -> None:
    """Generate a visual preview of YOLO annotations."""
    if SPLIT not in {"train", "val"}:
        raise SystemExit(
            "Invalid SPLIT. Choose either 'train' or 'val'."
        )

    if MAX_IMAGES < 0:
        raise SystemExit("MAX_IMAGES must be zero or a positive integer.")

    if not IMAGE_DIR.is_dir():
        raise SystemExit(
            f"Image directory does not exist: {IMAGE_DIR}"
        )

    if not LABEL_DIR.is_dir():
        raise SystemExit(
            f"Label directory does not exist: {LABEL_DIR}"
        )

    image_paths = sorted(
        path
        for path in IMAGE_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_paths:
        raise SystemExit(f"No images found in {IMAGE_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    samples = []
    missing_labels = 0

    for image_path in image_paths:
        label_path = LABEL_DIR / f"{image_path.stem}.txt"

        if not label_path.is_file():
            print(f"WARNING: Missing label for {image_path.name}")
            missing_labels += 1
            continue

        result = draw_yolo_boxes(image_path, label_path)

        if result is None:
            continue

        image_rgb, box_count = result
        samples.append((image_path.name, image_rgb, box_count))

        # Zero means process all matching images.
        if MAX_IMAGES > 0 and len(samples) >= MAX_IMAGES:
            break

    if not samples:
        raise SystemExit(
            "No matching image-label pairs could be visualized."
        )

    rows = math.ceil(len(samples) / COLUMNS)

    figure, axes = plt.subplots(
        rows,
        COLUMNS,
        figsize=(
            FIGURE_WIDTH,
            FIGURE_HEIGHT_PER_ROW * rows,
        ),
        squeeze=False,
    )

    for axis in axes.flat:
        axis.axis("off")

    for axis, (name, image_rgb, box_count) in zip(
        axes.flat,
        samples,
    ):
        axis.imshow(image_rgb)
        axis.set_title(
            f"{name}\nBounding boxes: {box_count}",
            fontsize=8,
        )
        axis.axis("off")

    figure.suptitle(
        f"YOLO Late-Blight Annotation Preview — {SPLIT.upper()}",
        fontsize=16,
    )

    figure.tight_layout(rect=(0, 0, 1, 0.98))

    output_path = OUTPUT_DIR / "yolo_annotation_preview.png"

    figure.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(figure)

    total_boxes = sum(box_count for _, _, box_count in samples)

    print(f"Dataset split: {SPLIT}")
    print(f"Images found: {len(image_paths)}")
    print(f"Images visualized: {len(samples)}")
    print(f"Missing labels: {missing_labels}")
    print(f"Total boxes visualized: {total_boxes}")
    print(f"Preview saved to: {output_path}")


if __name__ == "__main__":
    main()
