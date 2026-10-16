
from __future__ import annotations

from pathlib import Path
import math

import cv2
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGE_DIR = PROJECT_ROOT / "data" / "yolo" / "images" / "train"
LABEL_DIR = PROJECT_ROOT / "data" / "yolo" / "labels" / "train"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "yolo_label_preview"

MAX_IMAGES = 12


def draw_yolo_boxes(image_path: Path, label_path: Path):
    """Draw normalized YOLO bounding boxes on an image."""
    image = cv2.imread(str(image_path))

    if image is None:
        print(f"WARNING: Could not read image: {image_path.name}")
        return None

    height, width = image.shape[:2]

    with label_path.open("r", encoding="utf-8") as file:
        lines = [line.strip() for line in file if line.strip()]

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
            class_id, x_center, y_center, box_width, box_height = (
                float(value) for value in values
            )
        except ValueError:
            print(f"WARNING: Invalid numbers in {label_path.name}")
            continue

        # YOLO coordinates are normalized to the image dimensions.
        x1 = round((x_center - box_width / 2) * width)
        y1 = round((y_center - box_height / 2) * height)
        x2 = round((x_center + box_width / 2) * width)
        y2 = round((y_center + box_height / 2) * height)

        # Keep the box within the image.
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(0, min(width - 1, x2))
        y2 = max(0, min(height - 1, y2))

        if x2 <= x1 or y2 <= y1:
            print(f"WARNING: Invalid box in {label_path.name}")
            continue

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label = f"class {int(class_id)}"
        cv2.putText(
            image,
            label,
            (x1, max(20, y1 - 6)),
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
    if not IMAGE_DIR.is_dir() or not LABEL_DIR.is_dir():
        raise SystemExit(
            "Image or label directory is missing. Check the data/yolo folders."
        )

    image_paths = sorted(
        path
        for path in IMAGE_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )

    if not image_paths:
        raise SystemExit(f"No images found in {IMAGE_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    samples = []

    for image_path in image_paths:
        label_path = LABEL_DIR / f"{image_path.stem}.txt"

        if not label_path.is_file():
            print(f"Skipping image without label: {image_path.name}")
            continue

        result = draw_yolo_boxes(image_path, label_path)

        if result is None:
            continue

        image_rgb, box_count = result
        samples.append((image_path.name, image_rgb, box_count))

        if len(samples) >= MAX_IMAGES:
            break

    if not samples:
        raise SystemExit("No matching image-label pairs could be visualized.")

    columns = 3
    rows = math.ceil(len(samples) / columns)

    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(15, 5 * rows),
        squeeze=False,
    )

    for axis in axes.flat:
        axis.axis("off")

    for axis, (name, image_rgb, box_count) in zip(axes.flat, samples):
        axis.imshow(image_rgb)
        axis.set_title(f"{name}\nBounding boxes: {box_count}", fontsize=9)
        axis.axis("off")

    figure.suptitle("YOLO Late-Blight Annotation Preview", fontsize=16)
    figure.tight_layout()

    output_path = OUTPUT_DIR / "yolo_annotation_preview.png"
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)

    print(f"Visualized images: {len(samples)}")
    print(f"Preview saved to: {output_path}")


if __name__ == "__main__":
    main()
