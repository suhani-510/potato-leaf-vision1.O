
from __future__ import annotations

import math
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGE_DIR = PROJECT_ROOT / "data" / "yolo" / "images" / "train"
LABEL_DIR = PROJECT_ROOT / "data" / "yolo" / "labels" / "train"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "opencv_lesion_proposals"

MAX_IMAGES = 6
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_yolo_boxes(label_path: Path, width: int, height: int):
    """Convert normalized YOLO boxes into pixel coordinates."""
    boxes = []

    if not label_path.is_file():
        return boxes

    with label_path.open("r", encoding="utf-8") as file:
        for line in file:
            values = line.split()

            if len(values) != 5:
                continue

            try:
                class_id, xc, yc, bw, bh = map(float, values)
            except ValueError:
                continue

            x1 = max(0, int((xc - bw / 2) * width))
            y1 = max(0, int((yc - bh / 2) * height))
            x2 = min(width - 1, int((xc + bw / 2) * width))
            y2 = min(height - 1, int((yc + bh / 2) * height))

            if x2 > x1 and y2 > y1:
                boxes.append((int(class_id), x1, y1, x2, y2))

    return boxes


def find_leaf_mask(image: np.ndarray) -> np.ndarray:
    """
    Estimate the leaf area using color saturation and hue.

    This is a heuristic. Brown and yellow leaf regions may be missed,
    and shadows or other colored objects may be included.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Green/yellow-green pixels with some saturation.
    leaf_color = cv2.inRange(
        hsv,
        np.array([15, 25, 20], dtype=np.uint8),
        np.array([105, 255, 255], dtype=np.uint8),
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (9, 9)
    )

    leaf_color = cv2.morphologyEx(
        leaf_color, cv2.MORPH_CLOSE, kernel, iterations=2
    )
    leaf_color = cv2.morphologyEx(
        leaf_color, cv2.MORPH_OPEN, kernel, iterations=1
    )

    contours, _ = cv2.findContours(
        leaf_color, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return np.zeros(image.shape[:2], dtype=np.uint8)

    # Keep the largest color region as an initial leaf estimate.
    largest = max(contours, key=cv2.contourArea)

    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [largest], -1, 255, thickness=cv2.FILLED)

    # Slightly expand the mask to include nearby brown leaf tissue.
    expansion_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (11, 11)
    )
    mask = cv2.dilate(mask, expansion_kernel, iterations=1)

    return mask


def find_lesion_candidates(
    image: np.ndarray, leaf_mask: np.ndarray
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    """
    Propose brown/dark regions inside the estimated leaf area.

    These are unverified candidate regions, not confirmed lesions.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    brightness = hsv[:, :, 2]
    lab_a = lab[:, :, 1]

    # Brown/reddish colors and relatively dark, red-shifted LAB pixels.
    brown_hsv = (
        (hue <= 25)
        & (saturation >= 35)
        & (brightness >= 20)
        & (brightness <= 220)
    )

    reddish_lab = (
        (lab_a >= 132)
        & (lab[:, :, 0] <= 210)
    )

    # Dark pixels can help identify necrotic regions, but also produce
    # false positives around shadows and leaf veins.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dark_pixels = gray < 95

    leaf_pixels = leaf_mask > 0

    candidates = (
        (brown_hsv | reddish_lab | dark_pixels)
        & leaf_pixels
    ).astype(np.uint8) * 255

    # Remove isolated noise and connect nearby candidate pixels.
    small_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (3, 3)
    )
    large_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (7, 7)
    )

    candidates = cv2.morphologyEx(
        candidates, cv2.MORPH_OPEN, small_kernel
    )
    candidates = cv2.morphologyEx(
        candidates, cv2.MORPH_CLOSE, large_kernel
    )

    contours, _ = cv2.findContours(
        candidates, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    height, width = candidates.shape
    image_area = height * width

    boxes = []

    for contour in contours:
        area = cv2.contourArea(contour)

        # Ignore tiny noise and very large regions.
        if area < max(20, image_area * 0.00015):
            continue
        if area > image_area * 0.20:
            continue

        x, y, box_width, box_height = cv2.boundingRect(contour)

        if box_width < 5 or box_height < 5:
            continue

        boxes.append(
            (x, y, x + box_width, y + box_height)
        )

    boxes.sort(key=lambda box: (box[1], box[0]))

    return candidates, boxes


def draw_boxes(
    image: np.ndarray,
    boxes: list[tuple[int, ...]],
    yolo: bool = False,
) -> np.ndarray:
    """Draw either ground-truth YOLO boxes or CV candidate boxes."""
    output = image.copy()

    for box in boxes:
        if yolo:
            class_id, x1, y1, x2, y2 = box
            color = (0, 255, 0)
            label = f"YOLO class {class_id}"
        else:
            x1, y1, x2, y2 = box
            color = (0, 165, 255)
            label = "CV candidate"

        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

        cv2.putText(
            output,
            label,
            (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )

    return output


def main() -> None:
    if not IMAGE_DIR.is_dir() or not LABEL_DIR.is_dir():
        raise SystemExit(
            "Could not find the YOLO training images or labels. "
            "Check the data/yolo directory structure."
        )

    image_paths = sorted(
        path
        for path in IMAGE_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and (LABEL_DIR / f"{path.stem}.txt").is_file()
    )[:MAX_IMAGES]

    if not image_paths:
        raise SystemExit("No matching image-label pairs were found.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = len(image_paths)
    figure, axes = plt.subplots(
        rows, 3, figsize=(15, 4.5 * rows), squeeze=False
    )

    for row, image_path in enumerate(image_paths):
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Skipping unreadable image: {image_path.name}")
            for axis in axes[row]:
                axis.axis("off")
            continue

        height, width = image.shape[:2]
        label_path = LABEL_DIR / f"{image_path.stem}.txt"

        yolo_boxes = read_yolo_boxes(label_path, width, height)
        leaf_mask = find_leaf_mask(image)
        candidate_mask, candidate_boxes = find_lesion_candidates(
            image, leaf_mask
        )

        original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        yolo_rgb = cv2.cvtColor(
            draw_boxes(image, yolo_boxes, yolo=True),
            cv2.COLOR_BGR2RGB,
        )

        candidate_rgb = cv2.cvtColor(
            draw_boxes(image, candidate_boxes),
            cv2.COLOR_BGR2RGB,
        )

        axes[row, 0].imshow(original_rgb)
        axes[row, 0].set_title("Original image")

        axes[row, 1].imshow(yolo_rgb)
        axes[row, 1].set_title(
            f"Existing YOLO boxes: {len(yolo_boxes)}"
        )

        axes[row, 2].imshow(candidate_rgb)
        axes[row, 2].set_title(
            f"OpenCV proposals: {len(candidate_boxes)}"
        )

        for axis in axes[row]:
            axis.axis("off")

        # Save intermediate outputs for individual inspection.
        stem = image_path.stem
        cv2.imwrite(
            str(OUTPUT_DIR / f"{stem}_leaf_mask.png"), leaf_mask
        )
        cv2.imwrite(
            str(OUTPUT_DIR / f"{stem}_candidate_mask.png"),
            candidate_mask,
        )
        cv2.imwrite(
            str(OUTPUT_DIR / f"{stem}_opencv_boxes.jpg"),
            draw_boxes(image, candidate_boxes),
        )

        print(
            f"{image_path.name}: "
            f"{len(yolo_boxes)} YOLO boxes, "
            f"{len(candidate_boxes)} CV proposals"
        )

    figure.suptitle(
        "Potato Leaf: YOLO Annotations vs OpenCV Proposals",
        fontsize=15,
    )
    figure.tight_layout()

    preview_path = OUTPUT_DIR / "opencv_vs_yolo_preview.png"
    figure.savefig(preview_path, dpi=140, bbox_inches="tight")
    plt.close(figure)

    print("\nFinished. Original data was not modified.")
    print(f"Preview: {preview_path}")


if __name__ == "__main__":
    main()
