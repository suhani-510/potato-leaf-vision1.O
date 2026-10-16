
from __future__ import annotations

import csv
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from ultralytics import YOLO

from src.potato_vision.inference_gallery import results_to_detections


PROJECT_ROOT = Path(__file__).resolve().parents[2]

WEIGHTS_PATH = (
    PROJECT_ROOT
    / "runs"
    / "detect"
    / "potato_late_blight-4"
    / "weights"
    / "best.pt"
)

IMAGE_DIR = PROJECT_ROOT / "data" / "yolo" / "images" / "val"
LABEL_DIR = PROJECT_ROOT / "data" / "yolo" / "labels" / "val"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "yolo_comparison"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

IMAGE_SIZE = 640
CONFIDENCE = 0.25
NMS_IOU = 0.45

# A prediction is counted as matched if its IoU with an
# unmatched ground-truth box is at least this value.
MATCH_IOU = 0.50

# Display every validation image, not just images with detections.
IMAGES_PER_PAGE = 6
COLUMNS = 2


def read_yolo_labels(label_path: Path, width: int, height: int):
    """Read normalized YOLO labels and return pixel-coordinate boxes."""
    boxes = []

    if not label_path.exists():
        return boxes

    for line_number, line in enumerate(
        label_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue

        values = line.split()

        if len(values) != 5:
            print(f"WARNING: Invalid label format: {label_path.name}:{line_number}")
            continue

        try:
            class_id, xc, yc, bw, bh = map(float, values)
        except ValueError:
            print(f"WARNING: Invalid numbers: {label_path.name}:{line_number}")
            continue

        x1 = max(0, min(width - 1, round((xc - bw / 2) * width)))
        y1 = max(0, min(height - 1, round((yc - bh / 2) * height)))
        x2 = max(0, min(width - 1, round((xc + bw / 2) * width)))
        y2 = max(0, min(height - 1, round((yc + bh / 2) * height)))

        if x2 <= x1 or y2 <= y1:
            print(f"WARNING: Invalid box: {label_path.name}:{line_number}")
            continue

        boxes.append(
            {
                "class_id": int(class_id),
                "xyxy": (x1, y1, x2, y2),
            }
        )

    return boxes


def box_iou(box_a, box_b):
    """Calculate intersection over union for two pixel-coordinate boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def match_boxes(ground_truth, predictions):
    """
    Greedily match boxes using IoU >= MATCH_IOU.

    Each ground-truth box and each prediction can be matched at most once.
    This is an inspection summary, not a replacement for official evaluation.
    """
    possible_matches = []

    for gt_index, gt in enumerate(ground_truth):
        for pred_index, pred in enumerate(predictions):
            if gt["class_id"] != pred["class_id"]:
                continue

            overlap = box_iou(gt["xyxy"], pred["xyxy"])

            if overlap >= MATCH_IOU:
                possible_matches.append((overlap, gt_index, pred_index))

    possible_matches.sort(reverse=True)

    matched_gt = set()
    matched_predictions = set()

    for overlap, gt_index, pred_index in possible_matches:
        if gt_index in matched_gt or pred_index in matched_predictions:
            continue

        matched_gt.add(gt_index)
        matched_predictions.add(pred_index)

    return (
        len(matched_gt),
        len(ground_truth) - len(matched_gt),
        len(predictions) - len(matched_predictions),
    )


def draw_boxes(image, boxes, color, label_prefix, show_confidence=False):
    """Draw boxes on a copy of an RGB image."""
    canvas = image.copy()

    for box in boxes:
        x1, y1, x2, y2 = map(int, box["xyxy"])

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

        if show_confidence:
            label = f'{box["class_name"]} {box["confidence"]:.2f}'
        else:
            label = f'{label_prefix} {box["class_id"]}'

        text_y = max(18, y1 - 5)

        cv2.putText(
            canvas,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            2,
            cv2.LINE_AA,
        )

    return canvas


def main():
    required_paths = {
        "Model checkpoint": WEIGHTS_PATH,
        "Validation images": IMAGE_DIR,
        "Validation labels": LABEL_DIR,
    }

    for description, path in required_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{description} not found:\n{path}")

    image_paths = sorted(
        path
        for path in IMAGE_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_paths:
        raise RuntimeError(f"No validation images found in {IMAGE_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Validation images: {len(image_paths)}")
    print(f"Loading model: {WEIGHTS_PATH}")

    model = YOLO(str(WEIGHTS_PATH))
    summary_rows = []
    page_panels = []

    # Inference runs over every image, including images with no detections.
    for index, image_path in enumerate(image_paths, start=1):
        image_bgr = cv2.imread(str(image_path))

        if image_bgr is None:
            print(f"WARNING: Could not read {image_path.name}; skipping.")
            continue

        height, width = image_bgr.shape[:2]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        label_path = LABEL_DIR / f"{image_path.stem}.txt"

        ground_truth = read_yolo_labels(label_path, width, height)

        result = model.predict(
            source=str(image_path),
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE,
            iou=NMS_IOU,
            device="cpu",
            verbose=False,
        )[0]

        detections = results_to_detections(result, ("late_blight",))

        predictions = [
            {
                "class_id": detection.class_id,
                "xyxy": tuple(round(value) for value in detection.xyxy),
                "class_name": detection.class_name,
                "confidence": detection.confidence,
            }
            for detection in detections
        ]

        matched, missed, unmatched = match_boxes(ground_truth, predictions)

        summary_rows.append(
            {
                "image": image_path.name,
                "ground_truth_boxes": len(ground_truth),
                "predicted_boxes": len(predictions),
                "matched_boxes": matched,
                "missed_ground_truth_boxes": missed,
                "unmatched_predictions": unmatched,
                "label_file_exists": label_path.exists(),
                "match_iou_threshold": MATCH_IOU,
            }
        )

        gt_image = draw_boxes(
            image_rgb,
            ground_truth,
            (0, 200, 0),
            "GT",
        )

        pred_image = draw_boxes(
            image_rgb,
            predictions,
            (255, 140, 0),
            "Pred",
            show_confidence=True,
        )

        page_panels.append(
            (
                image_path.name,
                gt_image,
                pred_image,
                len(ground_truth),
                len(predictions),
                matched,
                missed,
                unmatched,
            )
        )

        print(f"[{index}/{len(image_paths)}] {image_path.name}")

    if not summary_rows:
        raise RuntimeError("No validation images could be processed.")

    # Save per-image summary.
    csv_path = OUTPUT_DIR / "validation_comparison_summary.csv"

    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    # Save all images in paginated side-by-side galleries.
    for page_start in range(0, len(page_panels), IMAGES_PER_PAGE):
        page = page_panels[page_start:page_start + IMAGES_PER_PAGE]
        rows = len(page)

        figure, axes = plt.subplots(
            rows,
            COLUMNS,
            figsize=(14, 4.5 * rows),
            squeeze=False,
        )

        for row_index, panel in enumerate(page):
            (
                name,
                gt_image,
                pred_image,
                gt_count,
                pred_count,
                matched,
                missed,
                unmatched,
            ) = panel

            axes[row_index, 0].imshow(gt_image)
            axes[row_index, 0].set_title(
                f"Ground truth — {name}\nBoxes: {gt_count}",
                fontsize=9,
            )

            axes[row_index, 1].imshow(pred_image)
            axes[row_index, 1].set_title(
                f"Predictions — {name}\n"
                f"Boxes: {pred_count} | Matched: {matched} | "
                f"Missed: {missed} | Unmatched: {unmatched}",
                fontsize=9,
            )

            axes[row_index, 0].axis("off")
            axes[row_index, 1].axis("off")

        figure.suptitle(
            "Potato Late Blight — Ground Truth vs YOLOv8 Predictions",
            fontsize=15,
            fontweight="bold",
        )

        figure.legend(
            handles=[
                Patch(color="#00C800", label="Ground-truth boxes"),
                Patch(color="#FF8C00", label="Predicted boxes"),
            ],
            loc="lower center",
            ncol=2,
        )

        figure.tight_layout(rect=(0, 0.035, 1, 0.97))

        page_number = page_start // IMAGES_PER_PAGE + 1
        image_output = OUTPUT_DIR / f"comparison_page_{page_number:02d}.png"

        figure.savefig(image_output, dpi=140, bbox_inches="tight")
        plt.close(figure)

        print(f"Saved comparison: {image_output}")

    total_gt = sum(row["ground_truth_boxes"] for row in summary_rows)
    total_pred = sum(row["predicted_boxes"] for row in summary_rows)
    total_matched = sum(row["matched_boxes"] for row in summary_rows)
    total_missed = sum(row["missed_ground_truth_boxes"] for row in summary_rows)
    total_unmatched = sum(row["unmatched_predictions"] for row in summary_rows)

    print("\nComparison completed.")
    print(f"Images processed: {len(summary_rows)}")
    print(f"Ground-truth boxes: {total_gt}")
    print(f"Predicted boxes: {total_pred}")
    print(f"Greedy IoU matches: {total_matched}")
    print(f"Unmatched ground-truth boxes: {total_missed}")
    print(f"Unmatched predictions: {total_unmatched}")
    print(f"Matching IoU threshold: {MATCH_IOU}")
    print(f"CSV summary: {csv_path}")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
