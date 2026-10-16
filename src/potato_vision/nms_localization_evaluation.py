
from pathlib import Path
import csv

import numpy as np
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]

IMAGE_DIR = ROOT / "data" / "yolo" / "images" / "val"
LABEL_DIR = ROOT / "data" / "yolo" / "labels" / "val"

MODEL_PATH = (
    ROOT / "runs" / "detect" / "potato_late_blight-4"
    / "weights" / "best.pt"
)

OUTPUT_DIR = ROOT / "outputs" / "nms_localization_evaluation"

IMAGE_SIZE = 640
CONFIDENCE = 0.10
NMS_THRESHOLDS = [0.70, 0.50]
MATCH_IOU = 0.50


def read_labels(path):
    boxes = []

    if not path.exists():
        return boxes

    for line in path.read_text(encoding="utf-8").splitlines():
        values = line.split()

        if len(values) != 5:
            continue

        class_id, xc, yc, width, height = map(float, values)

        # Convert normalized YOLO coordinates to center-based boxes.
        boxes.append([
            int(class_id),
            xc - width / 2,
            yc - height / 2,
            xc + width / 2,
            yc + height / 2,
        ])

    return boxes


def box_iou(box_a, box_b):
    # Boxes use normalized xyxy coordinates.
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)

    area_a = max(0.0, box_a[2] - box_a[0]) * max(
        0.0, box_a[3] - box_a[1]
    )
    area_b = max(0.0, box_b[2] - box_b[0]) * max(
        0.0, box_b[3] - box_b[1]
    )

    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def match_boxes(ground_truth, predictions):
    # Match each GT box to at most one prediction, choosing
    # the highest-IoU available match first.
    candidates = []

    for gt_index, gt in enumerate(ground_truth):
        for pred_index, pred in enumerate(predictions):
            if gt[0] != pred[0]:
                continue

            iou = box_iou(gt[1:5], pred[1:5])

            if iou >= MATCH_IOU:
                candidates.append((iou, gt_index, pred_index))

    candidates.sort(reverse=True)

    used_gt = set()
    used_predictions = set()

    for iou, gt_index, pred_index in candidates:
        if gt_index in used_gt or pred_index in used_predictions:
            continue

        used_gt.add(gt_index)
        used_predictions.add(pred_index)

    return len(used_gt), len(ground_truth), len(predictions)


def main():
    for path in (IMAGE_DIR, LABEL_DIR, MODEL_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Required path not found: {path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        p for p in IMAGE_DIR.iterdir()
        if p.is_file()
        and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )

    if not image_paths:
        raise RuntimeError("No validation images found.")

    model = YOLO(str(MODEL_PATH))
    rows = []

    for index, image_path in enumerate(image_paths, start=1):
        print(f"[{index}/{len(image_paths)}] {image_path.name}")

        gt = read_labels(LABEL_DIR / f"{image_path.stem}.txt")

        for nms_iou in NMS_THRESHOLDS:
            result = model.predict(
                source=str(image_path),
                imgsz=IMAGE_SIZE,
                conf=CONFIDENCE,
                iou=nms_iou,
                device="cpu",
                verbose=False,
            )[0]

            predictions = []

            if result.boxes is not None:
                coords = result.boxes.xyxy.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)

                image_height, image_width = result.orig_shape

                for xyxy, class_id in zip(coords, class_ids):
                    x1, y1, x2, y2 = xyxy

                    # Normalize to make IoU comparable with GT labels.
                    predictions.append([
                        int(class_id),
                        x1 / image_width,
                        y1 / image_height,
                        x2 / image_width,
                        y2 / image_height,
                    ])

            matched, gt_count, pred_count = match_boxes(gt, predictions)

            rows.append({
                "image": image_path.name,
                "nms_iou": nms_iou,
                "ground_truth_boxes": gt_count,
                "predicted_boxes": pred_count,
                "matched_boxes_iou50": matched,
                "unmatched_ground_truth": gt_count - matched,
                "unmatched_predictions": pred_count - matched,
            })

    detail_path = OUTPUT_DIR / "nms_localization_details.csv"

    with detail_path.open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = []

    for nms_iou in NMS_THRESHOLDS:
        subset = [r for r in rows if r["nms_iou"] == nms_iou]

        gt_total = sum(r["ground_truth_boxes"] for r in subset)
        pred_total = sum(r["predicted_boxes"] for r in subset)
        matched_total = sum(r["matched_boxes_iou50"] for r in subset)

        summary_rows.append({
            "nms_iou": nms_iou,
            "confidence": CONFIDENCE,
            "validation_images": len(subset),
            "ground_truth_boxes": gt_total,
            "predicted_boxes": pred_total,
            "matched_boxes_iou50": matched_total,
            "unmatched_ground_truth": gt_total - matched_total,
            "unmatched_predictions": pred_total - matched_total,
            "box_recall_iou50": (
                matched_total / gt_total if gt_total else None
            ),
            "note": (
                "Unmatched predictions are not automatically false positives; "
                "unannotated real lesions may exist."
            ),
        })

    summary_path = OUTPUT_DIR / "nms_localization_summary.csv"

    with summary_path.open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=list(summary_rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\nEvaluation complete.")
    print(f"Details: {detail_path}")
    print(f"Summary: {summary_path}")

    for row in summary_rows:
        print(
            f"NMS={row['nms_iou']:.2f}: "
            f"{row['matched_boxes_iou50']}/{row['ground_truth_boxes']} "
            f"GT boxes matched; "
            f"recall={row['box_recall_iou50']:.3f}; "
            f"predictions={row['predicted_boxes']}"
        )


if __name__ == "__main__":
    main()
