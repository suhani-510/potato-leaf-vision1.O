
from pathlib import Path
import csv
import json

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]

IMAGE_DIR = ROOT / "data" / "yolo" / "images" / "val"
LABEL_DIR = ROOT / "data" / "yolo" / "labels" / "val"
MODEL_PATH = (
    ROOT / "runs" / "detect" / "potato_late_blight-4"
    / "weights" / "best.pt"
)
OUTPUT_DIR = ROOT / "outputs" / "localization_evaluation"

IMAGE_SIZE = 640
THRESHOLDS = [0.25, 0.10]
IOU_MATCH_THRESHOLD = 0.50


def read_ground_truth(label_path, image_width, image_height):
    if not label_path.is_file():
        raise FileNotFoundError(f"Missing annotation: {label_path}")

    boxes = []

    for line in label_path.read_text(encoding="utf-8").splitlines():
        values = line.split()

        if len(values) != 5:
            continue

        try:
            class_id = int(values[0])
            x, y, width, height = map(float, values[1:])

            if not (
                class_id >= 0
                and all(0 <= v <= 1 for v in (x, y, width, height))
                and width > 0
                and height > 0
            ):
                continue

            boxes.append([
                (x - width / 2) * image_width,
                (y - height / 2) * image_height,
                (x + width / 2) * image_width,
                (y + height / 2) * image_height,
            ])

        except ValueError:
            continue

    return boxes


def calculate_iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])

    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def greedy_match(ground_truth, predictions):
    """
    Match each ground-truth box to at most one prediction.
    Prefer the highest-IoU available pairs first.
    """
    possible_matches = []

    for gt_index, gt_box in enumerate(ground_truth):
        for pred_index, pred in enumerate(predictions):
            iou = calculate_iou(gt_box, pred["box"])

            if iou >= IOU_MATCH_THRESHOLD:
                possible_matches.append(
                    (iou, gt_index, pred_index)
                )

    possible_matches.sort(reverse=True)

    matched_gt = set()
    matched_predictions = set()
    accepted_ious = []

    for iou, gt_index, pred_index in possible_matches:
        if gt_index in matched_gt or pred_index in matched_predictions:
            continue

        matched_gt.add(gt_index)
        matched_predictions.add(pred_index)
        accepted_ious.append(iou)

    return (
        len(matched_gt),
        len(matched_predictions),
        accepted_ious,
    )


def main():
    for path in (IMAGE_DIR, LABEL_DIR, MODEL_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Required path not found: {path}")

    images = sorted(
        p for p in IMAGE_DIR.iterdir()
        if p.is_file()
        and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )

    if not images:
        raise RuntimeError("No validation images found.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(MODEL_PATH))

    # Predict once at low confidence, then apply both thresholds to
    # the same returned candidate set for a fair comparison.
    image_results = model.predict(
        source=[str(p) for p in images],
        imgsz=IMAGE_SIZE,
        conf=0.001,
        device="cpu",
        verbose=False,
    )

    detail_rows = []
    summary = []

    for image_path, result in zip(images, image_results):
        image_height, image_width = result.orig_shape

        gt_boxes = read_ground_truth(
            LABEL_DIR / f"{image_path.stem}.txt",
            image_width,
            image_height,
        )

        raw_predictions = []

        if result.boxes is not None:
            for box in result.boxes:
                raw_predictions.append({
                    "confidence": float(box.conf[0]),
                    "box": [float(v) for v in box.xyxy[0].tolist()],
                })

        for threshold in THRESHOLDS:
            predictions = [
                pred for pred in raw_predictions
                if pred["confidence"] >= threshold
            ]

            matched_count, matched_prediction_count, matched_ious = (
                greedy_match(gt_boxes, predictions)
            )

            best_ious = [
                max(
                    (calculate_iou(gt, pred["box"]) for pred in predictions),
                    default=0.0,
                )
                for gt in gt_boxes
            ]

            detail_rows.append({
                "image": image_path.name,
                "threshold": threshold,
                "ground_truth_boxes": len(gt_boxes),
                "predicted_boxes": len(predictions),
                "matched_ground_truth_boxes_iou50": matched_count,
                "unmatched_ground_truth_boxes": (
                    len(gt_boxes) - matched_count
                ),
                "unmatched_predictions": (
                    len(predictions) - matched_prediction_count
                ),
                "mean_best_iou_per_ground_truth": (
                    round(sum(best_ious) / len(best_ious), 5)
                    if best_ious else ""
                ),
                "mean_iou_of_matched_pairs": (
                    round(sum(matched_ious) / len(matched_ious), 5)
                    if matched_ious else ""
                ),
            })

    csv_path = OUTPUT_DIR / "localization_details.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=detail_rows[0].keys())
        writer.writeheader()
        writer.writerows(detail_rows)

    for threshold in THRESHOLDS:
        rows = [r for r in detail_rows if r["threshold"] == threshold]

        total_gt = sum(r["ground_truth_boxes"] for r in rows)
        total_pred = sum(r["predicted_boxes"] for r in rows)
        total_matched = sum(
            r["matched_ground_truth_boxes_iou50"] for r in rows
        )
        total_unmatched_pred = sum(r["unmatched_predictions"] for r in rows)

        recall_iou50 = total_matched / total_gt if total_gt else None

        summary.append({
            "threshold": threshold,
            "validation_images": len(rows),
            "total_ground_truth_boxes": total_gt,
            "total_predicted_boxes": total_pred,
            "matched_ground_truth_boxes_iou50": total_matched,
            "unmatched_ground_truth_boxes": total_gt - total_matched,
            "unmatched_predictions": total_unmatched_pred,
            "box_recall_iou50": recall_iou50,
            "note": (
                "Unmatched predictions are not automatically false positives; "
                "some may represent unannotated real lesions. IoU matching "
                "uses a threshold of 0.50."
            ),
        })

    summary_path = OUTPUT_DIR / "localization_summary.json"

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print("\n--- BOX-LEVEL LOCALIZATION SUMMARY ---")
    print(json.dumps(summary, indent=4))
    print(f"\nDetailed results: {csv_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
