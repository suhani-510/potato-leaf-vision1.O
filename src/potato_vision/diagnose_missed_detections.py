
from pathlib import Path
import csv

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]

IMAGE_DIR = ROOT / "data" / "yolo" / "images" / "val"
LABEL_DIR = ROOT / "data" / "yolo" / "labels" / "val"
MODEL_PATH = (
    ROOT / "runs" / "detect" / "potato_late_blight-4"
    / "weights" / "best.pt"
)
OUTPUT_DIR = ROOT / "outputs" / "missed_detection_diagnosis"

IMAGE_PREFIXES = [
    "e95f2954-1b68-490d-9684-90b3080676b1",
    "ea356b9c-9090-4f0d-9083-d1f63c77372d",
    "ebd47dca-590b-43c4-812d-32f95b776690",
]

IMAGE_SIZE = 640
LOW_CONFIDENCE = 0.001


def read_ground_truth(label_path, width, height):
    boxes = []

    if not label_path.exists():
        raise FileNotFoundError(
            f"Ground-truth label not found: {label_path}"
        )

    for line in label_path.read_text(encoding="utf-8").splitlines():
        values = line.split()

        if len(values) != 5:
            continue

        class_id = int(values[0])
        x, y, w, h = map(float, values[1:])

        # Convert normalized YOLO coordinates to pixel coordinates.
        x1 = (x - w / 2) * width
        y1 = (y - h / 2) * height
        x2 = (x + w / 2) * width
        y2 = (y + h / 2) * height

        boxes.append({
            "class_id": class_id,
            "xyxy": [x1, y1, x2, y2],
        })

    return boxes


def calculate_iou(box_a, box_b):
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


def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(MODEL_PATH))

    summary_rows = []
    prediction_rows = []

    for prefix in IMAGE_PREFIXES:
        matches = list(IMAGE_DIR.glob(f"{prefix}*"))

        if len(matches) != 1:
            raise RuntimeError(
                f"Expected one image for {prefix}, found {len(matches)}"
            )

        image_path = matches[0]

        # Read image dimensions from the model's result.
        result = model.predict(
            source=str(image_path),
            imgsz=IMAGE_SIZE,
            conf=LOW_CONFIDENCE,
            device="cpu",
            verbose=False,
        )[0]

        image_height, image_width = result.orig_shape

        ground_truth = read_ground_truth(
            LABEL_DIR / f"{image_path.stem}.txt",
            image_width,
            image_height,
        )

        predictions = []

        if result.boxes is not None:
            for box in result.boxes:
                prediction = {
                    "confidence": float(box.conf[0]),
                    "xyxy": [float(v) for v in box.xyxy[0].tolist()],
                }
                predictions.append(prediction)

        print(f"\nImage: {image_path.name}")
        print(f"Ground-truth boxes: {len(ground_truth)}")
        print(f"Candidate predictions: {len(predictions)}")

        for index, gt in enumerate(ground_truth, start=1):
            overlaps = [
                (calculate_iou(gt["xyxy"], pred["xyxy"]), pred)
                for pred in predictions
            ]

            if overlaps:
                best_iou, best_match = max(
                    overlaps, key=lambda item: item[0]
                )
                best_iou_conf = best_match["confidence"]

                best_conf_prediction = max(
                    predictions, key=lambda pred: pred["confidence"]
                )
                highest_conf = best_conf_prediction["confidence"]
                highest_conf_iou = calculate_iou(
                    gt["xyxy"], best_conf_prediction["xyxy"]
                )
            else:
                best_iou = 0.0
                best_iou_conf = None
                highest_conf = None
                highest_conf_iou = 0.0

            summary_rows.append({
                "image": image_path.name,
                "ground_truth_index": index,
                "candidate_count": len(predictions),
                "best_iou_to_ground_truth": round(best_iou, 5),
                "confidence_of_best_iou_candidate": (
                    round(best_iou_conf, 5)
                    if best_iou_conf is not None else ""
                ),
                "highest_candidate_confidence": (
                    round(highest_conf, 5)
                    if highest_conf is not None else ""
                ),
                "iou_of_highest_confidence_candidate": round(
                    highest_conf_iou, 5
                ),
            })

            print(
                f"  GT box {index}: best IoU={best_iou:.3f}, "
                f"confidence of that candidate={best_iou_conf}"
            )

        for index, pred in enumerate(
            sorted(
                predictions,
                key=lambda item: item["confidence"],
                reverse=True,
            ),
            start=1,
        ):
            prediction_rows.append({
                "image": image_path.name,
                "prediction_rank": index,
                "confidence": round(pred["confidence"], 6),
                "x1": round(pred["xyxy"][0], 2),
                "y1": round(pred["xyxy"][1], 2),
                "x2": round(pred["xyxy"][2], 2),
                "y2": round(pred["xyxy"][3], 2),
            })

        # Save a standard prediction image at the low threshold.
        result.save(
            filename=str(
                OUTPUT_DIR / f"{image_path.stem}_candidates.jpg"
            )
        )

    summary_path = OUTPUT_DIR / "ground_truth_overlap_summary.csv"
    predictions_path = OUTPUT_DIR / "all_candidate_predictions.csv"

    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)

    prediction_fields = [
        "image", "prediction_rank", "confidence",
        "x1", "y1", "x2", "y2",
    ]

    with predictions_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=prediction_fields)
        writer.writeheader()
        writer.writerows(prediction_rows)

    print("\nDiagnostic complete.")
    print(f"Ground-truth comparison: {summary_path}")
    print(f"All candidate boxes: {predictions_path}")
    print(f"Annotated images: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
