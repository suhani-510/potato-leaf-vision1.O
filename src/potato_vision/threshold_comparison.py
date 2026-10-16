
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

OUTPUT_DIR = ROOT / "outputs" / "threshold_comparison"

IMAGE_SIZE = 640
THRESHOLDS = [0.25, 0.10]


def ground_truth_presence(label_path):
    """
    True: at least one valid labelled box exists.
    False: the label file exists but contains no valid boxes.
    None: the label file is missing.
    """
    if not label_path.is_file():
        return None

    for line in label_path.read_text(encoding="utf-8").splitlines():
        values = line.split()

        if len(values) != 5:
            continue

        try:
            class_id = int(values[0])
            x, y, width, height = map(float, values[1:])

            if (
                class_id >= 0
                and all(0 <= value <= 1 for value in (x, y, width, height))
                and width > 0
                and height > 0
            ):
                return True

        except ValueError:
            continue

    return False


def main():
    if not IMAGE_DIR.is_dir():
        raise FileNotFoundError(f"Images not found: {IMAGE_DIR}")

    if not LABEL_DIR.is_dir():
        raise FileNotFoundError(f"Labels not found: {LABEL_DIR}")

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    images = sorted(
        path for path in IMAGE_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )

    if not images:
        raise RuntimeError("No validation images found.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(MODEL_PATH))

    rows = []

    for image_path in images:
        label_path = LABEL_DIR / f"{image_path.stem}.txt"
        gt_present = ground_truth_presence(label_path)

        for threshold in THRESHOLDS:
            result = model.predict(
                source=str(image_path),
                imgsz=IMAGE_SIZE,
                conf=threshold,
                device="cpu",
                verbose=False,
            )[0]

            boxes = result.boxes
            box_count = len(boxes) if boxes is not None else 0
            predicted_present = box_count > 0

            if gt_present is None:
                status = "unknown_ground_truth"
            elif gt_present and predicted_present:
                status = "true_positive"
            elif gt_present and not predicted_present:
                status = "false_negative"
            elif not gt_present and predicted_present:
                status = "false_positive"
            else:
                status = "true_negative"

            rows.append({
                "image": image_path.name,
                "threshold": threshold,
                "ground_truth_present": gt_present,
                "predicted_boxes": box_count,
                "predicted_present": predicted_present,
                "status": status,
            })

            print(
                f"{image_path.name} | threshold={threshold:.2f} "
                f"| boxes={box_count} | status={status}"
            )

    results_path = OUTPUT_DIR / "threshold_results.csv"

    with results_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    summaries = []

    for threshold in THRESHOLDS:
        subset = [
            row for row in rows
            if row["threshold"] == threshold
        ]

        known = [
            row for row in subset
            if row["ground_truth_present"] is not None
        ]

        tp = sum(row["status"] == "true_positive" for row in known)
        fp = sum(row["status"] == "false_positive" for row in known)
        fn = sum(row["status"] == "false_negative" for row in known)
        tn = sum(row["status"] == "true_negative" for row in known)

        summaries.append({
            "threshold": threshold,
            "validation_images": len(subset),
            "known_ground_truth_images": len(known),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn,
            "image_level_recall": (
                tp / (tp + fn) if tp + fn else None
            ),
            "image_level_precision": (
                tp / (tp + fp) if tp + fp else None
            ),
            "image_level_specificity": (
                tn / (tn + fp) if tn + fp else None
            ),
            "note": (
                "Metrics compare predictions with existing annotations. "
                "Unannotated images are not necessarily healthy."
            ),
        })

    summary_path = OUTPUT_DIR / "threshold_summary.json"

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summaries, file, indent=4)

    print("\n--- THRESHOLD COMPARISON ---")
    print(json.dumps(summaries, indent=4))
    print(f"\nDetailed CSV: {results_path}")
    print(f"Summary JSON: {summary_path}")


if __name__ == "__main__":
    main()
