
from pathlib import Path
import csv
import json

from ultralytics import YOLO


# --------------------------------------------------
# 1. Project paths and model settings
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGE_DIR = PROJECT_ROOT / "data" / "yolo" / "images" / "val"
LABEL_DIR = PROJECT_ROOT / "data" / "yolo" / "labels" / "val"

MODEL_PATH = (
    PROJECT_ROOT
    / "runs"
    / "detect"
    / "potato_late_blight-4"
    / "weights"
    / "best.pt"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "image_presence_evaluation"

CONFIDENCE_THRESHOLD = 0.25
IMAGE_SIZE = 640


# --------------------------------------------------
# 2. Read ground-truth labels
# --------------------------------------------------

def read_ground_truth(label_path):
    """
    Returns:
        True  -> at least one valid annotated lesion
        False -> valid label file with no annotated lesions
        None  -> label file is missing or unreadable
    """

    if not label_path.is_file():
        return None

    try:
        with label_path.open("r", encoding="utf-8") as file:
            lines = [
                line.strip()
                for line in file
                if line.strip()
            ]

        for line in lines:
            values = line.split()

            if len(values) != 5:
                continue

            try:
                class_id = int(values[0])
                coordinates = [float(value) for value in values[1:]]

                if (
                    class_id >= 0
                    and all(0 <= value <= 1 for value in coordinates)
                    and coordinates[2] > 0
                    and coordinates[3] > 0
                ):
                    return True

            except ValueError:
                continue

        return False

    except OSError:
        return None


# --------------------------------------------------
# 3. Run YOLO and evaluate image-level presence
# --------------------------------------------------

def main():
    if not IMAGE_DIR.is_dir():
        raise FileNotFoundError(
            f"Validation image directory not found: {IMAGE_DIR}"
        )

    if not LABEL_DIR.is_dir():
        raise FileNotFoundError(
            f"Validation label directory not found: {LABEL_DIR}"
        )

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"YOLO checkpoint not found: {MODEL_PATH}"
        )

    image_paths = sorted(
        path
        for path in IMAGE_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )

    if not image_paths:
        raise RuntimeError("No validation images were found.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(MODEL_PATH))

    predictions = model.predict(
        source=[str(path) for path in image_paths],
        conf=CONFIDENCE_THRESHOLD,
        imgsz=IMAGE_SIZE,
        device="cpu",
        verbose=False,
    )

    rows = []

    for image_path, result in zip(image_paths, predictions):
        label_path = LABEL_DIR / f"{image_path.stem}.txt"

        ground_truth = read_ground_truth(label_path)

        predicted_boxes = (
            len(result.boxes)
            if result.boxes is not None
            else 0
        )

        predicted_presence = predicted_boxes > 0

        if ground_truth is None:
            status = "unknown_ground_truth"
        elif ground_truth and predicted_presence:
            status = "true_positive"
        elif ground_truth and not predicted_presence:
            status = "false_negative"
        elif not ground_truth and predicted_presence:
            status = "false_positive"
        else:
            status = "true_negative"

        rows.append({
            "image": image_path.name,
            "ground_truth_lesion_present": ground_truth,
            "predicted_boxes": predicted_boxes,
            "predicted_lesion_present": predicted_presence,
            "status": status,
        })

        print(
            f"{image_path.name}: "
            f"GT={ground_truth}, "
            f"YOLO boxes={predicted_boxes}, "
            f"status={status}"
        )

    # Save per-image results.
    csv_path = OUTPUT_DIR / "image_presence_results.csv"

    with csv_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image",
                "ground_truth_lesion_present",
                "predicted_boxes",
                "predicted_lesion_present",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    # Compute metrics only where ground truth is available.
    known_rows = [
        row for row in rows
        if row["ground_truth_lesion_present"] is not None
    ]

    tp = sum(row["status"] == "true_positive" for row in known_rows)
    fp = sum(row["status"] == "false_positive" for row in known_rows)
    fn = sum(row["status"] == "false_negative" for row in known_rows)
    tn = sum(row["status"] == "true_negative" for row in known_rows)

    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    accuracy = (
        (tp + tn) / (tp + tn + fp + fn)
        if tp + tn + fp + fn
        else None
    )

    summary = {
        "validation_images": len(image_paths),
        "images_with_known_ground_truth": len(known_rows),
        "images_with_unknown_ground_truth": len(rows) - len(known_rows),
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "image_level_precision": precision,
        "image_level_recall": recall,
        "image_level_specificity": specificity,
        "image_level_accuracy": accuracy,
        "note": (
            "Ground truth is based on the existing YOLO annotations. "
            "An empty annotation is treated as no annotated lesion, "
            "not as proof that the leaf is biologically healthy."
        ),
    }

    summary_path = OUTPUT_DIR / "image_presence_summary.json"

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print("\n--- IMAGE-LEVEL EVALUATION ---")
    print(json.dumps(summary, indent=4))
    print(f"\nPer-image results: {csv_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
