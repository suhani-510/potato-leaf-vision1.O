
from pathlib import Path
import csv

import cv2
import numpy as np
from ultralytics import YOLO


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

IMAGE_DIR = ROOT / "data" / "yolo" / "images" / "val"
LABEL_DIR = ROOT / "data" / "yolo" / "labels" / "val"

MODEL_PATH = (
    ROOT / "runs" / "detect" / "potato_late_blight-4"
    / "weights" / "best.pt"
)

OUTPUT_DIR = ROOT / "outputs" / "nms_comparison"

IMAGE_SIZE = 640
CONFIDENCE = 0.10

NMS_THRESHOLDS = [0.70, 0.50]


def read_ground_truth(label_path, width, height):
    boxes = []

    if not label_path.exists():
        return boxes

    with label_path.open("r", encoding="utf-8") as file:
        for line in file:
            values = line.strip().split()

            if len(values) != 5:
                continue

            class_id, xc, yc, bw, bh = map(float, values)

            x1 = int((xc - bw / 2) * width)
            y1 = int((yc - bh / 2) * height)
            x2 = int((xc + bw / 2) * width)
            y2 = int((yc + bh / 2) * height)

            boxes.append((int(class_id), x1, y1, x2, y2))

    return boxes


def draw_box(image, coords, color, label):
    height, width = image.shape[:2]
    x1, y1, x2, y2 = map(int, coords)

    x1 = max(0, min(x1, width - 1))
    x2 = max(0, min(x2, width - 1))
    y1 = max(0, min(y1, height - 1))
    y2 = max(0, min(y2, height - 1))

    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

    cv2.putText(
        image,
        label,
        (x1, max(16, y1 - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1,
        cv2.LINE_AA,
    )


def get_predictions(model, image_path, nms_iou):
    results = model.predict(
        source=str(image_path),
        imgsz=IMAGE_SIZE,
        conf=CONFIDENCE,
        iou=nms_iou,
        device="cpu",
        verbose=False,
    )

    predictions = []

    if results and results[0].boxes is not None:
        boxes = results[0].boxes

        xyxy = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)

        for coords, confidence, class_id in zip(
            xyxy, confidences, class_ids
        ):
            predictions.append({
                "coords": coords,
                "confidence": float(confidence),
                "class_id": int(class_id),
            })

    return predictions


def main():
    if not IMAGE_DIR.exists():
        raise FileNotFoundError(f"Image folder not found: {IMAGE_DIR}")

    if not LABEL_DIR.exists():
        raise FileNotFoundError(f"Label folder not found: {LABEL_DIR}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    comparison_dir = OUTPUT_DIR / "comparisons"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        path for path in IMAGE_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in {
            ".jpg", ".jpeg", ".png", ".bmp", ".webp"
        }
    )

    if not image_paths:
        raise RuntimeError("No validation images were found.")

    print(f"Loading model: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))

    summary_rows = []

    for index, image_path in enumerate(image_paths, start=1):
        print(f"[{index}/{len(image_paths)}] {image_path.name}")

        original = cv2.imread(str(image_path))

        if original is None:
            print("  Skipped: image could not be read.")
            continue

        height, width = original.shape[:2]

        label_path = LABEL_DIR / f"{image_path.stem}.txt"
        ground_truth = read_ground_truth(label_path, width, height)

        panel_images = []
        counts = {}

        for nms_iou in NMS_THRESHOLDS:
            predictions = get_predictions(
                model, image_path, nms_iou
            )

            panel = original.copy()

            # Green boxes show existing annotations.
            for class_id, x1, y1, x2, y2 in ground_truth:
                draw_box(
                    panel,
                    (x1, y1, x2, y2),
                    (0, 200, 0),
                    f"GT {class_id}",
                )

            # Orange boxes show model predictions.
            for prediction in predictions:
                draw_box(
                    panel,
                    prediction["coords"],
                    (0, 100, 255),
                    f"P {prediction['confidence']:.2f}",
                )

            cv2.putText(
                panel,
                f"NMS IoU={nms_iou:.2f} | "
                f"Conf={CONFIDENCE:.2f} | "
                f"Predictions={len(predictions)}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 255),
                3,
                cv2.LINE_AA,
            )

            cv2.putText(
                panel,
                f"NMS IoU={nms_iou:.2f} | "
                f"Conf={CONFIDENCE:.2f} | "
                f"Predictions={len(predictions)}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

            panel_images.append(panel)
            counts[nms_iou] = len(predictions)

        comparison = np.concatenate(panel_images, axis=1)

        output_path = comparison_dir / (
            f"{image_path.stem}_nms_comparison.jpg"
        )

        if not cv2.imwrite(str(output_path), comparison):
            raise RuntimeError(f"Could not save: {output_path}")

        count_070 = counts[0.70]
        count_050 = counts[0.50]

        summary_rows.append({
            "image": image_path.name,
            "ground_truth_boxes": len(ground_truth),
            "predictions_nms_070": count_070,
            "predictions_nms_050": count_050,
            "boxes_removed_at_nms_050": count_070 - count_050,
            "comparison_image": str(output_path.relative_to(ROOT)),
        })

    csv_path = OUTPUT_DIR / "nms_comparison_summary.csv"

    with csv_path.open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        fieldnames = [
            "image",
            "ground_truth_boxes",
            "predictions_nms_070",
            "predictions_nms_050",
            "boxes_removed_at_nms_050",
            "comparison_image",
        ]

        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\nNMS comparison completed.")
    print(f"Images processed: {len(summary_rows)}")
    print(f"Comparison images: {comparison_dir}")
    print(f"Summary CSV: {csv_path}")
    print("Green = ground truth; orange = predictions.")


if __name__ == "__main__":
    main()
