
from pathlib import Path
import csv

import cv2
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO


# --------------------------------------------------
# 1. PROJECT PATHS AND SETTINGS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

IMAGE_DIR = ROOT / "data" / "yolo" / "images" / "val"
LABEL_DIR = ROOT / "data" / "yolo" / "labels" / "val"

MODEL_PATH = (
    ROOT
    / "runs"
    / "detect"
    / "potato_late_blight-4"
    / "weights"
    / "best.pt"
)

OUTPUT_DIR = ROOT / "outputs" / "visual_audit_thresholds"

IMAGE_SIZE = 640
PREDICTION_FLOOR = 0.001
THRESHOLDS = [0.25, 0.10]

# Colors are in OpenCV BGR format.
GT_COLOR = (0, 200, 0)       # Green: ground truth
PRED_COLOR = (0, 80, 255)    # Red-orange: model predictions


# --------------------------------------------------
# 2. READ GROUND-TRUTH LABELS
# YOLO format: class x_center y_center width height
# All coordinates are normalized between 0 and 1.
# --------------------------------------------------

def read_ground_truth(label_path, image_width, image_height):
    boxes = []

    if not label_path.exists():
        return boxes

    with label_path.open("r", encoding="utf-8") as file:
        for line in file:
            values = line.strip().split()

            if len(values) != 5:
                continue

            class_id, xc, yc, width, height = map(float, values)

            x1 = int((xc - width / 2) * image_width)
            y1 = int((yc - height / 2) * image_height)
            x2 = int((xc + width / 2) * image_width)
            y2 = int((yc + height / 2) * image_height)

            boxes.append(
                (int(class_id), x1, y1, x2, y2)
            )

    return boxes


# --------------------------------------------------
# 3. DRAW BOXES
# --------------------------------------------------

def draw_box(image, x1, y1, x2, y2, color, text):
    height, width = image.shape[:2]

    x1 = max(0, min(int(x1), width - 1))
    x2 = max(0, min(int(x2), width - 1))
    y1 = max(0, min(int(y1), height - 1))
    y2 = max(0, min(int(y2), height - 1))

    cv2.rectangle(
        image, (x1, y1), (x2, y2), color, 2
    )

    # Put the label above the box when possible.
    text_y = max(18, y1 - 6)

    cv2.putText(
        image,
        text,
        (x1, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        color,
        1,
        cv2.LINE_AA,
    )


def draw_ground_truth(image, boxes):
    for class_id, x1, y1, x2, y2 in boxes:
        draw_box(
            image,
            x1, y1, x2, y2,
            GT_COLOR,
            f"GT class {class_id}",
        )


def draw_predictions(image, prediction_boxes, threshold):
    for x1, y1, x2, y2, confidence, class_id in prediction_boxes:
        if confidence < threshold:
            continue

        draw_box(
            image,
            x1, y1, x2, y2,
            PRED_COLOR,
            f"Pred {confidence:.2f}",
        )


# --------------------------------------------------
# 4. CREATE A TWO-PANEL IMAGE
# Left: confidence 0.25
# Right: confidence 0.10
# Green boxes = ground truth
# Red-orange boxes = predictions
# --------------------------------------------------

def create_comparison(
    image_path,
    label_path,
    prediction_boxes,
):
    original = cv2.imread(str(image_path))

    if original is None:
        print(f"Skipping unreadable image: {image_path.name}")
        return None, None

    height, width = original.shape[:2]

    gt_boxes = read_ground_truth(
        label_path, width, height
    )

    panels = []
    prediction_counts = {}

    for threshold in THRESHOLDS:
        panel = original.copy()

        draw_ground_truth(panel, gt_boxes)
        draw_predictions(panel, prediction_boxes, threshold)

        count = sum(
            1 for box in prediction_boxes
            if box[4] >= threshold
        )
        prediction_counts[threshold] = count

        cv2.putText(
            panel,
            f"Confidence >= {threshold:.2f} | "
            f"GT: {len(gt_boxes)} | Predictions: {count}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )

        cv2.putText(
            panel,
            f"Confidence >= {threshold:.2f} | "
            f"GT: {len(gt_boxes)} | Predictions: {count}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )

        panels.append(panel)

    comparison = np.concatenate(panels, axis=1)

    return comparison, {
        "ground_truth_boxes": len(gt_boxes),
        "predictions_025": prediction_counts[0.25],
        "predictions_010": prediction_counts[0.10],
    }


# --------------------------------------------------
# 5. MAIN
# --------------------------------------------------

def main():
    if not IMAGE_DIR.exists():
        raise FileNotFoundError(
            f"Validation image folder not found: {IMAGE_DIR}"
        )

    if not LABEL_DIR.exists():
        raise FileNotFoundError(
            f"Validation label folder not found: {LABEL_DIR}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_PATH}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    comparison_dir = OUTPUT_DIR / "comparisons"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        path
        for path in IMAGE_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in {
            ".jpg", ".jpeg", ".png", ".bmp", ".webp"
        }
    )

    if not image_paths:
        raise RuntimeError(
            f"No validation images found in {IMAGE_DIR}"
        )

    print(f"Loading model: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))

    rows = []
    contact_sheet_items = []

    for index, image_path in enumerate(image_paths, start=1):
        print(
            f"[{index}/{len(image_paths)}] "
            f"Auditing {image_path.name}"
        )

        # Predict once at a low confidence floor.
        # Both thresholds use this same prediction set.
        results = model.predict(
            source=str(image_path),
            imgsz=IMAGE_SIZE,
            conf=PREDICTION_FLOOR,
            device="cpu",
            verbose=False,
        )

        prediction_boxes = []

        if results and results[0].boxes is not None:
            boxes = results[0].boxes

            xyxy = boxes.xyxy.cpu().numpy()
            confidences = boxes.conf.cpu().numpy()
            class_ids = boxes.cls.cpu().numpy().astype(int)

            for coords, confidence, class_id in zip(
                xyxy, confidences, class_ids
            ):
                x1, y1, x2, y2 = coords

                prediction_boxes.append(
                    (
                        int(x1), int(y1),
                        int(x2), int(y2),
                        float(confidence),
                        int(class_id),
                    )
                )

        label_path = LABEL_DIR / f"{image_path.stem}.txt"

        comparison, counts = create_comparison(
            image_path,
            label_path,
            prediction_boxes,
        )

        if comparison is None:
            continue

        output_path = comparison_dir / (
            f"{image_path.stem}_comparison.png"
        )

        if not cv2.imwrite(str(output_path), comparison):
            raise RuntimeError(
                f"Could not save image: {output_path}"
            )

        rows.append({
            "image": image_path.name,
            "ground_truth_boxes": counts["ground_truth_boxes"],
            "predictions_conf_025": counts["predictions_025"],
            "predictions_conf_010": counts["predictions_010"],
            "additional_predictions_at_010": (
                counts["predictions_010"]
                - counts["predictions_025"]
            ),
            "comparison_image": str(output_path.relative_to(ROOT)),
        })

        # Create small previews for the contact sheet.
        rgb = cv2.cvtColor(comparison, cv2.COLOR_BGR2RGB)
        thumbnail = Image.fromarray(rgb)
        thumbnail.thumbnail((720, 240))

        contact_sheet_items.append(
            (image_path.name, thumbnail.copy())
        )

    # Save per-image counts.
    csv_path = OUTPUT_DIR / "visual_audit_summary.csv"

    with csv_path.open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        fieldnames = [
            "image",
            "ground_truth_boxes",
            "predictions_conf_025",
            "predictions_conf_010",
            "additional_predictions_at_010",
            "comparison_image",
        ]

        writer = csv.DictWriter(
            file, fieldnames=fieldnames
        )
        writer.writeheader()
        writer.writerows(rows)

    # Build a contact sheet for quick scanning.
    if contact_sheet_items:
        columns = 2
        cell_width = 740
        cell_height = 275
        rows_needed = (
            len(contact_sheet_items) + columns - 1
        ) // columns

        sheet = Image.new(
            "RGB",
            (columns * cell_width, rows_needed * cell_height),
            "white",
        )
        draw = ImageDraw.Draw(sheet)

        for i, (name, thumbnail) in enumerate(
            contact_sheet_items
        ):
            col = i % columns
            row = i // columns

            x = col * cell_width + 10
            y = row * cell_height + 10

            draw.text(
                (x, y),
                name,
                fill="black",
            )
            sheet.paste(thumbnail, (x, y + 22))

        sheet_path = OUTPUT_DIR / "contact_sheet.jpg"
        sheet.save(sheet_path, quality=90)

    print("\nVisual audit completed.")
    print(f"Images compared: {len(rows)}")
    print(f"Individual comparisons: {comparison_dir}")
    print(f"Contact sheet: {OUTPUT_DIR / 'contact_sheet.jpg'}")
    print(f"Counts CSV: {csv_path}")
    print("\nGreen = ground-truth annotations")
    print("Red-orange = model predictions")


if __name__ == "__main__":
    main()
