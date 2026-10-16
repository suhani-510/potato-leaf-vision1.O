
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from .opencv_lesion_proposals import (
    find_leaf_mask,
    find_lesion_candidates,
)
from .segmentation_pipeline import (
    analyse_image,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_WEIGHTS = (
    PROJECT_ROOT
    / "runs"
    / "detect"
    / "potato_late_blight-4"
    / "weights"
    / "best.pt"
)

DEFAULT_IMAGES = PROJECT_ROOT / "data" / "yolo" / "images" / "val"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "integrated_pipeline"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def draw_yolo_boxes(image, detections):
    """Draw trained-model detections in green."""
    output = image.copy()

    for detection in detections:
        x1, y1, x2, y2 = map(
            int, detection["box_xyxy"]
        )

        label = (
            f'{detection["class_name"]} '
            f'{detection["confidence"]:.2f}'
        )

        cv2.rectangle(
            output, (x1, y1), (x2, y2), (0, 255, 0), 2
        )

        cv2.putText(
            output,
            label,
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    return output


def draw_cv_boxes(image, boxes):
    """Draw OpenCV-generated candidate boxes in orange."""
    output = image.copy()

    for index, (x1, y1, x2, y2) in enumerate(boxes, start=1):
        cv2.rectangle(
            output,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (0, 165, 255),
            2,
        )

        cv2.putText(
            output,
            f"CV candidate {index}",
            (int(x1), max(18, int(y1) - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 165, 255),
            1,
            cv2.LINE_AA,
        )

    return output


def run_pipeline(
    weights: Path,
    images_dir: Path,
    output_dir: Path,
    confidence: float = 0.25,
    image_size: int = 640,
    max_images: int = 0,
) -> None:
    """Run YOLO, OpenCV candidate generation, and mask analysis."""

    if not weights.is_file():
        raise FileNotFoundError(
            f"YOLO checkpoint not found: {weights}"
        )

    if not images_dir.is_dir():
        raise FileNotFoundError(
            f"Input image directory not found: {images_dir}"
        )

    if not 0 <= confidence <= 1:
        raise ValueError("Confidence must be between 0 and 1.")

    if image_size <= 0 or max_images < 0:
        raise ValueError(
            "Image size must be positive and max_images non-negative."
        )

    image_paths = sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if max_images:
        image_paths = image_paths[:max_images]

    if not image_paths:
        raise ValueError(f"No supported images found in {images_dir}")

    visualization_dir = output_dir / "visualizations"
    mask_dir = output_dir / "masks"
    segmentation_dir = output_dir / "segmentation"

    for directory in (
        output_dir,
        visualization_dir,
        mask_dir,
        segmentation_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    print(f"Loading YOLO model: {weights}")
    model = YOLO(str(weights))

    summary_rows = []
    detail_records = []

    for index, image_path in enumerate(image_paths, start=1):
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Skipping unreadable image: {image_path.name}")
            continue

        height, width = image.shape[:2]

        # 1. Run the trained YOLO detector.
        prediction = model.predict(
            source=image,
            conf=confidence,
            imgsz=image_size,
            device="cpu",
            verbose=False,
        )[0]

        yolo_detections = []

        if prediction.boxes is not None:
            for box in prediction.boxes:
                coords = (
                    box.xyxy[0].cpu().numpy().tolist()
                )
                class_id = int(box.cls[0].cpu().item())
                score = float(box.conf[0].cpu().item())

                yolo_detections.append({
                    "class_id": class_id,
                    "class_name": prediction.names.get(
                        class_id, str(class_id)
                    ),
                    "confidence": round(score, 5),
                    "box_xyxy": [
                        round(value, 2) for value in coords
                    ],
                })

        # 2. Generate OpenCV candidate regions.
        leaf_mask = find_leaf_mask(image)

        candidate_mask, candidate_boxes = find_lesion_candidates(
            image, leaf_mask
        )

        # 3. Run the existing heuristic segmentation analysis.
        # This function saves its own masks and overlay.
        segmentation_result = analyse_image(
            image_path, segmentation_dir
        )

        # 4. Save independent visualizations.
        yolo_view = draw_yolo_boxes(image, yolo_detections)
        cv_view = draw_cv_boxes(image, candidate_boxes)

        comparison = np.concatenate(
            [yolo_view, cv_view], axis=1
        )

        stem = image_path.stem

        cv2.imwrite(
            str(visualization_dir / f"{stem}_comparison.jpg"),
            comparison,
        )

        cv2.imwrite(
            str(mask_dir / f"{stem}_leaf_mask.png"),
            leaf_mask,
        )

        cv2.imwrite(
            str(mask_dir / f"{stem}_opencv_candidates.png"),
            candidate_mask,
        )

        # 5. Record detections and measurements.
        candidate_ratio = segmentation_result[
            "candidate_ratio_percent"
        ]

        record = {
            "image": image_path.name,
            "width": width,
            "height": height,
            "yolo_detection_count": len(yolo_detections),
            "opencv_candidate_count": len(candidate_boxes),
            "segmentation_candidate_region_count":
                segmentation_result["candidate_region_count"],
            "segmentation_candidate_ratio_percent":
                candidate_ratio,
            "segmentation_mask_type":
                "Unverified HSV heuristic candidates",
            "yolo_detections": yolo_detections,
            "opencv_candidate_boxes_xyxy": [
                list(map(int, box)) for box in candidate_boxes
            ],
        }

        detail_records.append(record)

        summary_rows.append({
            key: value
            for key, value in record.items()
            if key not in {
                "yolo_detections",
                "opencv_candidate_boxes_xyxy",
            }
        })

        print(
            f"[{index}/{len(image_paths)}] {image_path.name} | "
            f"YOLO: {len(yolo_detections)} | "
            f"OpenCV: {len(candidate_boxes)} | "
            f"Mask regions: "
            f"{segmentation_result['candidate_region_count']}"
        )

    # Save a flat CSV for inspection or future frontend use.
    csv_path = output_dir / "integrated_summary.csv"

    fieldnames = [
        "image",
        "width",
        "height",
        "yolo_detection_count",
        "opencv_candidate_count",
        "segmentation_candidate_region_count",
        "segmentation_candidate_ratio_percent",
        "segmentation_mask_type",
    ]

    with csv_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    # Save full detection coordinates and confidences.
    json_path = output_dir / "integrated_results.json"

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "description": (
                    "YOLO detections and unverified OpenCV/"
                    "segmentation candidates. Candidate masks are "
                    "not confirmed disease measurements."
                ),
                "images_processed": len(detail_records),
                "confidence_threshold": confidence,
                "image_size": image_size,
                "results": detail_records,
            },
            file,
            indent=2,
        )

    print("\nIntegrated pipeline completed.")
    print(f"Summary CSV: {csv_path}")
    print(f"Detailed JSON: {json_path}")
    print(f"Comparisons: {visualization_dir}")
    print(f"OpenCV masks: {mask_dir}")
    print(f"Segmentation outputs: {segmentation_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Integrated potato-leaf detection workflow."
    )

    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Maximum images; 0 processes all images.",
    )

    args = parser.parse_args()

    run_pipeline(
        weights=args.weights,
        images_dir=args.images,
        output_dir=args.output,
        confidence=args.confidence,
        image_size=args.imgsz,
        max_images=args.max_images,
    )


if __name__ == "__main__":
    main()
