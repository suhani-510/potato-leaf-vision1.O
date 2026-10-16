
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
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "detection_pipeline"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def draw_boxes(image, boxes, color, prefix):
    """Draw pixel-coordinate boxes on a copy of an image."""
    output = image.copy()

    for index, box in enumerate(boxes, start=1):
        x1, y1, x2, y2 = map(int, box[:4])
        label = (
            f"{prefix} {box[4]:.2f}"
            if len(box) > 4
            else f"{prefix} {index}"
        )

        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            output,
            label,
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )

    return output


def run_pipeline(
    weights: Path,
    images_dir: Path,
    output_dir: Path,
    confidence: float = 0.25,
    image_size: int = 640,
    max_images: int = 12,
) -> None:
    """Run YOLO and OpenCV candidate generation on the same images."""

    if not weights.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {weights}")

    if not images_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {images_dir}")

    image_paths = sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if max_images > 0:
        image_paths = image_paths[:max_images]

    if not image_paths:
        raise ValueError(f"No images found in: {images_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "visualizations").mkdir(exist_ok=True)
    (output_dir / "masks").mkdir(exist_ok=True)

    model = YOLO(str(weights))
    csv_path = output_dir / "detection_results.csv"
    jsonl_path = output_dir / "detection_results.jsonl"

    with (
        csv_path.open("w", newline="", encoding="utf-8") as csv_file,
        jsonl_path.open("w", encoding="utf-8") as jsonl_file,
    ):
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "image",
                "width",
                "height",
                "yolo_boxes",
                "opencv_candidates",
                "leaf_mask_pixels",
                "status",
            ],
        )
        writer.writeheader()

        for image_path in image_paths:
            image = cv2.imread(str(image_path))

            if image is None:
                print(f"Skipping unreadable image: {image_path.name}")
                continue

            height, width = image.shape[:2]

            # Method 1: the trained YOLO detector.
            prediction = model.predict(
                source=image,
                conf=confidence,
                imgsz=image_size,
                device="cpu",
                verbose=False,
            )[0]

            yolo_boxes = []
            if prediction.boxes is not None:
                for box in prediction.boxes:
                    x1, y1, x2, y2 = (
                        box.xyxy[0].cpu().numpy().tolist()
                    )
                    score = float(box.conf[0].cpu().item())
                    class_id = int(box.cls[0].cpu().item())

                    yolo_boxes.append(
                        {
                            "class_id": class_id,
                            "class_name": prediction.names.get(
                                class_id, str(class_id)
                            ),
                            "confidence": score,
                            "box_xyxy": [
                                round(x1, 2),
                                round(y1, 2),
                                round(x2, 2),
                                round(y2, 2),
                            ],
                        }
                    )

            # Method 2: the existing OpenCV candidate generator.
            leaf_mask = find_leaf_mask(image)
            candidate_mask, cv_boxes = find_lesion_candidates(
                image, leaf_mask
            )

            yolo_coords = [
                (*item["box_xyxy"], item["confidence"])
                for item in yolo_boxes
            ]

            yolo_view = draw_boxes(
                image, yolo_coords, (0, 255, 0), "YOLO"
            )
            cv_view = draw_boxes(
                image, cv_boxes, (0, 165, 255), "CV"
            )

            comparison = np.concatenate([yolo_view, cv_view], axis=1)

            # Save masks separately for inspection.
            stem = image_path.stem
            cv2.imwrite(
                str(output_dir / "masks" / f"{stem}_leaf.png"),
                leaf_mask,
            )
            cv2.imwrite(
                str(output_dir / "masks" / f"{stem}_candidates.png"),
                candidate_mask,
            )
            cv2.imwrite(
                str(
                    output_dir
                    / "visualizations"
                    / f"{stem}_comparison.jpg"
                ),
                comparison,
            )

            record = {
                "image": image_path.name,
                "width": width,
                "height": height,
                "yolo": yolo_boxes,
                "opencv_candidates": [
                    list(map(int, box)) for box in cv_boxes
                ],
                "leaf_mask_pixels": int(
                    np.count_nonzero(leaf_mask)
                ),
            }

            jsonl_file.write(json.dumps(record) + "\n")

            writer.writerow(
                {
                    "image": image_path.name,
                    "width": width,
                    "height": height,
                    "yolo_boxes": len(yolo_boxes),
                    "opencv_candidates": len(cv_boxes),
                    "leaf_mask_pixels": record["leaf_mask_pixels"],
                    "status": "processed",
                }
            )

            print(
                f"{image_path.name}: "
                f"YOLO={len(yolo_boxes)}, "
                f"OpenCV={len(cv_boxes)}"
            )

    print("\nPipeline completed.")
    print(f"CSV summary: {csv_path}")
    print(f"Detailed detections: {jsonl_path}")
    print(f"Visualizations: {output_dir / 'visualizations'}")
    print(f"Masks: {output_dir / 'masks'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare YOLO detections with OpenCV lesion proposals."
    )
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--max-images",
        type=int,
        default=12,
        help="Maximum images to process; use 0 for all images.",
    )

    args = parser.parse_args()

    if not 0 <= args.confidence <= 1:
        parser.error("--confidence must be between 0 and 1.")
    if args.imgsz <= 0:
        parser.error("--imgsz must be positive.")
    if args.max_images < 0:
        parser.error("--max-images must be zero or positive.")

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
