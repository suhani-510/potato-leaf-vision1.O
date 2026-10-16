
"""FastAPI backend for potato leaf detection and candidate segmentation."""

import base64
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

from src.potato_vision.segmentation_pipeline import (
    preprocess_image,
    estimate_leaf_context,
    hsv_candidate_mask,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = (
    PROJECT_ROOT
    / "runs"
    / "detect"
    / "potato_late_blight-4"
    / "weights"
    / "best.pt"
)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MAX_FILE_SIZE = 10 * 1024 * 1024
CONFIDENCE_THRESHOLD = 0.10

app = FastAPI(
    title="Potato Leaf Vision API",
    description="Late-blight detection and heuristic candidate segmentation.",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def load_model():
    """Load the trained YOLO detector once."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"YOLO weights not found: {MODEL_PATH}")

    return YOLO(str(MODEL_PATH))


def image_to_base64(image: np.ndarray) -> str:
    """Encode a BGR or grayscale image as a PNG base64 string."""
    success, encoded = cv2.imencode(".png", image)

    if not success:
        raise ValueError("Could not encode output image.")

    return base64.b64encode(encoded.tobytes()).decode("utf-8")


def make_segmentation_images(
    original: np.ndarray,
    boxes: list[dict],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Create candidate mask, segmentation overlay and combined overlay."""
    enhanced = preprocess_image(original)
    leaf_context = estimate_leaf_context(enhanced)
    candidate_mask = hsv_candidate_mask(enhanced, leaf_context)

    # Keep a binary mask for the segmentation view.
    mask_image = candidate_mask.astype(np.uint8) * 255

    # Red marks HSV-based candidate pixels, not confirmed lesions.
    segmentation = original.copy()
    red = np.zeros_like(original)
    red[:, :] = (0, 0, 255)

    segmentation[candidate_mask] = cv2.addWeighted(
        original, 0.55, red, 0.45, 0
    )[candidate_mask]

    # Combined view: segmentation overlay + YOLO bounding boxes.
    combined = segmentation.copy()

    for detection in boxes:
        x1, y1, x2, y2 = detection["box_xyxy"]
        label = (
            f'{detection["class_name"]} '
            f'{detection["confidence"]:.2f}'
        )

        cv2.rectangle(
            combined,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        text_y = max(20, y1 - 8)
        cv2.putText(
            combined,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    leaf_pixels = int(leaf_context.sum())
    candidate_pixels = int(candidate_mask.sum())

    stats = {
        "candidate_region_pixels": candidate_pixels,
        "estimated_leaf_context_pixels": leaf_pixels,
        "candidate_ratio_percent": round(
            100 * candidate_pixels / leaf_pixels, 2
        ) if leaf_pixels else 0.0,
        "mask_type": "HSV heuristic; unverified candidate pixels",
    }

    return mask_image, segmentation, combined, stats


@app.get("/")
def root():
    return {
        "message": "Potato Leaf Vision API is running.",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_exists": MODEL_PATH.exists(),
        "model_path": str(MODEL_PATH),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Detect late-blight regions and generate candidate segmentation."""
    filename = file.filename or "uploaded_image.jpg"
    extension = Path(filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported image format. Use JPG, PNG, WEBP or BMP.",
        )

    contents = await file.read(MAX_FILE_SIZE + 1)

    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Image is too large. Maximum size is 10 MB.",
        )

    encoded = np.frombuffer(contents, dtype=np.uint8)
    original = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    if original is None:
        raise HTTPException(
            status_code=400,
            detail="Could not read this image. Try another image file.",
        )

    height, width = original.shape[:2]

    try:
        model = load_model()

        prediction = model.predict(
            source=original,
            conf=CONFIDENCE_THRESHOLD,
            imgsz=640,
            device="cpu",
            verbose=False,
        )[0]

        detections = []

        if prediction.boxes is not None:
            for box in prediction.boxes:
                class_id = int(box.cls.item())
                confidence = float(box.conf.item())
                coordinates = box.xyxy[0].cpu().numpy().tolist()

                x1, y1, x2, y2 = [
                    int(round(value)) for value in coordinates
                ]

                class_name = model.names.get(class_id, str(class_id))

                detections.append({
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": round(confidence, 4),
                    "box_xyxy": [x1, y1, x2, y2],
                })

        mask, segmentation, combined, segmentation_stats = (
            make_segmentation_images(original, detections)
        )

        # Keep the existing frontend's annotated_image field compatible.
        annotated = prediction.plot()

        return {
            "filename": filename,
            "width": width,
            "height": height,
            "detection_count": len(detections),
            "detections": detections,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "annotated_image": image_to_base64(annotated),
            "segmentation_mask": image_to_base64(mask),
            "segmentation_image": image_to_base64(segmentation),
            "combined_image": image_to_base64(combined),
            "segmentation_stats": segmentation_stats,
            "disclaimer": (
                "YOLO boxes are model predictions for late blight. "
                "Red segmentation pixels are unverified HSV-based "
                "candidates and must not be treated as confirmed lesions."
            ),
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Image analysis failed: {exc}",
        ) from exc
