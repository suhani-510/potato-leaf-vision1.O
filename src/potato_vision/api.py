
from __future__ import annotations

import base64
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "runs"
    / "detect"
    / "potato_late_blight-4"
    / "weights"
    / "best.pt"
)

# API configuration
MAX_FILE_SIZE = 10 * 1024 * 1024
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

app = FastAPI(
    title="Potato Vision API",
    description=(
        "Upload a potato-leaf image to detect "
        "potential late-blight regions."
    ),
    version="1.0.0",
)

# Allow a local frontend to call this API during development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def load_model() -> YOLO:
    """Load the existing YOLO checkpoint once."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"YOLO checkpoint not found: {MODEL_PATH}"
        )

    return YOLO(str(MODEL_PATH))


@app.get("/")
def home():
    return {
        "message": "Potato Vision API is running.",
        "docs": "/docs",
        "model": "YOLO late-blight detector",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_checkpoint_exists": MODEL_PATH.is_file(),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Receive an image and return YOLO detections
    plus a JPEG image with bounding boxes.
    """
    filename = file.filename or "uploaded_image.jpg"
    extension = Path(filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail="Upload a JPG, JPEG, PNG, WEBP, or BMP image.",
        )

    # Read at most MAX_FILE_SIZE + 1 bytes to enforce the limit.
    contents = await file.read(MAX_FILE_SIZE + 1)
    await file.close()

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Image exceeds the 10 MB upload limit.",
        )

    # Decode the uploaded bytes into an OpenCV image.
    image_array = np.frombuffer(contents, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Could not decode the uploaded image.",
        )

    try:
        model = load_model()

        result = model.predict(
            source=image,
            conf=0.10,
            imgsz=640,
            device="cpu",
            verbose=False,
        )[0]

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {exc}",
        ) from exc

    detections = []

    if result.boxes is not None:
        for box in result.boxes:
            coords = box.xyxy[0].cpu().numpy().tolist()
            class_id = int(box.cls[0].cpu().item())
            confidence = float(box.conf[0].cpu().item())

            detections.append(
                {
                    "class_id": class_id,
                    "class_name": result.names.get(
                        class_id, str(class_id)
                    ),
                    "confidence": round(confidence, 4),
                    "box_xyxy": [
                        round(float(value), 2)
                        for value in coords
                    ],
                }
            )

    # Ultralytics draws the predicted boxes on the image.
    annotated_image = result.plot()

    success, encoded_image = cv2.imencode(
        ".jpg", annotated_image
    )

    if not success:
        raise HTTPException(
            status_code=500,
            detail="Could not encode the annotated image.",
        )

    image_base64 = base64.b64encode(
        encoded_image.tobytes()
    ).decode("utf-8")

    height, width = image.shape[:2]

    return {
        "filename": Path(filename).name,
        "image_width": width,
        "image_height": height,
        "model": "YOLO late-blight detector",
        "detection_count": len(detections),
        "detections": detections,
        "annotated_image": f"data:image/jpeg;base64,{image_base64}",
        "notice": (
            "This model detects late-blight regions only. "
            "A missing detection does not prove the leaf is healthy."
        ),
    }
