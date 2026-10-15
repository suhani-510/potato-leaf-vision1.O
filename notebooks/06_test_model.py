
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    ROOT / "runs" / "detect" /
    "potato_late_blight-4" / "weights" / "best.pt"
)

# Search for images in both folders
SEARCH_DIRS = [
    ROOT / "data" / "raw" / "late_blight",
    ROOT / "data" / "yolo" / "images" / "val",
]

extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
images = []

for folder in SEARCH_DIRS:
    if folder.exists():
        images.extend(
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in extensions
        )

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

if not images:
    raise FileNotFoundError(
        "No images found. Check data/raw/late_blight "
        "and data/yolo/images/val."
    )

IMAGE_PATH = images[0]

print("Model:", MODEL_PATH)
print("Testing image:", IMAGE_PATH)
print("Image exists:", IMAGE_PATH.exists())

model = YOLO(str(MODEL_PATH))

results = model.predict(
    source=str(IMAGE_PATH),
    conf=0.40,
    save=True
)

for result in results:
    print("Detected objects:", len(result.boxes))
    print("Confidence scores:", result.boxes.conf.tolist())

print("Prediction completed. Check the saved output path above.")
