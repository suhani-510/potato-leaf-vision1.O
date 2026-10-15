
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = ROOT / "data" / "yolo" / "data.yaml"

# Pretrained YOLO model
model = YOLO("yolov8n.pt")

# Train the detector
results = model.train(
    data=str(DATA_YAML),
    epochs=50,
    imgsz=640,
    batch=4,
    project=str(ROOT / "runs" / "detect"),
    name="potato_late_blight",
    patience=10,
    workers=0
)

print("Training completed!")
print("Best model:", ROOT / "runs" / "detect" /
      "potato_late_blight" / "weights" / "best.pt")
