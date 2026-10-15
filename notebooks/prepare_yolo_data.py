
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent.parent

LABEL_DIR = ROOT / "data" / "yolo" / "labels" / "train"
SOURCE_DIR = ROOT / "data" / "raw" / "late_blight"
IMAGE_DIR = ROOT / "data" / "yolo" / "images" / "train"
TRAIN_FILE = ROOT / "data" / "yolo" / "train.txt"

# Find images using case-insensitive filenames
source_images = {}

for path in SOURCE_DIR.rglob("*"):
    if path.is_file() and path.suffix.lower() in {
        ".jpg", ".jpeg", ".png", ".bmp"
    }:
        source_images[path.stem.lower()] = path

labels = sorted(LABEL_DIR.glob("*.txt"))
train_paths = []
missing = []

for label in labels:
    source = source_images.get(label.stem.lower())

    if source is None:
        missing.append(label.name)
        continue

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    destination = IMAGE_DIR / source.name
    shutil.copy2(source, destination)

    train_paths.append(f"images/train/{source.name}")

if not train_paths:
    print("No matching images found. train.txt was not changed.")
    raise SystemExit(1)

# Save paths relative to data/yolo/
temp_file = TRAIN_FILE.with_suffix(".tmp")
temp_file.write_text(
    "\n".join(train_paths) + "\n",
    encoding="utf-8"
)
temp_file.replace(TRAIN_FILE)

print("Labels found:", len(labels))
print("Images matched:", len(train_paths))
print("Images not found:", len(missing))
print("Images copied to:", IMAGE_DIR)

if missing:
    print("First unmatched labels:", missing[:10])
