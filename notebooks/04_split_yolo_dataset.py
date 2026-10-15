
from pathlib import Path
import random
import shutil

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "yolo"

IMAGE_DIR = DATA_DIR / "images" / "train"
LABEL_DIR = DATA_DIR / "labels" / "train"

VAL_IMAGE_DIR = DATA_DIR / "images" / "val"
VAL_LABEL_DIR = DATA_DIR / "labels" / "val"

TRAIN_FILE = DATA_DIR / "train.txt"
VAL_FILE = DATA_DIR / "val.txt"

random.seed(42)

# Match each image with its annotation
pairs = []

for image in IMAGE_DIR.iterdir():
    if image.is_file() and image.suffix.lower() in {
        ".jpg", ".jpeg", ".png", ".bmp"
    }:
        label = LABEL_DIR / f"{image.stem}.txt"

        if label.exists():
            pairs.append((image, label))

if len(pairs) < 2:
    raise RuntimeError(
        f"Need at least 2 matched image-label pairs; found {len(pairs)}."
    )

random.shuffle(pairs)

# Reserve 20% for validation, at least one image
val_count = max(1, round(len(pairs) * 0.2))
val_pairs = pairs[:val_count]
train_pairs = pairs[val_count:]

VAL_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
VAL_LABEL_DIR.mkdir(parents=True, exist_ok=True)

for image, label in val_pairs:
    shutil.move(str(image), str(VAL_IMAGE_DIR / image.name))
    shutil.move(str(label), str(VAL_LABEL_DIR / label.name))

# Save paths relative to data/yolo/
TRAIN_FILE.write_text(
    "".join(f"images/train/{img.name}\n" for img, _ in train_pairs),
    encoding="utf-8"
)

VAL_FILE.write_text(
    "".join(f"images/val/{img.name}\n" for img, _ in val_pairs),
    encoding="utf-8"
)

print("Dataset split complete!")
print("Training images:", len(train_pairs))
print("Validation images:", len(val_pairs))
print("Training labels:", len(list(LABEL_DIR.glob("*.txt"))))
print("Validation labels:", len(list(VAL_LABEL_DIR.glob("*.txt"))))
