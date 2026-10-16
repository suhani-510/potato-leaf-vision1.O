
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.patches import Rectangle


ROOT = Path.cwd()
IMAGE_DIR = ROOT / "data/yolo/images/val"
LABEL_DIR = ROOT / "data/yolo/labels/val"
OUTPUT_DIR = ROOT / "outputs/missed_image_annotation_review"

PREFIXES = [
    "e95f2954-1b68-490d-9684-90b3080676b1",
    "ea356b9c-9090-4f0d-9083-d1f63c77372d",
    "ebd47dca-590b-43c4-812d-32f95b776690",
]


def read_boxes(label_path, image_width, image_height):
    boxes = []

    if not label_path.exists():
        return boxes, False

    for line in label_path.read_text(encoding="utf-8").splitlines():
        values = line.split()

        if len(values) != 5:
            continue

        try:
            class_id = int(values[0])
            x, y, w, h = map(float, values[1:])

            if w <= 0 or h <= 0:
                continue

            x1 = (x - w / 2) * image_width
            y1 = (y - h / 2) * image_height
            box_width = w * image_width
            box_height = h * image_height

            boxes.append((class_id, x1, y1, box_width, box_height))

        except ValueError:
            continue

    return boxes, True


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(1, 3, figsize=(16, 6))

for ax, prefix in zip(axes, PREFIXES):
    matches = list(IMAGE_DIR.glob(f"{prefix}*"))

    if len(matches) != 1:
        ax.set_title(f"Expected one image; found {len(matches)}")
        ax.axis("off")
        continue

    image_path = matches[0]
    label_path = LABEL_DIR / f"{image_path.stem}.txt"

    with Image.open(image_path) as source:
        image = source.convert("RGB")

    width, height = image.size
    boxes, label_exists = read_boxes(label_path, width, height)

    ax.imshow(image)

    for class_id, x, y, box_width, box_height in boxes:
        ax.add_patch(
            Rectangle(
                (x, y),
                box_width,
                box_height,
                fill=False,
                edgecolor="lime",
                linewidth=2,
            )
        )
        ax.text(
            x,
            max(0, y - 4),
            f"class {class_id}",
            color="lime",
            fontsize=9,
            bbox={"facecolor": "black", "alpha": 0.6, "pad": 2},
        )

    label_status = (
        f"{len(boxes)} boxes"
        if label_exists
        else "LABEL FILE MISSING"
    )

    ax.set_title(f"{image_path.name}\n{label_status}", fontsize=8)
    ax.axis("off")

    print(f"\nImage: {image_path.name}")
    print(f"Label: {label_path.name}")
    print(f"Label file exists: {label_exists}")
    print(f"Boxes: {len(boxes)}")

plt.tight_layout()

output_path = OUTPUT_DIR / "missed_images_ground_truth.png"
fig.savefig(output_path, dpi=180, bbox_inches="tight")
plt.close(fig)

print(f"\nPreview saved to: {output_path}")
