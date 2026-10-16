
from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGE_DIR = PROJECT_ROOT / "data" / "yolo" / "images" / "train"
LABEL_DIR = PROJECT_ROOT / "data" / "yolo" / "labels" / "train"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "region_annotation_proposals"
PROPOSED_LABEL_DIR = OUTPUT_DIR / "labels"

MAX_IMAGES = 12

# Boxes are merged only when they overlap or nearly touch.
# This is a proposal parameter, not a medically validated threshold.
MERGE_GAP_PIXELS = 12

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_yolo_boxes(label_path: Path, width: int, height: int):
    """Read YOLO boxes and convert them to pixel coordinates."""
    boxes = []

    for line_number, line in enumerate(
        label_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue

        parts = line.split()
        if len(parts) != 5:
            raise ValueError(
                f"{label_path.name}, line {line_number}: expected 5 values"
            )

        class_id, xc, yc, bw, bh = map(float, parts)

        if class_id != 0:
            raise ValueError(
                f"{label_path.name}, line {line_number}: "
                f"unexpected class {class_id}"
            )

        x1 = max(0, min(width - 1, round((xc - bw / 2) * width)))
        y1 = max(0, min(height - 1, round((yc - bh / 2) * height)))
        x2 = max(0, min(width - 1, round((xc + bw / 2) * width)))
        y2 = max(0, min(height - 1, round((yc + bh / 2) * height)))

        if x2 <= x1 or y2 <= y1:
            raise ValueError(
                f"{label_path.name}, line {line_number}: invalid box"
            )

        boxes.append((x1, y1, x2, y2))

    return boxes


def merge_nearby_boxes(boxes, gap: int):
    """
    Merge overlapping or nearby boxes into connected groups.

    The result is a larger enclosing rectangle for each group.
    It is not a pixel-level disease segmentation.
    """
    if not boxes:
        return []

    groups = [{i} for i in range(len(boxes))]

    def bounds(group):
        return (
            min(boxes[i][0] for i in group),
            min(boxes[i][1] for i in group),
            max(boxes[i][2] for i in group),
            max(boxes[i][3] for i in group),
        )

    changed = True
    while changed:
        changed = False

        for i in range(len(groups)):
            if changed:
                break

            for j in range(i + 1, len(groups)):
                ax1, ay1, ax2, ay2 = bounds(groups[i])
                bx1, by1, bx2, by2 = bounds(groups[j])

                separated = (
                    ax2 + gap < bx1
                    or bx2 + gap < ax1
                    or ay2 + gap < by1
                    or by2 + gap < ay1
                )

                if not separated:
                    groups[i] |= groups[j]
                    groups.pop(j)
                    changed = True
                    break

    return sorted(
        (bounds(group) for group in groups),
        key=lambda b: (b[1], b[0]),
    )


def save_yolo_labels(boxes, width: int, height: int, output_path: Path):
    """Save proposed boxes in YOLO format without touching source labels."""
    lines = []

    for x1, y1, x2, y2 in boxes:
        xc = ((x1 + x2) / 2) / width
        yc = ((y1 + y2) / 2) / height
        bw = (x2 - x1) / width
        bh = (y2 - y1) / height

        lines.append(
            f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"
        )

    output_path.write_text(
        "\n".join(lines) + ("\n" if lines else ""),
        encoding="utf-8",
    )


def draw_boxes(image, boxes, color, prefix):
    for index, (x1, y1, x2, y2) in enumerate(boxes, start=1):
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            image,
            f"{prefix}{index}",
            (x1, max(18, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )
    return image


def main():
    if not IMAGE_DIR.is_dir() or not LABEL_DIR.is_dir():
        raise SystemExit(
            "Training image or label directory is missing. "
            "Run this from the project root."
        )

    image_paths = sorted(
        p for p in IMAGE_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PROPOSED_LABEL_DIR.mkdir(parents=True, exist_ok=True)

    samples = []
    total_original = 0
    total_proposed = 0
    skipped = 0

    for image_path in image_paths:
        label_path = LABEL_DIR / f"{image_path.stem}.txt"

        if not label_path.is_file():
            print(f"Skipping missing label: {image_path.name}")
            skipped += 1
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Skipping unreadable image: {image_path.name}")
            skipped += 1
            continue

        height, width = image.shape[:2]

        try:
            original_boxes = read_yolo_boxes(label_path, width, height)
        except ValueError as exc:
            print(f"Skipping invalid annotation: {exc}")
            skipped += 1
            continue

        proposed_boxes = merge_nearby_boxes(
            original_boxes, MERGE_GAP_PIXELS
        )

        save_yolo_labels(
            proposed_boxes,
            width,
            height,
            PROPOSED_LABEL_DIR / f"{image_path.stem}.txt",
        )

        total_original += len(original_boxes)
        total_proposed += len(proposed_boxes)

        original_view = draw_boxes(
            image.copy(), original_boxes, (0, 255, 0), "O"
        )
        proposed_view = draw_boxes(
            image.copy(), proposed_boxes, (0, 165, 255), "R"
        )

        original_view = cv2.cvtColor(original_view, cv2.COLOR_BGR2RGB)
        proposed_view = cv2.cvtColor(proposed_view, cv2.COLOR_BGR2RGB)

        samples.append(
            (image_path.name, original_view, proposed_view,
             len(original_boxes), len(proposed_boxes))
        )

    if not samples:
        raise SystemExit("No images could be processed.")

    samples = samples[:MAX_IMAGES]
    rows = len(samples)

    figure, axes = plt.subplots(
        rows, 2, figsize=(14, 4 * rows), squeeze=False
    )

    for row, sample in enumerate(samples):
        name, original, proposed, old_count, new_count = sample

        axes[row, 0].imshow(original)
        axes[row, 0].set_title(
            f"{name} — original boxes: {old_count}"
        )
        axes[row, 0].axis("off")

        axes[row, 1].imshow(proposed)
        axes[row, 1].set_title(
            f"Proposed merged boxes: {new_count}"
        )
        axes[row, 1].axis("off")

    figure.suptitle(
        "Original Annotations vs. Region Proposals\n"
        "Green = original; orange = proposed merged regions",
        fontsize=14,
    )
    figure.tight_layout()

    preview_path = OUTPUT_DIR / "region_proposal_comparison.png"
    figure.savefig(preview_path, dpi=140, bbox_inches="tight")
    plt.close(figure)

    print("\n========== REGION PROPOSAL AUDIT ==========")
    print("Training images found:", len(image_paths))
    print("Images processed:", len(samples))
    print("Images skipped:", skipped)
    print("Original boxes across processed dataset:", total_original)
    print("Proposed boxes across processed dataset:", total_proposed)
    print("Merge gap in pixels:", MERGE_GAP_PIXELS)
    print("Preview:", preview_path)
    print("Proposed labels:", PROPOSED_LABEL_DIR)
    print("Original labels were not modified.")


if __name__ == "__main__":
    main()
