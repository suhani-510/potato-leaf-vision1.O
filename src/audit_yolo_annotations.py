
from pathlib import Path
from collections import Counter
import math

root = Path.cwd() / "data" / "yolo"
image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

for split in ("train", "val"):
    image_dir = root / "images" / split
    label_dir = root / "labels" / split

    print(f"\n========== {split.upper()} ==========")

    if not image_dir.is_dir() or not label_dir.is_dir():
        print("ERROR: image or label directory is missing.")
        continue

    images = sorted(
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in image_extensions
    )

    box_counts = Counter()
    issues = []
    total_boxes = 0
    areas = []

    for image in images:
        label_file = label_dir / f"{image.stem}.txt"

        if not label_file.is_file():
            issues.append(f"Missing label: {image.name}")
            continue

        lines = [
            line.strip()
            for line in label_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        count = 0

        for line_no, line in enumerate(lines, start=1):
            parts = line.split()

            try:
                if len(parts) != 5:
                    raise ValueError("expected 5 values")

                cls, x, y, w, h = map(float, parts)

                if cls != 0:
                    raise ValueError(f"unexpected class {cls}")

                if not all(math.isfinite(v) for v in (x, y, w, h)):
                    raise ValueError("non-finite coordinate")

                if not (0 <= x <= 1 and 0 <= y <= 1):
                    raise ValueError("center outside image")

                if not (0 < w <= 1 and 0 < h <= 1):
                    raise ValueError("invalid width or height")

                if x - w / 2 < -1e-6 or x + w / 2 > 1 + 1e-6:
                    raise ValueError("box extends beyond image horizontally")

                if y - h / 2 < -1e-6 or y + h / 2 > 1 + 1e-6:
                    raise ValueError("box extends beyond image vertically")

                areas.append(w * h)
                count += 1

            except ValueError as exc:
                issues.append(
                    f"{label_file.name}, line {line_no}: {exc} | {line}"
                )

        total_boxes += count
        box_counts[count] += 1

    print("Images:", len(images))
    print("Total valid boxes:", total_boxes)
    print("Images with 0 boxes:", box_counts[0])
    print("Images with 1 box:", box_counts[1])
    print("Images with 2+ boxes:", sum(
        n for count, n in box_counts.items() if count >= 2
    ))
    print("Box count distribution:", dict(sorted(box_counts.items())))

    if areas:
        print(
            "Median box area (% of image):",
            round(sorted(areas)[len(areas) // 2] * 100, 2),
        )
        print(
            "Smallest box area (% of image):",
            round(min(areas) * 100, 4),
        )
        print(
            "Largest box area (% of image):",
            round(max(areas) * 100, 2),
        )

    print("Label issues:", len(issues))
    for issue in issues[:20]:
        print(" -", issue)

    if len(issues) > 20:
        print(" ... and", len(issues) - 20, "more")
