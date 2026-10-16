
from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    raise SystemExit(
        "Pillow is required. Install it with: "
        ".\\.venv\\Scripts\\python.exe -m pip install Pillow"
    )

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

SPLITS = ("train", "val", "test")
CLASSES = ("early_blight", "healthy", "late_blight")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def sha256_file(path: Path) -> str:
    """Calculate a hash of the original file bytes."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def readable_image(path: Path) -> bool:
    """Check that the file can be decoded as an image."""
    try:
        with Image.open(path) as image:
            image.verify()

        # Reopen to ensure the image can actually be decoded.
        with Image.open(path) as image:
            image.load()

        return True
    except Exception:
        return False


def audit_dataset() -> None:
    print("=" * 65)
    print("POTATO LEAF VISION - DATASET AUDIT")
    print("=" * 65)
    print(f"Dataset location: {DATA_DIR}\n")

    if not DATA_DIR.is_dir():
        print("ERROR: Dataset directory does not exist.")
        return

    all_files: dict[str, list[Path]] = defaultdict(list)
    all_stems: dict[str, set[str]] = {}
    corrupt_files: list[Path] = []
    total_images = 0

    # Audit each classification split.
    for split in SPLITS:
        split_dir = DATA_DIR / split

        print(f"\n--- {split.upper()} ---")
        split_stems: set[str] = set()
        split_count = 0

        if not split_dir.is_dir():
            print(f"WARNING: Missing split directory: {split_dir}")
            all_stems[split] = split_stems
            continue

        for class_name in CLASSES:
            class_dir = split_dir / class_name

            if not class_dir.is_dir():
                print(f"{class_name}: MISSING CLASS DIRECTORY")
                continue

            # Direct files only: exclude nested processing artifacts.
            images = sorted(
                path for path in class_dir.iterdir()
                if path.is_file()
                and path.suffix.lower() in IMAGE_EXTENSIONS
            )

            print(f"{class_name}: {len(images)} images")
            split_count += len(images)

            for path in images:
                split_stems.add(path.stem.lower())
                all_files[split].append(path)

                if not readable_image(path):
                    corrupt_files.append(path)
                    continue

                # Hash only successfully decoded files.
                file_hash = sha256_file(path)
                all_stems.setdefault(split, set())
                # Keep hashes separately below.
                image_hashes[file_hash].append((split, path))

        all_stems[split] = split_stems
        total_images += split_count
        print(f"Total {split}: {split_count}")

    print("\n" + "=" * 65)
    print("AUDIT RESULTS")
    print("=" * 65)
    print(f"Total classification images: {total_images}")

    # Filename stem overlaps.
    print("\n1. Filename stem overlaps:")
    found_overlap = False

    for i, left in enumerate(SPLITS):
        for right in SPLITS[i + 1:]:
            shared = all_stems.get(left, set()) & all_stems.get(right, set())

            if shared:
                found_overlap = True
                print(f"  {left} vs {right}: {len(shared)} shared stems")
                print(f"    Examples: {sorted(shared)[:10]}")

    if not found_overlap:
        print("  No filename stem overlaps found.")

    # Corrupt or unreadable images.
    print("\n2. Unreadable images:")
    if corrupt_files:
        print(f"  Found {len(corrupt_files)} unreadable image(s).")
        for path in corrupt_files[:20]:
            print(f"  {path.relative_to(PROJECT_ROOT)}")
    else:
        print("  No unreadable images found.")

    # Exact duplicates by SHA-256.
    print("\n3. Exact duplicate files across splits:")
    cross_split_duplicates = []

    for file_hash, entries in image_hashes.items():
        splits_present = {split for split, _ in entries}

        if len(splits_present) > 1:
            cross_split_duplicates.append(entries)

    if cross_split_duplicates:
        print(f"  Found {len(cross_split_duplicates)} duplicate group(s).")

        for entries in cross_split_duplicates[:20]:
            print("  Duplicate group:")
            for split, path in entries:
                print(f"    [{split}] {path.relative_to(PROJECT_ROOT)}")
    else:
        print("  No exact cross-split duplicates found.")

    print("\nAudit complete. No files were changed.")


if __name__ == "__main__":
    image_hashes: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    audit_dataset()
