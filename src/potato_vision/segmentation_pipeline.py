
"""Batch segmentation baseline for potato leaf images.

IMPORTANT:
The HSV mask highlights suspicious-looking pixels, not confirmed disease.
Original images, YOLO labels, and model weights are never modified.
"""

from pathlib import Path
import csv
import cv2
import numpy as np

from skimage.filters import threshold_otsu
from skimage.morphology import (
    binary_closing,
    binary_opening,
    remove_small_objects,
    disk,
    binary_dilation,
)

# --------------------------------------------------
# PATHS
# --------------------------------------------------

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = PROJECT_ROOT / "data" / "yolo" / "images" / "val"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "segmentation_pipeline"

MAX_IMAGES = 0  # 0 means process every validation image


# --------------------------------------------------
# REUSED SEGMENTATION FUNCTIONS
# Based on the single-image script you shared.
# --------------------------------------------------

def preprocess_image(bgr: np.ndarray) -> np.ndarray:
    """Improve local contrast and apply mild denoising."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    enhanced_l = clahe.apply(l_channel)
    enhanced_lab = cv2.merge(
        (enhanced_l, a_channel, b_channel)
    )

    enhanced = cv2.cvtColor(
        enhanced_lab,
        cv2.COLOR_LAB2BGR,
    )

    return cv2.GaussianBlur(enhanced, (3, 3), 0)


def estimate_leaf_context(bgr: np.ndarray) -> np.ndarray:
    """Estimate vegetation and nearby tissue; not precise leaf segmentation."""
    rgb = cv2.cvtColor(
        bgr, cv2.COLOR_BGR2RGB
    ).astype(np.float32) / 255.0

    red, green, blue = cv2.split(rgb)
    excess_green = 2 * green - red - blue

    exg = cv2.normalize(
        excess_green,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    ).astype(np.uint8)

    if int(exg.max()) == int(exg.min()):
        return np.ones(exg.shape, dtype=bool)

    threshold = threshold_otsu(exg)
    green_mask = exg > threshold

    green_mask = binary_closing(green_mask, disk(5))
    green_mask = remove_small_objects(
        green_mask,
        min_size=max(30, int(green_mask.size * 0.00005)),
    )

    context = binary_dilation(green_mask, disk(8))
    return context.astype(bool)


def hsv_candidate_mask(
    bgr: np.ndarray,
    leaf_context: np.ndarray,
) -> np.ndarray:
    """Find brown, reddish, or dark candidate pixels inside leaf context."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)

    brown = (
        (hue >= 2)
        & (hue <= 28)
        & (saturation >= 25)
        & (value >= 25)
        & (value <= 225)
    )

    reddish = (
        ((hue <= 8) | (hue >= 170))
        & (saturation >= 35)
        & (value >= 35)
    )

    dark_tissue = (value <= 75) & (saturation >= 15)

    mask = (brown | reddish | dark_tissue) & leaf_context

    mask = binary_opening(mask, disk(1))
    mask = binary_closing(mask, disk(2))
    mask = remove_small_objects(mask, min_size=3)

    return mask.astype(bool)


def analyse_image(image_path: Path, output_dir: Path) -> dict:
    """Process one image and save its masks, overlay, and region statistics."""
    original = cv2.imread(str(image_path))

    if original is None:
        raise ValueError(f"Could not read image: {image_path}")

    enhanced = preprocess_image(original)
    leaf_context = estimate_leaf_context(enhanced)
    candidate_mask = hsv_candidate_mask(enhanced, leaf_context)

    candidate_mask = binary_closing(
        candidate_mask,
        disk(2),
    ).astype(bool)

    mask_u8 = candidate_mask.astype(np.uint8)

    component_count, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            mask_u8,
            connectivity=8,
        )
    )

    overlay = cv2.cvtColor(
        enhanced, cv2.COLOR_BGR2RGB
    ).copy()

    # Red overlay indicates candidate pixels, not confirmed disease.
    overlay[candidate_mask] = (
        0.55 * overlay[candidate_mask]
        + 0.45 * np.array([255, 0, 0])
    ).astype(np.uint8)

    regions = []

    for index in range(1, component_count):
        x, y, width, height, area = stats[index]

        if area < 3:
            continue

        regions.append({
            "image": image_path.name,
            "region_id": len(regions) + 1,
            "x": int(x),
            "y": int(y),
            "width": int(width),
            "height": int(height),
            "area_pixels": int(area),
        })

        cv2.rectangle(
            overlay,
            (int(x), int(y)),
            (int(x + width), int(y + height)),
            (255, 255, 0),
            1,
        )

    stem = image_path.stem

    cv2.imwrite(
        str(output_dir / f"{stem}_enhanced.png"),
        enhanced,
    )
    cv2.imwrite(
        str(output_dir / f"{stem}_leaf_context.png"),
        leaf_context.astype(np.uint8) * 255,
    )
    cv2.imwrite(
        str(output_dir / f"{stem}_candidate_mask.png"),
        mask_u8 * 255,
    )
    cv2.imwrite(
        str(output_dir / f"{stem}_overlay.png"),
        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
    )

    leaf_pixels = int(leaf_context.sum())
    candidate_pixels = int(candidate_mask.sum())

    candidate_ratio = (
        100.0 * candidate_pixels / leaf_pixels
        if leaf_pixels > 0
        else 0.0
    )

    return {
        "image": image_path.name,
        "width": int(original.shape[1]),
        "height": int(original.shape[0]),
        "candidate_region_count": len(regions),
        "leaf_context_pixels": leaf_pixels,
        "candidate_pixels": candidate_pixels,
        "candidate_ratio_percent": round(candidate_ratio, 4),
        "mask_type": "HSV heuristic; unverified candidates",
        "regions": regions,
    }


def main() -> None:
    if not INPUT_DIR.exists():
        raise FileNotFoundError(
            f"Validation image directory not found: {INPUT_DIR}"
        )

    images = sorted(
        path
        for path in INPUT_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in {
            ".jpg", ".jpeg", ".png", ".bmp", ".webp"
        }
    )

    if MAX_IMAGES > 0:
        images = images[:MAX_IMAGES]

    if not images:
        raise RuntimeError(f"No images found in {INPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_summaries = []
    all_regions = []

    print(f"Input images: {len(images)}")
    print(f"Output directory: {OUTPUT_DIR}")

    for index, image_path in enumerate(images, start=1):
        result = analyse_image(image_path, OUTPUT_DIR)

        all_regions.extend(result.pop("regions"))
        all_summaries.append(result)

        print(
            f"[{index}/{len(images)}] {image_path.name}: "
            f"{result['candidate_region_count']} candidate regions, "
            f"{result['candidate_ratio_percent']:.2f}% candidate ratio"
        )

    summary_path = OUTPUT_DIR / "segmentation_summary.csv"
    with summary_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(all_summaries[0].keys()),
        )
        writer.writeheader()
        writer.writerows(all_summaries)

    regions_path = OUTPUT_DIR / "candidate_regions.csv"
    with regions_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        fields = [
            "image", "region_id", "x", "y",
            "width", "height", "area_pixels",
        ]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_regions)

    print("\nSegmentation baseline completed.")
    print(f"Image summaries: {summary_path}")
    print(f"Candidate regions: {regions_path}")
    print(f"Saved masks and overlays: {OUTPUT_DIR}")
    print(
        "Reminder: candidate pixels and area ratios are not "
        "validated disease measurements."
    )


if __name__ == "__main__":
    main()
