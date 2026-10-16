
from pathlib import Path

from src.potato_vision.inference_gallery import build_inference_gallery


# Resolve paths from the project root, not the terminal's current directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

WEIGHTS_PATH = (
    PROJECT_ROOT
    / "runs"
    / "detect"
    / "potato_late_blight-4"
    / "weights"
    / "best.pt"
)

IMAGES_DIR = PROJECT_ROOT / "data" / "yolo" / "images" / "val"
LABELS_DIR = PROJECT_ROOT / "data" / "yolo" / "labels" / "val"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "reference_gallery"


def main():
    # Stop early if a path is incorrect.
    required_paths = {
        "Model checkpoint": WEIGHTS_PATH,
        "Validation images": IMAGES_DIR,
        "Validation labels": LABELS_DIR,
    }

    for description, path in required_paths.items():
        if not path.exists():
            raise FileNotFoundError(
                f"{description} not found:\n{path}"
            )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    outputs = build_inference_gallery(
        weights_path=WEIGHTS_PATH,
        images_dir=IMAGES_DIR,
        labels_dir=LABELS_DIR,
        output_dir=OUTPUT_DIR,
        output_stem="potato_yolov8_validation",
        n_images=12,
        n_cols=3,
        seed=42,
        imgsz=640,
        conf=0.25,
        iou=0.45,
        device="cpu",
        backend="pil",
        class_names=("late_blight",),
        class_colors={"late_blight": "#D55E00"},
        title="Potato Late Blight — YOLOv8 Validation Predictions",
        save_pdf=True,
        save_manifest=True,
        web_png=True,
        show=False,
    )

    print("\nGallery generation completed.")
    for name, path in outputs.items():
        if name in {"figure", "panels"}:
            continue
        if path is not None:
            print(f"{name}: {path}")


if __name__ == "__main__":
    main()
