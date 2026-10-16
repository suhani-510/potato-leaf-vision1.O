
from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INITIAL_WEIGHTS = (
    PROJECT_ROOT
    / "runs"
    / "detect"
    / "potato_late_blight-4"
    / "weights"
    / "best.pt"
)

DATA_YAML = PROJECT_ROOT / "data" / "yolo" / "data.yaml"

OUTPUT_PROJECT = PROJECT_ROOT / "runs" / "detect"
RUN_NAME = "potato_late_blight_finetune_1"


def main() -> None:
    if not INITIAL_WEIGHTS.is_file():
        raise FileNotFoundError(
            f"Starting checkpoint not found: {INITIAL_WEIGHTS}"
        )

    if not DATA_YAML.is_file():
        raise FileNotFoundError(
            f"Dataset configuration not found: {DATA_YAML}"
        )

    print("Starting YOLO fine-tuning preparation")
    print(f"Initial weights: {INITIAL_WEIGHTS}")
    print(f"Dataset: {DATA_YAML}")
    print(f"Output run: {OUTPUT_PROJECT / RUN_NAME}")
    print("Device: CPU")

    model = YOLO(str(INITIAL_WEIGHTS))

    model.train(
        data=str(DATA_YAML),
        epochs=40,
        patience=10,
        batch=4,
        imgsz=640,
        device="cpu",
        workers=0,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.0005,
        lrf=0.01,
        weight_decay=0.0005,
        cos_lr=True,
        seed=42,
        deterministic=True,
        amp=False,
        cache=False,
        close_mosaic=10,
        fliplr=0.5,
        flipud=0.0,
        degrees=5.0,
        translate=0.05,
        scale=0.2,
        mosaic=0.5,
        mixup=0.0,
        project=str(OUTPUT_PROJECT),
        name=RUN_NAME,
        exist_ok=False,
        save=True,
        plots=True,
        val=True,
    )

    print("\nFine-tuning run completed.")
    print(f"Results directory: {OUTPUT_PROJECT / RUN_NAME}")


if __name__ == "__main__":
    main()
