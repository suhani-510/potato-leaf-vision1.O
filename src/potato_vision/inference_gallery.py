"""
Publication-quality inference gallery for the potato-leaf detection model.

Renders a mosaic of real model predictions (Exp 12 - YOLO26-N BoxGain, 1024 px)
on the held-out test split, with bounding boxes and *legible* labels
("<Class> <confidence>") suitable for both the README and a journal figure.

Design notes
------------
The previous gallery was unreadable because pre-rendered inference JPEGs were
downscaled to 416 px *before* being placed in the mosaic, so Ultralytics'
auto-scaled font ended up sub-pixel. Here the pipeline is inverted:

    1. inference at the training resolution (imgsz=1024)
    2. annotation drawn on the FULL-resolution frame with an EXPLICIT font size
       and line width, both proportional to the image's shorter side
    3. the mosaic is assembled from those full-resolution frames, with the
       panel size in inches chosen so the final raster is ~1:1 with the source
       pixels at 300 DPI

Every box, class name and confidence value comes from a real
`ultralytics` forward pass. Nothing is drawn by hand or edited.

Annotation backends
-------------------
``"pil"``           default. Full control: TrueType text, filled label boxes,
                    and guaranteed in-frame label placement.
``"supervision"``   ``sv.BoxAnnotator`` + ``sv.LabelAnnotator`` with explicit
                    thickness / text_scale. Label anchors are pre-clamped so
                    the filled background never leaves the frame.
``"ultralytics"``   ``results.plot(line_width=..., font_size=...)``. Uses the
                    Ultralytics colour palette instead of the custom one.

Usage
-----
As a script::

    python src/inference_gallery.py
    python src/inference_gallery.py --backend supervision --n-images 20 --seed 42

From a notebook (08_visual_inference.ipynb / 10_paper_graphics_generation.ipynb)::

    import sys; sys.path.insert(0, "src")
    from inference_gallery import build_inference_gallery
    build_inference_gallery(n_images=20, n_cols=5, seed=42)
"""

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from PIL import Image, ImageDraw, ImageFont

# ═══════════════════════════════════════════════════════════════════════════
#  CONFIGURATION - edit these paths / defaults, everything else follows
# ═══════════════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: Best checkpoint - Exp 12, YOLO26-N, BoxGain (box=15.0), trained at 1024 px.
WEIGHTS_PATH = (
    PROJECT_ROOT
    / "runs/detect/runs/detect/Experimentos_Abril/Exp_12_Nano_BoxGain/weights/best.pt"
)

#: Held-out test split (images + YOLO-format ground-truth labels).
TEST_IMAGES_DIR = PROJECT_ROOT / "datasets/test/images"
TEST_LABELS_DIR = PROJECT_ROOT / "datasets/test/labels"

#: Where the figure is written (PNG for the README, PDF for the paper).
OUTPUT_DIR = PROJECT_ROOT / "docs/figures"
OUTPUT_STEM = "inference_gallery"

#: Class order MUST match `datasets/data.yaml` (nc: 4).
CLASS_NAMES: tuple[str, ...] = ("Alternaria", "Lancha", "Sana", "Septoria")

#: Fixed, colour-blind-safe (Okabe-Ito) palette. No green: the subject is
#: foliage, so green boxes would vanish against the background.
CLASS_COLORS: dict[str, str] = {
    "Alternaria": "#D55E00",  # vermillion
    "Lancha": "#CC79A7",  # reddish purple
    "Sana": "#0072B2",  # blue
    "Septoria": "#F0E442",  # yellow
}

#: Inference parameters. `IMGSZ` must match the training resolution.
IMGSZ = 1024
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

#: Figure title (set ``title=None`` to drop it, e.g. for a journal figure
#: whose caption lives in the manuscript).
GALLERY_TITLE = "Visual Inference Gallery — Exp 12 · YOLO26-N BoxGain · Test Set"

#: Layout / export defaults.
N_IMAGES = 20
N_COLS = 5
SEED = 42
DPI = 300
PANEL_INCHES = 2.8  # 2.8 in x 300 DPI = 840 px per panel
PAD = 0.02  # inter-panel spacing, as a fraction of panel size
LETTERBOX_COLOR = (255, 255, 255)  # pad colour for non-square sources

#: Annotation geometry, expressed as a fraction of the image's shorter side,
#: so the result is resolution independent.
#: At the defaults (1024 px source, 2.8 in panel, 300 DPI) these give a ~1.2 pt
#: stroke and ~8 pt label text in the exported figure - above the 7 pt floor
#: most journals impose on figure lettering.
BOX_THICKNESS_RATIO = 0.006  # 1024 px -> 6 px stroke
FONT_SIZE_RATIO = 0.040  # 1024 px -> 41 px text

_FONT_CANDIDATES = (
    "arialbd.ttf",  # Windows
    "DejaVuSans-Bold.ttf",  # matplotlib / Linux
    "Helvetica-Bold.ttf",
    "LiberationSans-Bold.ttf",
)


# ═══════════════════════════════════════════════════════════════════════════
#  Small helpers
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class Detection:
    """A single model prediction, in absolute pixel coordinates."""

    xyxy: tuple[float, float, float, float]
    class_id: int
    class_name: str
    confidence: float

    @property
    def label(self) -> str:
        """Rendered label text, e.g. ``"Alternaria 0.97"``."""
        return f"{self.class_name} {self.confidence:.2f}"


@dataclass
class Panel:
    """One annotated frame plus the detections it contains."""

    image_path: Path
    image_rgb: np.ndarray
    detections: list[Detection] = field(default_factory=list)


def hex_to_rgb(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    return tuple(int(color[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def contrasting_text_color(rgb: Sequence[int]) -> tuple[int, int, int]:
    """Black on light fills, white on dark fills (WCAG relative luminance)."""
    srgb = [c / 255.0 for c in rgb]
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in srgb]
    luminance = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]
    return (0, 0, 0) if luminance > 0.35 else (255, 255, 255)


def load_font(size: int) -> ImageFont.FreeTypeFont:
    """Load a bold TrueType face at an explicit pixel size."""
    for name in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:  # matplotlib always ships DejaVu
        from matplotlib import font_manager

        path = font_manager.findfont("DejaVu Sans:bold", fallback_to_default=True)
        return ImageFont.truetype(path, size)
    except Exception:  # pragma: no cover - last resort, will look poor
        return ImageFont.load_default()


def _annotation_geometry(height: int, width: int) -> tuple[int, int]:
    """Return ``(box_thickness_px, font_size_px)`` for a frame of this size."""
    short_side = min(height, width)
    thickness = max(2, round(short_side * BOX_THICKNESS_RATIO))
    font_size = max(12, round(short_side * FONT_SIZE_RATIO))
    return thickness, font_size


# ═══════════════════════════════════════════════════════════════════════════
#  1. Image selection - seeded and class-stratified
# ═══════════════════════════════════════════════════════════════════════════


def ground_truth_classes(label_path: Path) -> set[int]:
    if not label_path.exists():
        return set()
    ids: set[int] = set()
    for line in label_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            ids.add(int(float(line.split()[0])))
    return ids


def select_candidate_images(
    images_dir: Path = TEST_IMAGES_DIR,
    labels_dir: Path = TEST_LABELS_DIR,
    *,
    class_names: Sequence[str] = CLASS_NAMES,
    seed: int = SEED,
    max_per_class: int = 40,
) -> list[Path]:
    """Build a reproducible, class-stratified candidate pool from the test split.

    Images are grouped by the classes present in their *ground-truth* labels,
    shuffled with a fixed seed, then interleaved round-robin so every class is
    represented before any class is over-sampled. Selection never looks at the
    model's output, so it cannot bias the gallery towards easy cases.

    Parameters
    ----------
    images_dir, labels_dir
        Test split directories (YOLO layout).
    class_names
        Class list in `data.yaml` order.
    seed
        RNG seed - fixes the sample for reproducibility.
    max_per_class
        Cap on the pool size per class (keeps inference cheap).

    Returns
    -------
    list[Path]
        Candidate image paths, interleaved by class.
    """
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    images = sorted({p for ext in exts for p in images_dir.glob(ext)})
    if not images:
        raise FileNotFoundError(f"No images found in {images_dir}")

    rng = random.Random(seed)
    by_class: dict[int, list[Path]] = {i: [] for i in range(len(class_names))}
    unlabelled: list[Path] = []

    for img in images:
        gt = ground_truth_classes(labels_dir / f"{img.stem}.txt")
        if not gt:
            unlabelled.append(img)
            continue
        # Assign to the lowest-frequency class present, so rare classes
        # (Sana, Septoria) are not starved by multi-label images.
        for class_id in sorted(gt):
            by_class[class_id].append(img)

    for paths in by_class.values():
        rng.shuffle(paths)
    rng.shuffle(unlabelled)

    pool: list[Path] = []
    seen: set[Path] = set()
    for rank in range(max_per_class):
        for class_id in range(len(class_names)):
            bucket = by_class[class_id]
            if rank < len(bucket) and bucket[rank] not in seen:
                pool.append(bucket[rank])
                seen.add(bucket[rank])
    pool.extend(p for p in unlabelled if p not in seen)
    return pool


# ═══════════════════════════════════════════════════════════════════════════
#  2. Inference - real predictions only
# ═══════════════════════════════════════════════════════════════════════════


def run_inference(
    image_paths: Sequence[Path],
    *,
    weights_path: Path = WEIGHTS_PATH,
    imgsz: int = IMGSZ,
    conf: float = CONF_THRESHOLD,
    iou: float = IOU_THRESHOLD,
    device: str | int | None = None,
    batch_size: int = 8,
    model=None,
):
    """Run the detector over ``image_paths`` at the training resolution.

    Yields ``(path, ultralytics.engine.results.Results)`` pairs in input order.
    """
    from ultralytics import YOLO

    if model is None:
        if not Path(weights_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {weights_path}")
        model = YOLO(str(weights_path))

    for start in range(0, len(image_paths), batch_size):
        chunk = list(image_paths[start : start + batch_size])
        results = model.predict(
            source=[str(p) for p in chunk],
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device,
            verbose=False,
        )
        yield from zip(chunk, results)


def results_to_detections(result, class_names: Sequence[str]) -> list[Detection]:
    """Convert an Ultralytics `Results` object into `Detection` records."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    names = result.names or {i: n for i, n in enumerate(class_names)}
    xyxy = boxes.xyxy.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)
    conf = boxes.conf.cpu().numpy()
    return [
        Detection(
            xyxy=(float(x1), float(y1), float(x2), float(y2)),
            class_id=int(c),
            class_name=str(names.get(int(c), class_names[int(c)])),
            confidence=float(p),
        )
        for (x1, y1, x2, y2), c, p in zip(xyxy, cls, conf)
    ]


# ═══════════════════════════════════════════════════════════════════════════
#  3. Annotation backends
# ═══════════════════════════════════════════════════════════════════════════


def annotate_pil(
    image_rgb: np.ndarray,
    detections: Iterable[Detection],
    *,
    class_colors: dict[str, str] = CLASS_COLORS,
) -> np.ndarray:
    """Draw boxes + filled labels on a full-resolution RGB frame (PIL backend).

    Guarantees, for every detection:

    * stroke width and font size proportional to the frame's shorter side;
    * label text ``"<Class> <conf>"`` on a solid class-coloured background,
      with automatic black/white text for maximum contrast;
    * the label background is clamped to stay entirely inside the frame -
      it flips from above the box to inside the box near the top edge, and
      is shifted left near the right edge.
    """
    canvas = Image.fromarray(image_rgb.copy())
    draw = ImageDraw.Draw(canvas)
    width, height = canvas.size
    thickness, font_size = _annotation_geometry(height, width)
    font = load_font(font_size)
    pad_x, pad_y = max(2, round(font_size * 0.35)), max(1, round(font_size * 0.22))

    for det in detections:
        x1, y1, x2, y2 = det.xyxy
        fill = hex_to_rgb(class_colors.get(det.class_name, "#FFFFFF"))
        text_color = contrasting_text_color(fill)

        draw.rectangle([x1, y1, x2, y2], outline=fill, width=thickness)

        left, top, right, bottom = draw.textbbox((0, 0), det.label, font=font)
        text_w, text_h = right - left, bottom - top
        bg_w, bg_h = text_w + 2 * pad_x, text_h + 2 * pad_y

        # Preferred anchor: just above the box, flush with its left edge.
        bg_x = x1 - thickness / 2
        bg_y = y1 - bg_h - thickness / 2
        if bg_y < 0:  # no room above -> drop the label inside the box
            bg_y = y1 + thickness / 2
        # Hard clamp: the filled label must never leave the frame.
        bg_x = min(max(0.0, bg_x), max(0.0, width - bg_w))
        bg_y = min(max(0.0, bg_y), max(0.0, height - bg_h))

        draw.rectangle([bg_x, bg_y, bg_x + bg_w, bg_y + bg_h], fill=fill)
        draw.text(
            (bg_x + pad_x - left, bg_y + pad_y - top),
            det.label,
            font=font,
            fill=text_color,
        )

    return np.asarray(canvas)


def annotate_supervision(
    image_rgb: np.ndarray,
    detections: Sequence[Detection],
    *,
    class_colors: dict[str, str] = CLASS_COLORS,
    class_names: Sequence[str] = CLASS_NAMES,
) -> np.ndarray:
    """Same rendering contract as :func:`annotate_pil`, via ``supervision``.

    Uses ``sv.BoxAnnotator`` + ``sv.LabelAnnotator`` with an explicit
    ``thickness`` / ``text_scale``. ``LabelAnnotator`` anchors its filled
    background to the box's top-left corner without clipping to the frame, so
    the label anchors are pre-clamped here on a *copy* of the boxes: the drawn
    boxes still come from the raw model output.
    """
    import supervision as sv

    if not detections:
        return image_rgb.copy()

    height, width = image_rgb.shape[:2]
    thickness, font_size = _annotation_geometry(height, width)
    # cv2 FONT_HERSHEY_SIMPLEX has a cap height of ~28 px at scale 1.0, so this
    # keeps the label roughly the same size as the PIL backend's TrueType text.
    text_scale = font_size / 28.0
    text_thickness = max(1, round(font_size / 12))
    text_padding = max(2, round(font_size * 0.30))

    palette = sv.ColorPalette(
        [sv.Color.from_hex(class_colors[name]) for name in class_names]
    )
    text_palette = sv.ColorPalette(
        [
            sv.Color(*contrasting_text_color(hex_to_rgb(class_colors[name])))
            for name in class_names
        ]
    )

    xyxy = np.array([d.xyxy for d in detections], dtype=np.float32)
    class_id = np.array([d.class_id for d in detections], dtype=int)
    confidence = np.array([d.confidence for d in detections], dtype=np.float32)
    labels = [d.label for d in detections]
    sv_detections = sv.Detections(xyxy=xyxy, class_id=class_id, confidence=confidence)

    # --- pre-clamp the label anchors ------------------------------------
    label_xyxy = xyxy.copy()
    for i, text in enumerate(labels):
        (text_w, text_h), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_thickness
        )
        bg_w = text_w + 2 * text_padding
        bg_h = text_h + 2 * text_padding
        # background is drawn above y1 -> keep y1 at least bg_h from the top
        label_xyxy[i, 1] = float(np.clip(label_xyxy[i, 1], bg_h, max(bg_h, height)))
        label_xyxy[i, 0] = float(np.clip(label_xyxy[i, 0], 0, max(0, width - bg_w)))
        label_xyxy[i, 2] = max(label_xyxy[i, 2], label_xyxy[i, 0] + 1)
        label_xyxy[i, 3] = max(label_xyxy[i, 3], label_xyxy[i, 1] + 1)
    label_detections = sv.Detections(
        xyxy=label_xyxy, class_id=class_id, confidence=confidence
    )

    # supervision annotators follow the cv2 BGR convention: annotate in BGR,
    # then hand back RGB. Skipping this silently permutes the class colours.
    scene = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    scene = sv.BoxAnnotator(color=palette, thickness=thickness).annotate(
        scene=scene, detections=sv_detections
    )
    scene = sv.LabelAnnotator(
        color=palette,
        text_color=text_palette,
        text_scale=text_scale,
        text_thickness=text_thickness,
        text_padding=text_padding,
        text_position=sv.Position.TOP_LEFT,
    ).annotate(scene=scene, detections=label_detections, labels=labels)
    return cv2.cvtColor(scene, cv2.COLOR_BGR2RGB)


def annotate_ultralytics(result, *, class_names: Sequence[str] = CLASS_NAMES) -> np.ndarray:
    """``results.plot()`` with EXPLICIT ``line_width`` / ``font_size``.

    Kept as a reference backend. Note it uses the Ultralytics colour palette,
    not :data:`CLASS_COLORS` - the legend adapts accordingly.
    """
    height, width = result.orig_shape
    thickness, font_size = _annotation_geometry(height, width)
    # pil=True routes through the TrueType annotator, which honours `font_size`
    # exactly (the cv2 path quantises it) and returns an RGB PIL image.
    return np.asarray(result.plot(line_width=thickness, font_size=font_size, pil=True))


def _ultralytics_class_colors(class_names: Sequence[str]) -> dict[str, str]:
    from ultralytics.utils.plotting import colors

    return {
        name: "#{:02X}{:02X}{:02X}".format(*colors(i, True)[::-1])
        for i, name in enumerate(class_names)
    }


# ═══════════════════════════════════════════════════════════════════════════
#  4. Mosaic assembly
# ═══════════════════════════════════════════════════════════════════════════


def letterbox(image_rgb: np.ndarray, size: int, color=LETTERBOX_COLOR) -> np.ndarray:
    """Resize to a `size` x `size` square, preserving aspect ratio via padding."""
    height, width = image_rgb.shape[:2]
    if height == width == size:
        return image_rgb
    scale = size / max(height, width)
    new_w, new_h = max(1, round(width * scale)), max(1, round(height * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=interp)
    canvas = np.full((size, size, 3), color, dtype=np.uint8)
    top, left = (size - new_h) // 2, (size - new_w) // 2
    canvas[top : top + new_h, left : left + new_w] = resized
    return canvas


def compose_mosaic(
    panels: Sequence[Panel],
    *,
    n_cols: int = N_COLS,
    panel_inches: float = PANEL_INCHES,
    dpi: int = DPI,
    pad: float = PAD,
    title: str | None = None,
    legend_colors: dict[str, str] | None = None,
    panel_size_px: int | None = None,
) -> plt.Figure:
    """Assemble annotated panels into a uniform grid with a class legend.

    Panels are letterboxed to a common square size, so every cell has an
    identical aspect ratio; axes are removed and inter-panel padding is
    minimal. The class legend is drawn *outside* the images, below the grid.
    """
    legend_colors = legend_colors or CLASS_COLORS
    n = len(panels)
    n_rows = (n + n_cols - 1) // n_cols
    size_px = panel_size_px or max(max(p.image_rgb.shape[:2]) for p in panels)

    legend_inches = 0.55
    title_inches = 0.45 if title else 0.0
    fig_w = n_cols * panel_inches
    fig_h = n_rows * panel_inches + legend_inches + title_inches

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi, facecolor="white")
    grid = fig.add_gridspec(
        n_rows,
        n_cols,
        left=0.005,
        right=0.995,
        bottom=legend_inches / fig_h,
        top=1.0 - (title_inches / fig_h) - 0.005,
        wspace=pad,
        hspace=pad,
    )

    for idx in range(n_rows * n_cols):
        ax = fig.add_subplot(grid[idx // n_cols, idx % n_cols])
        ax.set_axis_off()
        if idx < n:
            ax.imshow(letterbox(panels[idx].image_rgb, size_px), interpolation="antialiased")
        ax.set_xticks([])
        ax.set_yticks([])

    handles = [
        Patch(facecolor=color, edgecolor="#333333", linewidth=0.6, label=name)
        for name, color in legend_colors.items()
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        frameon=False,
        fontsize=11,
        handlelength=1.6,
        handleheight=1.0,
        columnspacing=2.0,
        bbox_to_anchor=(0.5, 0.004),
    )
    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", y=1.0 - 0.30 * title_inches / fig_h)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
#  5. End-to-end entry point
# ═══════════════════════════════════════════════════════════════════════════


def build_inference_gallery(
    *,
    weights_path: Path | str = WEIGHTS_PATH,
    images_dir: Path | str = TEST_IMAGES_DIR,
    labels_dir: Path | str = TEST_LABELS_DIR,
    output_dir: Path | str = OUTPUT_DIR,
    output_stem: str = OUTPUT_STEM,
    n_images: int = N_IMAGES,
    n_cols: int = N_COLS,
    seed: int = SEED,
    imgsz: int = IMGSZ,
    conf: float = CONF_THRESHOLD,
    iou: float = IOU_THRESHOLD,
    device: str | int | None = None,
    backend: str = "pil",
    class_names: Sequence[str] = CLASS_NAMES,
    class_colors: dict[str, str] | None = None,
    title: str | None = GALLERY_TITLE,
    dpi: int = DPI,
    panel_inches: float = PANEL_INCHES,
    skip_empty: bool = True,
    save_pdf: bool = True,
    save_manifest: bool = True,
    extra_png_paths: Sequence[Path | str] = (),
    web_png: bool = False,
    web_png_max_width: int = 2400,
    show: bool = False,
) -> dict[str, object]:
    """Build the full publication-quality inference gallery, end to end.

    Selects a seeded, class-stratified sample of test images, runs real
    inference at the training resolution, annotates each *full-resolution*
    frame with legible boxes and labels, assembles the mosaic and exports it.

    Parameters
    ----------
    weights_path
        Checkpoint to load (default: Exp 12 / YOLO26-N BoxGain ``best.pt``).
    images_dir, labels_dir
        Test split. Labels are used only to stratify the sample by class.
    output_dir, output_stem
        Figure destination: ``<output_dir>/<output_stem>.png`` (+ ``.pdf``).
    n_images, n_cols
        Mosaic geometry - 20 and 5 give the 4x5 grid used in the README.
    seed
        Fixes the image sample, making the figure reproducible.
    imgsz, conf, iou
        Inference settings. ``imgsz`` must match the training resolution.
    device
        Passed straight to Ultralytics (``None`` = auto, ``0`` = first GPU).
    backend
        ``"pil"`` (default), ``"supervision"`` or ``"ultralytics"``.
    class_colors
        Fixed colour per class; ignored for the ``"ultralytics"`` backend,
        which uses its own palette (the legend follows it).
    skip_empty
        Skip images the model produced no detection for - they would render
        as an empty panel. Selection is never conditioned on confidence, so
        the gallery is not cherry-picked towards easy cases.
    extra_png_paths
        Additional PNG destinations (e.g. the README copy in ``assets/``).
    web_png, web_png_max_width
        Also write ``<output_stem>_web.png``, a downscaled Lanczos copy. The
        300 DPI master is ~20 MB, which is heavy to ship in a repository and
        slow to load in a README; the web copy is ~1/10th the size.

    Returns
    -------
    dict
        ``{"png", "pdf", "manifest", "figure", "panels"}``.
    """
    backend = backend.lower()
    if backend not in {"pil", "supervision", "ultralytics"}:
        raise ValueError(f"Unknown backend: {backend!r}")

    images_dir, labels_dir = Path(images_dir), Path(labels_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    colors_map = dict(class_colors or CLASS_COLORS)
    legend_colors = (
        _ultralytics_class_colors(class_names) if backend == "ultralytics" else colors_map
    )

    # --- 1. reproducible, class-stratified candidate pool ----------------
    candidates = select_candidate_images(
        images_dir, labels_dir, class_names=class_names, seed=seed
    )

    # --- 2/3. inference at training resolution + high-res annotation -----
    panels: list[Panel] = []
    for path, result in run_inference(
        candidates,
        weights_path=Path(weights_path),
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
    ):
        detections = results_to_detections(result, class_names)
        if skip_empty and not detections:
            continue

        if backend == "ultralytics":
            annotated = annotate_ultralytics(result, class_names=class_names)
        else:
            image_rgb = cv2.cvtColor(result.orig_img, cv2.COLOR_BGR2RGB)
            annotated = (
                annotate_pil(image_rgb, detections, class_colors=colors_map)
                if backend == "pil"
                else annotate_supervision(
                    image_rgb,
                    detections,
                    class_colors=colors_map,
                    class_names=class_names,
                )
            )

        panels.append(Panel(image_path=path, image_rgb=annotated, detections=detections))
        if len(panels) >= n_images:
            break

    if not panels:
        raise RuntimeError("No detections produced - cannot build a gallery.")
    if len(panels) < n_images:
        print(f"[warn] only {len(panels)}/{n_images} panels had detections.")

    # --- 4. mosaic -------------------------------------------------------
    fig = compose_mosaic(
        panels,
        n_cols=n_cols,
        panel_inches=panel_inches,
        dpi=dpi,
        title=title,
        legend_colors=legend_colors,
    )

    # --- 5. export -------------------------------------------------------
    png_path = output_dir / f"{output_stem}.png"
    fig.savefig(png_path, dpi=dpi, facecolor="white", bbox_inches="tight", pad_inches=0.02)
    outputs: dict[str, object] = {
        "png": png_path,
        "pdf": None,
        "manifest": None,
        "web_png": None,
    }

    if save_pdf:
        pdf_path = output_dir / f"{output_stem}.pdf"
        fig.savefig(pdf_path, facecolor="white", bbox_inches="tight", pad_inches=0.02)
        outputs["pdf"] = pdf_path

    for extra in extra_png_paths:
        extra = Path(extra)
        extra.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(extra, dpi=dpi, facecolor="white", bbox_inches="tight", pad_inches=0.02)

    if web_png:
        web_path = output_dir / f"{output_stem}_web.png"
        with Image.open(png_path) as master:
            scale = min(1.0, web_png_max_width / master.width)
            master.convert("RGB").resize(
                (round(master.width * scale), round(master.height * scale)),
                Image.LANCZOS,
            ).save(web_path, optimize=True)
        outputs["web_png"] = web_path

    if save_manifest:
        manifest_path = output_dir / f"{output_stem}_detections.csv"
        with manifest_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                ["panel", "image", "class", "confidence", "x1", "y1", "x2", "y2"]
            )
            for i, panel in enumerate(panels):
                for det in panel.detections:
                    writer.writerow(
                        [
                            i,
                            panel.image_path.name,
                            det.class_name,
                            f"{det.confidence:.4f}",
                            *(f"{v:.1f}" for v in det.xyxy),
                        ]
                    )
        outputs["manifest"] = manifest_path

    if show:
        plt.show()
    else:
        plt.close(fig)

    outputs["figure"] = fig
    outputs["panels"] = panels

    n_det = sum(len(p.detections) for p in panels)
    print(
        f"Gallery ({backend}): {len(panels)} panels, {n_det} detections "
        f"-> {png_path}" + (f" | {outputs['pdf']}" if save_pdf else "")
    )
    return outputs


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--weights", type=Path, default=WEIGHTS_PATH)
    parser.add_argument("--images-dir", type=Path, default=TEST_IMAGES_DIR)
    parser.add_argument("--labels-dir", type=Path, default=TEST_LABELS_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--output-stem", default=OUTPUT_STEM)
    parser.add_argument("--n-images", type=int, default=N_IMAGES)
    parser.add_argument("--n-cols", type=int, default=N_COLS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--imgsz", type=int, default=IMGSZ)
    parser.add_argument("--conf", type=float, default=CONF_THRESHOLD)
    parser.add_argument("--iou", type=float, default=IOU_THRESHOLD)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--backend", choices=("pil", "supervision", "ultralytics"), default="pil"
    )
    parser.add_argument("--dpi", type=int, default=DPI)
    parser.add_argument("--panel-inches", type=float, default=PANEL_INCHES)
    parser.add_argument("--no-title", action="store_true")
    parser.add_argument("--no-pdf", action="store_true")
    parser.add_argument(
        "--web-png", action="store_true",
        help="Also write a downscaled <stem>_web.png (lightweight README copy).",
    )
    parser.add_argument("--web-png-max-width", type=int, default=2400)
    parser.add_argument(
        "--extra-png", type=Path, action="append", default=[],
        help="Additional PNG destination (repeatable), e.g. the README copy.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    build_inference_gallery(
        weights_path=args.weights,
        images_dir=args.images_dir,
        labels_dir=args.labels_dir,
        output_dir=args.output_dir,
        output_stem=args.output_stem,
        n_images=args.n_images,
        n_cols=args.n_cols,
        seed=args.seed,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        backend=args.backend,
        dpi=args.dpi,
        panel_inches=args.panel_inches,
        title=None if args.no_title else GALLERY_TITLE,
        save_pdf=not args.no_pdf,
        extra_png_paths=args.extra_png,
        web_png=args.web_png,
        web_png_max_width=args.web_png_max_width,
    )


if __name__ == "__main__":
    main()
