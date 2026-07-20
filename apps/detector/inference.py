"""Edge Impulse FOMO object detection inference via the Linux Python SDK.

Provides `detect(image_path)` which:
  1. Loads the .eim model (lazy singleton)
  2. Runs inference on the given image
  3. Returns centroid detections + saves an annotated image with overlays

Requires:
  - pip install edge_impulse_linux opencv-python-headless
  - Model file at models/modelfile.eim (download via edge-impulse-linux-runner)
"""

from __future__ import annotations

import logging
import os
import platform
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import logfire
import numpy as np

log = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent


def default_model_path(machine: str | None = None) -> Path:
    """Return the architecture-specific Edge Impulse runner path."""
    override = os.getenv("EI_MODEL_PATH")
    if override:
        return Path(override).expanduser().resolve()

    architecture = (machine or platform.machine()).lower()
    filename = (
        "modelfile.aarch64.eim"
        if architecture in {"aarch64", "arm64"}
        else "modelfile.eim"
    )
    return ROOT_DIR / "models" / filename


MODEL_PATH = default_model_path()
OUTPUT_DIR = ROOT_DIR / "data" / "detections"

# Colors per label (BGR for OpenCV)
LABEL_COLORS = {
    "scalpel": (0, 200, 255),    # orange
    "scissors": (255, 100, 100),  # blue
    "sponge": (100, 255, 100),    # green
    "tweezers": (180, 100, 255),  # purple
}
DEFAULT_COLOR = (200, 200, 200)
FONT = cv2.FONT_HERSHEY_SIMPLEX
CONFIDENCE_THRESHOLD = 0.9


@dataclass
class Detection:
    label: str
    confidence: float
    x: int
    y: int
    width: int
    height: int
    green_context_fraction: float = 0.0


@dataclass
class DetectionResult:
    detections: list[Detection] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    annotated_path: Optional[str] = None
    inference_ms: float = 0.0
    model_name: str = ""
    frame_width: int = 0
    frame_height: int = 0


# ── Lazy model singleton ─────────────────────────────────────────────

_runner = None
_model_info = None


def _get_runner():
    global _runner, _model_info
    if _runner is not None:
        return _runner, _model_info

    if not MODEL_PATH.exists():
        log.warning("Model file not found at %s", MODEL_PATH)
        return None, None

    # Bypass edge_impulse_linux.__init__ which tries to import pyaudio
    import sys, types, importlib.util
    if "edge_impulse_linux" not in sys.modules:
        spec = importlib.util.find_spec("edge_impulse_linux")
        pkg = types.ModuleType("edge_impulse_linux")
        pkg.__path__ = list(spec.submodule_search_locations)
        sys.modules["edge_impulse_linux"] = pkg

    from edge_impulse_linux.image import ImageImpulseRunner

    _runner = ImageImpulseRunner(str(MODEL_PATH))
    _model_info = _runner.init()
    project = _model_info.get("project", {})
    log.info(
        "Loaded EI model: %s / %s",
        project.get("owner", "?"),
        project.get("name", "?"),
    )
    return _runner, _model_info


def _draw_overlay(img_rgb: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Draw centroid markers and labels on a copy of the image.

    FOMO (constrained_object_detection) returns centroids, not true
    bounding boxes — x/y are the center of the detected grid cell.
    """
    out = img_rgb.copy()
    h, w = out.shape[:2]
    thickness = max(1, min(h, w) // 300)
    font_scale = max(0.35, min(h, w) / 1200)
    pad = max(2, thickness * 2)
    radius = max(4, min(h, w) // 60)

    for det in detections:
        color = LABEL_COLORS.get(det.label, DEFAULT_COLOR)
        cx = det.x + det.width // 2
        cy = det.y + det.height // 2

        # Centroid circle
        cv2.circle(out, (cx, cy), radius, color, thickness)
        cv2.circle(out, (cx, cy), 2, color, -1)  # filled dot at center

        # Label background + text
        text = f"{det.label} {det.confidence:.0%}"
        (tw, th), baseline = cv2.getTextSize(text, FONT, font_scale, thickness)
        label_y = max(cy - radius - pad, th + pad)
        label_x = cx - tw // 2
        cv2.rectangle(
            out,
            (label_x - pad, label_y - th - pad),
            (label_x + tw + pad, label_y + pad),
            color,
            -1,
        )
        cv2.putText(
            out, text, (label_x, label_y), FONT, font_scale, (0, 0, 0), thickness
        )

    return out


def _green_context_fraction(
    img_rgb: np.ndarray,
    detection: Detection,
    radius: int = 24,
) -> float:
    """Measure green-drape coverage around a detected object centroid."""
    height, width = img_rgb.shape[:2]
    center_x = detection.x + detection.width // 2
    center_y = detection.y + detection.height // 2
    x0 = max(0, center_x - radius)
    x1 = min(width, center_x + radius + 1)
    y0 = max(0, center_y - radius)
    y1 = min(height, center_y + radius + 1)
    pixels = img_rgb[y0:y1, x0:x1].astype(np.int16)
    green_pixels = (
        (pixels[..., 1] - pixels[..., 0] > 25)
        & (pixels[..., 1] - pixels[..., 2] > 15)
    )
    return float(green_pixels.mean())


@logfire.instrument("detect image={image_path}")
def detect(image_path: str | Path) -> DetectionResult:
    """Run object detection on an image file.

    Returns DetectionResult with centroid detections and path to annotated image.
    Falls back gracefully if the model isn't available.
    """
    image_path = Path(image_path)
    result = DetectionResult()

    runner, model_info = _get_runner()
    if runner is None:
        return result

    project = model_info.get("project", {})
    result.model_name = project.get("name", "Edge Impulse model")
    result.labels = model_info.get("model_parameters", {}).get("labels", [])

    # Read and convert image
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        log.error("Failed to read image: %s", image_path)
        return result

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Get features and classify
    features, cropped = runner.get_features_from_image_auto_studio_settings(img_rgb)
    result.frame_height, result.frame_width = cropped.shape[:2]

    t0 = time.perf_counter()
    res = runner.classify(features)
    result.inference_ms = (time.perf_counter() - t0) * 1000

    # Parse bounding boxes
    if "bounding_boxes" in res.get("result", {}):
        for bb in res["result"]["bounding_boxes"]:
            if bb.get("value", 0) < CONFIDENCE_THRESHOLD:
                continue
            result.detections.append(
                Detection(
                    label=bb["label"],
                    confidence=bb["value"],
                    x=bb["x"],
                    y=bb["y"],
                    width=bb["width"],
                    height=bb["height"],
                )
            )

    for detection in result.detections:
        detection.green_context_fraction = _green_context_fraction(cropped, detection)

    # Draw overlay on the cropped/resized image the model actually saw
    annotated = _draw_overlay(cropped, result.detections)

    # Save annotated image
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_name = f"det_{image_path.stem}.jpg"
    out_path = OUTPUT_DIR / out_name
    cv2.imwrite(str(out_path), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
    result.annotated_path = f"detections/{out_name}"

    counts = dict(Counter(d.label for d in result.detections))
    logfire.info(
        "detection result: {count} objects in {inference_ms:.0f}ms",
        count=len(result.detections),
        inference_ms=result.inference_ms,
        counts=counts,
        image=str(image_path.name),
        detections=[
            {"label": d.label, "confidence": round(d.confidence, 3),
             "x": d.x, "y": d.y, "w": d.width, "h": d.height,
             "green_context_fraction": round(d.green_context_fraction, 3)}
            for d in result.detections
        ],
    )

    return result


def shutdown():
    """Stop the runner process (call on app shutdown)."""
    global _runner
    if _runner is not None:
        _runner.stop()
        _runner = None
