"""Initialize single-class YOLO26n with compatible official pretrained tensors."""

from pathlib import Path

from ultralytics import YOLO

from ..data.prepare import sha256


def build_detector(architecture: Path, weights: Path, expected_sha256: str) -> YOLO:
    if sha256(weights) != expected_sha256:
        raise ValueError("YOLO26n pretrained weights hash mismatch")
    model = YOLO(str(architecture), task="detect", verbose=False)
    if model.model.model[-1].nc != 1:
        raise ValueError("Detector architecture must contain a single class")
    model.model.names = {0: "die"}
    model.load(str(weights))
    model.model.names = {0: "die"}
    head = model.model.model[-1]
    if getattr(head, "one2one_cv2", None) is None:
        raise ValueError("YOLO26 one-to-one detection branch is required")
    # In 8.4.173 the architecture creates both branches, but the selector defaults
    # to one-to-many until explicitly set (or selected by AutoBackend).
    head.end2end = True
    return model


def predict_components(model: YOLO, image, device: str = "cpu"):
    """Pinned Ultralytics 8.4.173: nms=False explicitly selects one-to-one inference."""
    return model.predict(source=image, imgsz=640, device=device, quantize=32,
                         nms=False, conf=.25, max_det=64, verbose=False, save=False)
