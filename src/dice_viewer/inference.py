"""Shared detector-to-reader inference path for evaluation and serving."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

from dice_viewer.reader import classify, load_reader


@dataclass(frozen=True)
class DiePrediction:
    box: tuple[int, int, int, int]
    sides: int
    value: int
    confidence: float
    detector_confidence: float | None
    type_confidence: float
    value_confidence: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def crop_xyxy(image: Image.Image, box: tuple[int, int, int, int], padding: float = 0.15) -> Image.Image:
    left, top, right, bottom = box
    width, height = right - left, bottom - top
    bounds = (max(0, round(left - width * padding)), max(0, round(top - height * padding)),
              min(image.width, round(right + width * padding)), min(image.height, round(bottom + height * padding)))
    return image.crop(bounds)


class InferenceEngine:
    def __init__(self, detector_path: Path, reader_folder: Path, require_gpu: bool = True) -> None:
        if require_gpu and not torch.cuda.is_available():
            raise RuntimeError("GPU is required by this deployment")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.detector = YOLO(str(detector_path))
        self.reader, self.reader_metadata = load_reader(reader_folder, self.device)

    @torch.inference_mode()
    def predict_single(self, image: Image.Image) -> DiePrediction:
        """Read a supplied photo of one die without relying on object detection."""
        sides, value, type_conf, value_conf = classify(self.reader, image, self.device)
        return DiePrediction(
            (0, 0, image.width, image.height), sides, value,
            type_conf * value_conf, None, type_conf, value_conf,
        )

    @torch.inference_mode()
    def predict(self, image: Image.Image) -> list[DiePrediction]:
        rgb = image.convert("RGB")
        results = self.detector.predict(np.asarray(rgb), imgsz=640, conf=0.25, device=str(self.device), verbose=False)
        boxes = results[0].boxes
        if boxes is None:
            return []
        output: list[DiePrediction] = []
        for coordinates, confidence in zip(boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist(), strict=True):
            box = tuple(int(round(value)) for value in coordinates)
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            sides, value, type_conf, value_conf = classify(self.reader, crop_xyxy(rgb, box), self.device)
            score = float(confidence) * type_conf * value_conf
            output.append(DiePrediction(box, sides, value, score, float(confidence), type_conf, value_conf))
        return output
