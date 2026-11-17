"""Deterministic body crops compatible with the approved Stage 2 artifacts."""

import numpy as np
import torch
from PIL import Image, ImageOps

from ..data.prepare import make_crop

VERSION = "body-margin10-letterbox224-imagenet-v1"
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def tensor_from_prepared_crop(image: Image.Image) -> torch.Tensor:
    """Normalize an already padded crop; do not apply geometry a second time."""
    if image.size != (224, 224):
        raise ValueError("Expected a prepared 224x224 crop")
    pixels = np.array(image.convert("RGB"), dtype=np.float32, copy=True) / 255.0
    tensor = torch.from_numpy(pixels).permute(2, 0, 1)
    mean = torch.tensor(MEAN, dtype=torch.float32)[:, None, None]
    std = torch.tensor(STD, dtype=torch.float32)[:, None, None]
    return (tensor - mean) / std


def crop_body(image: Image.Image, bbox_xyxy) -> tuple[torch.Tensor, dict]:
    """Box coordinates must refer to pixels AFTER EXIF orientation."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    x1, y1, x2, y2 = bbox_xyxy
    if not (0 <= x1 < x2 <= image.width and 0 <= y1 < y2 <= image.height):
        raise ValueError("Body bounding box outside oriented image")
    prepared, geometry = make_crop(image, bbox_xyxy, margin=.1, size=224)
    return tensor_from_prepared_crop(prepared), {**geometry, "version": VERSION}
