"""Inference-only reader architecture required by the frozen bundle."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import load_file
from torch import nn
from torchvision import models, transforms

from dice_viewer import SIDES

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class DieReader(nn.Module):
    def __init__(self, architecture: str) -> None:
        super().__init__()
        if architecture != "efficientnet_b0":
            raise ValueError(f"Unsupported architecture: {architecture}")
        backbone = models.efficientnet_b0(weights=None)
        feature_count = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()
        self.architecture = architecture
        self.backbone = backbone
        self.type_head = nn.Linear(feature_count, len(SIDES))
        self.value_head = nn.Sequential(nn.Linear(feature_count + len(SIDES), 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 20))

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(images)
        type_logits = self.type_head(features)
        type_ids = type_logits.argmax(dim=1)
        type_one_hot = nn.functional.one_hot(type_ids, len(SIDES)).to(dtype=features.dtype)
        value_logits = self.value_head(torch.cat((features, type_one_hot), dim=1))
        limits = torch.tensor(SIDES, device=value_logits.device)[type_ids]
        labels = torch.arange(1, 21, device=value_logits.device).unsqueeze(0)
        value_logits = value_logits.masked_fill(labels > limits.unsqueeze(1), -1e4)
        return type_logits, value_logits


IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(MEAN, STD),
])


def load_reader(folder: Path, device: torch.device) -> tuple[DieReader, dict[str, object]]:
    metadata = json.loads((folder / "reader.json").read_text(encoding="utf-8"))
    model = DieReader(str(metadata["architecture"]))
    model.load_state_dict(load_file(folder / "reader.safetensors", device="cpu"))
    model.to(device).eval()
    return model, metadata


@torch.inference_mode()
def classify(model: DieReader, image: Image.Image, device: torch.device) -> tuple[int, int, float, float]:
    tensor = IMAGE_TRANSFORM(image.convert("RGB")).unsqueeze(0).to(device)
    type_logits, value_logits = model(tensor)
    type_probabilities = type_logits.softmax(dim=1)[0]
    type_index = int(type_probabilities.argmax())
    value_probabilities = value_logits.softmax(dim=1)[0]
    value_index = int(value_probabilities.argmax())
    return SIDES[type_index], value_index + 1, float(type_probabilities[type_index]), float(value_probabilities[value_index])
