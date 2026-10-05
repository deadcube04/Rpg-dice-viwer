"""Small conditional classifiers for die type and upper-face value."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image, ImageOps
from safetensors.torch import load_file, save_file
from torch import nn
from torch.utils.data import Dataset
from torchvision import models, transforms

from dice_viewer import SIDES

ARCHITECTURES = ("mobilenet_v3_small", "efficientnet_b0")
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class DieReader(nn.Module):
    def __init__(self, architecture: str, pretrained: bool = False) -> None:
        super().__init__()
        if architecture == "mobilenet_v3_small":
            weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            backbone = models.mobilenet_v3_small(weights=weights)
            feature_count = backbone.classifier[0].in_features
            backbone.classifier = nn.Identity()
        elif architecture == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            backbone = models.efficientnet_b0(weights=weights)
            feature_count = backbone.classifier[1].in_features
            backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")
        self.architecture = architecture
        self.backbone = backbone
        self.type_head = nn.Linear(feature_count, len(SIDES))
        self.value_head = nn.Sequential(nn.Linear(feature_count + len(SIDES), 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 20))

    def forward(self, images: torch.Tensor, type_ids: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(images)
        type_logits = self.type_head(features)
        if type_ids is None:
            type_ids = type_logits.argmax(dim=1)
        type_one_hot = nn.functional.one_hot(type_ids, len(SIDES)).to(dtype=features.dtype)
        value_logits = self.value_head(torch.cat((features, type_one_hot), dim=1))
        limits = torch.tensor(SIDES, device=value_logits.device)[type_ids]
        labels = torch.arange(1, 21, device=value_logits.device).unsqueeze(0)
        value_logits = value_logits.masked_fill(labels > limits.unsqueeze(1), -1e4)
        return type_logits, value_logits


def image_transform(training: bool) -> transforms.Compose:
    steps: list[transforms.Transform] = [transforms.Resize((224, 224))]
    if training:
        steps.extend((transforms.RandomRotation(180), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15)))
    steps.extend((transforms.ToTensor(), transforms.Normalize(MEAN, STD)))
    return transforms.Compose(steps)


class CropDataset(Dataset[tuple[torch.Tensor, int, int]]):
    def __init__(self, processed: Path, split: str, training: bool = False) -> None:
        self.processed = processed
        self.rows = [json.loads(line) for line in (processed / "reader_manifest.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        self.rows = [row for row in self.rows if row["split"] == split]
        if not self.rows:
            raise ValueError(f"No trusted crops in split {split}")
        self.transform = image_transform(training)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        row = self.rows[index]
        with Image.open(self.processed / row["path"]) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        return self.transform(image), SIDES.index(int(row["type"][1:])), int(row["value"]) - 1


def save_reader(model: DieReader, folder: Path, metadata: dict[str, object]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    save_file({name: tensor.detach().cpu().contiguous() for name, tensor in model.state_dict().items()}, folder / "reader.safetensors")
    (folder / "reader.json").write_text(json.dumps({"architecture": model.architecture, **metadata}, indent=2, sort_keys=True), encoding="utf-8")


def load_reader(folder: Path, device: torch.device) -> tuple[DieReader, dict[str, object]]:
    metadata = json.loads((folder / "reader.json").read_text(encoding="utf-8"))
    model = DieReader(str(metadata["architecture"]), pretrained=False)
    model.load_state_dict(load_file(folder / "reader.safetensors", device="cpu"))
    model.to(device).eval()
    return model, metadata


@torch.inference_mode()
def classify(model: DieReader, image: Image.Image, device: torch.device) -> tuple[int, int, float, float]:
    tensor = image_transform(False)(image.convert("RGB")).unsqueeze(0).to(device)
    type_logits, value_logits = model(tensor)
    type_probabilities = type_logits.softmax(dim=1)[0]
    type_index = int(type_probabilities.argmax())
    value_probabilities = value_logits.softmax(dim=1)[0]
    value_index = int(value_probabilities.argmax())
    return SIDES[type_index], value_index + 1, float(type_probabilities[type_index]), float(value_probabilities[value_index])
