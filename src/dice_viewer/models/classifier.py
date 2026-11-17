"""ResNet18 backbone with independent type and global-token linear heads."""

from pathlib import Path

import torch
from torch import nn
from torchvision.models import resnet18

from ..data.prepare import sha256
from .labels import LabelMaps


class MultiheadResNet18(nn.Module):
    def __init__(self, labels: LabelMaps, weights_path: Path | None = None,
                 weights_sha256: str | None = None):
        super().__init__()
        self.labels = labels
        self.backbone = resnet18(weights=None)
        if weights_path is not None:
            if weights_sha256 is None or sha256(weights_path) != weights_sha256:
                raise ValueError("ResNet18 pretrained weights hash mismatch")
            self.backbone.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
        features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.type_head = nn.Linear(features, len(labels.types))
        self.value_head = nn.Linear(features, len(labels.tokens))
        self.backbone_frozen = False

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(images)
        return {"type_logits": self.type_head(features), "value_logits": self.value_head(features)}

    def freeze_backbone(self, frozen: bool = True) -> None:
        self.backbone_frozen = frozen
        self.backbone.requires_grad_(not frozen)
        self.backbone.train(self.training and not frozen)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.backbone_frozen:
            self.backbone.eval()  # Also keep BatchNorm running statistics fixed.
        return self
