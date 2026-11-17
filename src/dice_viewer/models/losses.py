"""Per-task cross-entropy with independent masks and an empty-task zero."""

import torch
from torch import nn
from torch.nn import functional as F


def masked_cross_entropy(logits: torch.Tensor, targets: torch.Tensor,
                         mask: torch.Tensor) -> tuple[torch.Tensor, int]:
    if logits.ndim != 2 or targets.shape != (logits.shape[0],) or mask.shape != targets.shape:
        raise ValueError("Expected logits [B,C], targets [B] and mask [B]")
    if targets.dtype != torch.long or mask.dtype != torch.bool:
        raise ValueError("Targets must be int64 and masks boolean")
    count = int(mask.sum().item())
    if count == 0:
        return logits.sum() * 0.0, 0
    return F.cross_entropy(logits[mask], targets[mask]), count


class MaskedMultitaskLoss(nn.Module):
    def __init__(self, type_weight: float = 1.0, value_weight: float = 1.0):
        super().__init__()
        if type_weight < 0 or value_weight < 0:
            raise ValueError("Loss weights must be non-negative")
        self.type_weight, self.value_weight = type_weight, value_weight

    def forward(self, outputs, type_targets, value_targets, type_mask, value_mask):
        type_loss, type_count = masked_cross_entropy(outputs["type_logits"], type_targets, type_mask)
        value_loss, value_count = masked_cross_entropy(outputs["value_logits"], value_targets, value_mask)
        return {"loss": self.type_weight * type_loss + self.value_weight * value_loss,
                "type_loss": type_loss, "value_loss": value_loss,
                "type_count": type_count, "value_count": value_count}
