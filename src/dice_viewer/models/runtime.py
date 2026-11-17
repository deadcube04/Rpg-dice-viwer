"""Explicit device policy; importing components does not require CUDA."""

import random
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class DeviceSelection:
    requested: str
    resolved: str
    reason: str


def select_device(requested: str = "cpu", *, allow_cpu_fallback: bool = True) -> DeviceSelection:
    if requested not in ("cpu", "cuda", "auto"):
        raise ValueError("Device must be cpu, cuda or auto")
    if requested == "cpu":
        return DeviceSelection(requested, "cpu", "explicit_cpu_default")
    if torch.cuda.is_available():
        return DeviceSelection(requested, "cuda:0", "cuda_available")
    if requested == "cuda" and not allow_cpu_fallback:
        raise RuntimeError("CUDA requested but unavailable")
    return DeviceSelection(requested, "cpu", "cuda_unavailable_cpu_fallback")


def seed_components(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # FP32 baseline also disables TF32 approximation where available.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
