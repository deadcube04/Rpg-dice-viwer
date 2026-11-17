"""Reload the explicitly versioned initial component states, without pickle."""

import json
from pathlib import Path

from safetensors.torch import load_file

from ..data.prepare import sha256


def load_initial_state(model, directory: Path, component: str, label_maps: Path, architecture: Path) -> str:
    names = {"classifier": "resnet18-multihead.safetensors", "detector": "yolo26n-die.safetensors"}
    name = names[component]
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "initialization_only_not_trained_not_approved_for_serving":
        raise ValueError("Expected an initial component artifact")
    if sha256(label_maps) != manifest["label_maps_sha256"] or sha256(architecture) != manifest["architecture_sha256"]:
        raise ValueError("Initial artifact label map or architecture mismatch")
    path = directory / name
    digest = sha256(path)
    if digest != manifest["states"][name]:
        raise ValueError("Initial state hash mismatch")
    model.load_state_dict(load_file(str(path), device="cpu"), strict=True)
    return digest
