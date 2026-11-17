"""Adapter of the E2 cascade for already decoded, EXIF-oriented images."""

import json
import os
from importlib.metadata import version
from pathlib import Path

from ..data.prepare import sha256
from ..evaluation.cascade import FrozenCascade
from ..models.runtime import seed_components, select_device


def load_approved(root: Path, requested_device: str):
    if (root / "RELEASE.json").exists():
        from .package import verify_package
        verify_package(root, os.environ["DICE_BUNDLE_SHA256"])
    approval = json.loads((root / "reports/approvals/model-initial-v1-20261117.json").read_text())
    if approval.get("approved") is not True or approval["status"] != "approved_initial_baseline":
        raise ValueError("Explicit initial-model approval is required")
    candidate = root / "artifacts" / approval["candidate_id"] / "candidate.json"
    protocol_path = root / "reports/evaluation" / approval["evaluation_run_id"] / "protocol.json"
    if sha256(candidate) != approval["evaluated_candidate_manifest_sha256"]:
        raise ValueError("Approved candidate manifest changed")
    if sha256(protocol_path) != approval["evaluation_protocol_sha256"]:
        raise ValueError("Approved evaluation protocol changed")
    protocol = json.loads(protocol_path.read_text())
    frozen_cascade = protocol_path.parent / "provenance/cascade.py"
    if sha256(root / "src/dice_viewer/evaluation/cascade.py") != sha256(frozen_cascade):
        raise ValueError("Evaluated cascade source changed")
    for name, expected in protocol["versions"].items():
        if version(name) != expected:
            raise ValueError(f"Frozen inference version differs: {name}")
    for name, expected in protocol["supporting_hashes"].items():
        path = root.joinpath(*name.replace("\\", "/").split("/"))
        if sha256(path) != expected:
            raise ValueError(f"Frozen inference dependency differs: {name}")
    for component, item in approval["weights"].items():
        path = root.joinpath(*item["path"].replace("\\", "/").split("/"))
        if sha256(path) != item["sha256"]:
            raise ValueError(f"Approved weights changed: {component}")
    import torch
    torch.set_num_threads(4)
    seed_components(42)
    device = select_device(requested_device)
    engine = FrozenCascade(root, protocol["configuration"], device.resolved)
    return engine, protocol, device


class DecodedImage:
    """File-like object recognized by the E2 adapter, without touching its source."""

    def __init__(self, image):
        # E2 starts by Image.open(path). A lossless, oriented PNG preserves exact
        # RGB pixels, and keeps the evaluated detection/classification code intact.
        import io
        self.stream = io.BytesIO()
        image.save(self.stream, format="PNG")
        self.stream.seek(0)

    def __enter__(self):
        return self.stream

    def __exit__(self, *args):
        self.stream.close()
