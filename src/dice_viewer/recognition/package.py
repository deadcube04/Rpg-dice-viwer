"""Create an immutable, self-contained release from explicitly approved artifacts."""

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def safe_relative(name):
    path = Path(name.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or ":" in name:
        raise ValueError("Release paths must remain inside the package")
    return path


def verify_package(root: Path, expected: str):
    manifest = root / "RELEASE.json"
    if digest(manifest) != expected:
        raise ValueError("Release manifest does not match the deployment pin")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if data["model_approved"] is not True or data["integrated_flow_approved"] is not True:
        raise ValueError("Release requires model and integrated-flow approvals")
    for name, expected_hash in data["files"].items():
        path = (root / safe_relative(name)).resolve()
        if not path.is_relative_to(root.resolve()) or digest(path) != expected_hash:
            raise ValueError(f"Release file changed: {name}")
    return data


def create(root: Path, release_id: str):
    if not release_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in release_id):
        raise ValueError("Invalid release ID")
    model_approval = root / "reports/approvals/model-initial-v1-20261117.json"
    flow_approval = root / "reports/approvals/stage5-flow-20261117.json"
    model = json.loads(model_approval.read_text(encoding="utf-8"))
    flow = json.loads(flow_approval.read_text(encoding="utf-8"))
    if model["approved"] is not True or flow["approved"] is not True or flow["model_id"] != model["candidate_id"]:
        raise ValueError("Explicit compatible approvals are required")
    stage5_index = root / "reports/stage5-20261117/DELIVERY_CHECKSUMS.json"
    if digest(stage5_index) != flow["stage5_delivery_manifest_sha256"]:
        raise ValueError("Approved integrated-flow record changed")
    candidate = root / "artifacts" / model["candidate_id"] / "candidate.json"
    if digest(candidate) != model["evaluated_candidate_manifest_sha256"]:
        raise ValueError("Approved candidate manifest changed")
    evaluation = root / "reports/evaluation" / model["evaluation_run_id"]
    if digest(evaluation / "protocol.json") != model["evaluation_protocol_sha256"]:
        raise ValueError("Approved evaluation protocol changed")
    protocol = json.loads((evaluation / "protocol.json").read_text(encoding="utf-8"))
    files = {model_approval, flow_approval, candidate, stage5_index}
    for component, item in model["weights"].items():
        path = root / safe_relative(item["path"])
        if digest(path) != item["sha256"]:
            raise ValueError(f"Approved checkpoint changed: {component}")
        files.add(path)
    for name, expected in protocol["supporting_hashes"].items():
        path = root / safe_relative(name)
        if digest(path) != expected:
            raise ValueError(f"Frozen scientific dependency changed: {name}")
        files.add(path)
    e2_index = json.loads((evaluation / "DELIVERY_CHECKSUMS.json").read_text(encoding="utf-8"))
    for name, expected in e2_index.items():
        path = root / safe_relative(name)
        if digest(path) != expected:
            raise ValueError(f"Frozen evaluation delivery changed: {name}")
        files.add(path)
    files.add(evaluation / "DELIVERY_CHECKSUMS.json")
    pretrained_report = root / "reports/stage3-20261117/pretrained-weights.json"
    pretrained = json.loads(pretrained_report.read_text(encoding="utf-8"))["detector"]
    pretrained_path = root / safe_relative(pretrained["path"])
    if digest(pretrained_path) != pretrained["sha256"]:
        raise ValueError("Detector initialization artifact changed")
    files.update((pretrained_report, pretrained_path, root / "recognition_service.py"))
    files.update((root / "src/dice_viewer").rglob("*.py"))
    files.update(p for p in (root / "deploy/v1").rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    # Only maps are copied from the dataset. No training/validation images,
    # audit databases, credentials or optimizer states enter the runtime bundle.
    target = root / "releases" / release_id
    target.mkdir(parents=True, exist_ok=False)
    try:
        index = {}
        for source in sorted(files):
            relative = source.relative_to(root)
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            index[relative.as_posix()] = digest(destination)
        manifest = {
            "release_id": release_id, "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "candidate_id": model["candidate_id"], "evaluation_run_id": model["evaluation_run_id"],
            "model_approved": True, "integrated_flow_approved": True,
            "model_approval_sha256": digest(model_approval), "flow_approval_sha256": digest(flow_approval),
            "files": index, "calibration": "none", "retraining": "manual_future_execution",
            "runtime": {"python": "3.11.17", "platform": "linux/amd64", "precision": "FP32",
                        "torch": "2.14.1+cu126", "torchvision": "0.29.1+cu126",
                        "default_device": "cpu", "optional_device": "cuda"},
        }
        (target / "RELEASE.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        sha = digest(target / "RELEASE.json")
        verify_package(target, sha)
    except Exception:
        # Preserve an incomplete directory for diagnosis; never silently reuse it.
        raise
    print(json.dumps({"release_id": release_id, "directory": str(target), "manifest_sha256": sha,
                      "files": len(index), "bytes": sum(p.stat().st_size for p in target.rglob("*") if p.is_file())}, indent=2))
    return target, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release_id")
    args = parser.parse_args()
    create(Path(__file__).resolve().parents[3], args.release_id)


if __name__ == "__main__":
    main()
