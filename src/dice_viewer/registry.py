"""Promote the frozen candidate as an immutable BentoML model bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import bentoml
import boto3

from dice_viewer.provenance import git_revision, source_digest


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def promote(artifacts: Path) -> dict[str, str]:
    selection = json.loads((artifacts / "selection.json").read_text(encoding="utf-8"))
    evaluation = json.loads((artifacts / "evaluation.json").read_text(encoding="utf-8"))
    detector = Path(selection["detector_path"])
    detector_run = json.loads((artifacts / "detector" / "run.json").read_text(encoding="utf-8"))
    reader = artifacts / "readers" / selection["selected"]
    metadata = json.loads((reader / "reader.json").read_text(encoding="utf-8"))
    provenance = {"architecture": selection["selected"], "threshold": float(selection["threshold"]),
                  "mlflow_run_id": metadata["mlflow_run_id"], "source_sha256": metadata["source_sha256"],
                  "detector_mlflow_run_id": detector_run["mlflow_run_id"],
                  "split_sha256": metadata["split_sha256"], "code_sha256": source_digest(),
                  "git_revision": git_revision() or "uncommitted", "detector_sha256": digest(detector),
                  "reader_sha256": digest(reader / "reader.safetensors"),
                  "test_joint_accuracy": evaluation["end_to_end"]["joint_accuracy"]}
    with bentoml.models.create("dice_bundle", metadata=provenance) as model:
        folder = Path(model.path)
        shutil.copy2(detector, folder / "detector.pt")
        shutil.copy2(reader / "reader.safetensors", folder / "reader.safetensors")
        shutil.copy2(reader / "reader.json", folder / "reader.json")
        (folder / "threshold.json").write_text(json.dumps({"threshold": provenance["threshold"]}), encoding="utf-8")
        tag = str(model.tag)
    export_path = artifacts / f"{tag.replace(':', '-')}.bentomodel"
    exported = Path(bentoml.models.export_model(tag, str(export_path.resolve())))
    s3 = boto3.client("s3", endpoint_url=os.environ.get("S3_ENDPOINT", "http://127.0.0.1:9000"),
                      aws_access_key_id=os.environ["S3_ACCESS_KEY"],
                      aws_secret_access_key=os.environ["S3_SECRET_KEY"],
                      region_name=os.environ.get("S3_REGION", "us-east-1"))
    s3.upload_file(str(exported), "dice-models", f"{tag}.bentomodel")
    active_model_file = Path(__file__).with_name("active_model.py")
    active_model_file.write_text(f'"""Model tag frozen during promotion."""\n\nACTIVE_MODEL_TAG = {tag!r}\n', encoding="utf-8")
    result = {"model_tag": tag, "git_revision": provenance["git_revision"],
              "code_sha256": provenance["code_sha256"], "source_sha256": provenance["source_sha256"],
              "export_sha256": digest(exported), "s3_key": f"dice-models/{tag}.bentomodel"}
    (artifacts / "promotion.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    promote(args.artifacts)


if __name__ == "__main__":
    main()
