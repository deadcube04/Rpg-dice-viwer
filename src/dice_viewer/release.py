"""Record a deployment of the frozen bundle without training artifacts."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from dice_viewer.active_model import ACTIVE_MODEL_TAG, BUNDLE_SHA256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Built Docker image tag or ID")
    parser.add_argument("--output", type=Path, default=Path("runtime/releases.jsonl"))
    args = parser.parse_args()
    inspected = subprocess.run(
        ["docker", "image", "inspect", args.image],
        capture_output=True, text=True, check=True,
    )
    images = json.loads(inspected.stdout)
    if len(images) != 1 or not images[0].get("Id"):
        raise ValueError("Docker did not return exactly one image ID")
    image = images[0]
    labels = image.get("Config", {}).get("Labels") or {}
    if labels.get("io.dice-viewer.model") != ACTIVE_MODEL_TAG:
        raise ValueError("Docker image does not declare the saved model")
    if labels.get("io.dice-viewer.bundle-sha256") != BUNDLE_SHA256:
        raise ValueError("Docker image does not declare the saved bundle checksum")
    target = args.output
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a+", encoding="utf-8") as stream:
        stream.seek(0)
        previous = None
        for line in stream:
            if line.strip():
                previous = json.loads(line)
        release = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model_tag": ACTIVE_MODEL_TAG, "bundle_sha256": BUNDLE_SHA256,
            "image_tag": args.image, "image_id": image["Id"],
            "repo_digests": image.get("RepoDigests", []),
            "rollback_image_id": previous["image_id"] if previous else None,
            "rollback_model_tag": previous["model_tag"] if previous else None,
        }
        stream.write(json.dumps(release) + "\n")
    print(json.dumps(release, indent=2))


if __name__ == "__main__":
    main()
