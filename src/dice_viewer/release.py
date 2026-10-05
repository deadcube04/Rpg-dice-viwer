"""Record an immutable Bento container image and its rollback predecessor."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Built local image tag")
    parser.add_argument("--bento", required=True, help="Immutable Bento tag")
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    promotion = json.loads((args.artifacts / "promotion.json").read_text(encoding="utf-8"))
    inspected = subprocess.run(["docker", "image", "inspect", args.image, "--format", "{{json .}}"],
                               capture_output=True, text=True, check=True)
    image = json.loads(inspected.stdout)
    if not image.get("Id"):
        raise ValueError("Docker did not return an image ID")
    target = args.artifacts / "releases.jsonl"
    previous = None
    if target.is_file():
        lines = [line for line in target.read_text(encoding="utf-8").splitlines() if line.strip()]
        if lines:
            previous = json.loads(lines[-1])
    release = {"created_at": datetime.now(timezone.utc).isoformat(), "model_tag": promotion["model_tag"],
               "bento_tag": args.bento, "image_tag": args.image, "image_id": image["Id"],
               "repo_digests": image.get("RepoDigests", []),
               "rollback_image_id": previous["image_id"] if previous else None,
               "rollback_model_tag": previous["model_tag"] if previous else None}
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(release) + "\n")
    print(json.dumps(release, indent=2))


if __name__ == "__main__":
    main()
