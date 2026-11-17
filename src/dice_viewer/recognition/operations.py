"""Runtime verification, private storage bootstrap and container entry point."""

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

from .package import verify_package


def secret(name):
    file = os.environ.get(name + "_FILE")
    if file:
        return Path(file).read_text(encoding="utf-8").strip()
    return os.environ[name]


def storage_client():
    import boto3
    from botocore.config import Config
    return boto3.client("s3", endpoint_url=os.environ.get("S3_ENDPOINT", "http://minio:9000"),
                        aws_access_key_id=secret("S3_ACCESS_KEY"), aws_secret_access_key=secret("S3_SECRET_KEY"),
                        region_name="us-east-1", config=Config(signature_version="s3v4",
                            s3={"addressing_style": "path"}, connect_timeout=3, read_timeout=5,
                            retries={"total_max_attempts": 2, "mode": "standard"}))


def bootstrap():
    from botocore.exceptions import ClientError, EndpointConnectionError
    bucket = os.environ.get("DICE_IMAGE_BUCKET", "dice-originals-v1")
    client = storage_client()
    deadline = time.monotonic() + 90
    try:
        while True:
            try:
                client.head_bucket(Bucket=bucket)
                break
            except ClientError as exc:
                if exc.response["Error"]["Code"] in ("404", "NoSuchBucket"):
                    client.create_bucket(Bucket=bucket)
                    break
                if time.monotonic() >= deadline:
                    raise
            except EndpointConnectionError:
                if time.monotonic() >= deadline:
                    raise
            time.sleep(1)
        for method, absent in ((client.get_bucket_policy, "NoSuchBucketPolicy"),
                               (client.get_bucket_lifecycle_configuration, "NoSuchLifecycleConfiguration")):
            try:
                method(Bucket=bucket)
            except ClientError as exc:
                if exc.response["Error"]["Code"] == absent:
                    continue
                raise
            raise RuntimeError("Unexpected bucket access/lifecycle policy; inspect before starting")
        print("Private original-image bucket ready; no lifecycle expiration configured", flush=True)
    finally:
        client.close()


def warmup_and_record(engine, root, device):
    import torch
    from PIL import Image
    from ..training.timing import synchronize
    from .engine import DecodedImage
    started = time.perf_counter()
    with DecodedImage(Image.new("RGB", (640, 640))) as source:
        engine.infer(source)
    with torch.inference_mode():
        engine.classifier(torch.zeros((1, 3, 224, 224), dtype=torch.float32, device=device.resolved))
    synchronize(device.resolved)
    record = {
        "ready_at_utc": datetime.now(timezone.utc).isoformat(), "release_sha256": os.environ["DICE_BUNDLE_SHA256"],
        "release_id": json.loads((root / "RELEASE.json").read_text())["release_id"],
        "requested_device": device.requested, "resolved_device": device.resolved, "device_reason": device.reason,
        "warmup_seconds": time.perf_counter() - started, "python": platform.python_version(),
        "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if device.resolved.startswith("cuda") else None,
    }
    state = Path(os.environ.get("DICE_STATE_DIR", "/state"))
    state.mkdir(parents=True, exist_ok=True)
    temporary = state / "startup.json.tmp"
    temporary.write_text(json.dumps(record, indent=2), encoding="utf-8")
    temporary.replace(state / "startup.json")
    print(json.dumps({"warmup_completed": record}), flush=True)


def health():
    with urlopen("http://127.0.0.1:3000/readyz", timeout=3) as response:
        if response.status != 200:
            raise RuntimeError("BentoML is not ready")
    state = json.loads((Path(os.environ.get("DICE_STATE_DIR", "/state")) / "startup.json").read_text())
    if state["release_sha256"] != os.environ["DICE_BUNDLE_SHA256"]:
        raise RuntimeError("Readiness belongs to another release")
    client = storage_client()
    try:
        client.head_bucket(Bucket=os.environ.get("DICE_IMAGE_BUCKET", "dice-originals-v1"))
    finally:
        client.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("serve", "verify", "bootstrap", "health", "identify"))
    args = parser.parse_args()
    root = Path(os.environ.get("DICE_PROJECT_ROOT", "/opt/dice")).resolve()
    if args.command == "bootstrap":
        bootstrap()
    elif args.command == "health":
        health()
    elif args.command == "identify":
        print((Path(os.environ.get("DICE_STATE_DIR", "/state")) / "startup.json").read_text())
    else:
        data = verify_package(root, os.environ["DICE_BUNDLE_SHA256"])
        print(f"Verified release {data['release_id']} with {len(data['files'])} files", flush=True)
        if args.command == "serve":
            os.chdir(root)
            os.execv(sys.executable, [sys.executable, "-m", "bentoml", "serve",
                                     "recognition_service:DiceRecognitionService", "--host", "0.0.0.0", "--port", "3000"])


if __name__ == "__main__":
    main()
