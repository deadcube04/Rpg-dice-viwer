"""Launch local V1 validation without changing the historical demo configuration."""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=13000)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--local-minio-env", type=Path,
                        help="Explicit credentials file of the isolated stage5 MinIO")
    parser.add_argument("--bootstrap-bucket", action="store_true",
                        help="Create the configured private bucket if absent")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    os.environ["DICE_PROJECT_ROOT"] = str(root)
    os.environ["DICE_DEVICE"] = args.device
    os.environ.setdefault("BENTOML_HOME", str(root / ".cache/bentoml-v1"))
    os.environ.setdefault("YOLO_CONFIG_DIR", str(root / ".cache/ultralytics"))
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    if args.local_minio_env:
        values = dict(line.split("=", 1) for line in args.local_minio_env.read_text().splitlines()
                      if line and not line.startswith("#"))
        os.environ["S3_ACCESS_KEY"] = values["MINIO_ROOT_USER"]
        os.environ["S3_SECRET_KEY"] = values["MINIO_ROOT_PASSWORD"]
        os.environ.setdefault("S3_ENDPOINT", "http://127.0.0.1:19000")
        os.environ.setdefault("DICE_IMAGE_BUCKET", "dice-stage5-originals")
    if args.bootstrap_bucket:
        import boto3
        from botocore.config import Config
        from botocore.exceptions import ClientError
        client = boto3.client("s3", endpoint_url=os.environ["S3_ENDPOINT"],
                              aws_access_key_id=os.environ["S3_ACCESS_KEY"],
                              aws_secret_access_key=os.environ["S3_SECRET_KEY"],
                              region_name="us-east-1", config=Config(signature_version="s3v4",
                                  s3={"addressing_style": "path"}, connect_timeout=3, read_timeout=5))
        bucket = os.environ.get("DICE_IMAGE_BUCKET", "dice-originals-v1")
        try:
            client.head_bucket(Bucket=bucket)
        except ClientError as exc:
            if exc.response["Error"]["Code"] not in ("404", "NoSuchBucket"):
                raise
            client.create_bucket(Bucket=bucket)
        finally:
            client.close()
    return subprocess.call([sys.executable, "-m", "bentoml", "serve",
                            "recognition_service:DiceRecognitionService", "--host", "127.0.0.1",
                            "--port", str(args.port)], cwd=root)


if __name__ == "__main__":
    raise SystemExit(main())
