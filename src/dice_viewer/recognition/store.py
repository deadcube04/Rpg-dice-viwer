"""Persist exact original uploads in a private MinIO bucket before inference."""

import hashlib
import os
import uuid
from datetime import datetime, timezone

import boto3
from botocore.config import Config

from .operations import secret


class OriginalStore:
    def __init__(self):
        self.bucket = os.environ.get("DICE_IMAGE_BUCKET", "dice-originals-v1")
        self.client = boto3.client(
            "s3", endpoint_url=os.environ.get("S3_ENDPOINT", "http://127.0.0.1:9000"),
            aws_access_key_id=secret("S3_ACCESS_KEY"),
            aws_secret_access_key=secret("S3_SECRET_KEY"),
            region_name=os.environ.get("S3_REGION", "us-east-1"),
            config=Config(signature_version="s3v4", s3={"addressing_style": "path"},
                          connect_timeout=3, read_timeout=5,
                          retries={"total_max_attempts": 2, "mode": "standard"}),
        )
        self.client.head_bucket(Bucket=self.bucket)

    def save(self, body: bytes, content_type: str, request_id: str, model_version: str):
        image_id = uuid.uuid4().hex
        day = datetime.now(timezone.utc).strftime("%Y/%m/%d")
        extension = "jpg" if content_type == "image/jpeg" else "png"
        key = f"originals/{day}/{image_id}.{extension}"
        self.client.put_object(
            Bucket=self.bucket, Key=key, Body=body, ContentType=content_type,
            Metadata={"sha256": hashlib.sha256(body).hexdigest(), "image-id": image_id,
                      "request-id": request_id, "model-version": model_version},
        )
        return image_id
