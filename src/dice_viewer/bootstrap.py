"""Initialize only the feedback bucket used by the serving stack."""

from __future__ import annotations

import os
import time

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError, EndpointConnectionError


def main() -> None:
    client = boto3.client(
        "s3", endpoint_url=os.environ.get("S3_ENDPOINT", "http://127.0.0.1:9000"),
        aws_access_key_id=os.environ["S3_ACCESS_KEY"],
        aws_secret_access_key=os.environ["S3_SECRET_KEY"],
        region_name=os.environ.get("S3_REGION", "us-east-1"),
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"},
                      connect_timeout=2, read_timeout=3, retries={"total_max_attempts": 1}),
    )
    bucket = os.environ.get("DICE_IMAGE_BUCKET", "dice-feedback")
    for attempt in range(30):
        try:
            client.head_bucket(Bucket=bucket)
            break
        except EndpointConnectionError:
            if attempt == 29:
                raise
            time.sleep(1)
        except ClientError as exc:
            if exc.response["ResponseMetadata"]["HTTPStatusCode"] != 404:
                raise
            try:
                client.create_bucket(Bucket=bucket)
            except ClientError as creation_error:
                if creation_error.response["Error"]["Code"] != "BucketAlreadyOwnedByYou":
                    raise
            break
    print(f"Feedback bucket ready: {bucket}")


if __name__ == "__main__":
    main()
