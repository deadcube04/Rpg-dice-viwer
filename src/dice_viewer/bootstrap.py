"""Create the local S3 buckets needed by DVC, MLflow, and feedback."""

from __future__ import annotations

import os
import time

import boto3
from botocore.exceptions import ClientError, EndpointConnectionError


BUCKETS = ("dice-datasets", "dice-mlflow-artifacts", "dice-models", "dice-feedback")


def main() -> None:
    client = boto3.client(
        "s3",
        endpoint_url=os.environ.get("S3_ENDPOINT", "http://127.0.0.1:9000"),
        aws_access_key_id=os.environ["S3_ACCESS_KEY"],
        aws_secret_access_key=os.environ["S3_SECRET_KEY"],
        region_name=os.environ.get("S3_REGION", "us-east-1"),
    )
    for attempt in range(30):
        try:
            client.list_buckets()
            break
        except EndpointConnectionError:
            if attempt == 29:
                raise
            time.sleep(1)
    for bucket in BUCKETS:
        try:
            client.head_bucket(Bucket=bucket)
        except ClientError as exc:
            if exc.response["ResponseMetadata"]["HTTPStatusCode"] != 404:
                raise
            client.create_bucket(Bucket=bucket)
        print(bucket)


if __name__ == "__main__":
    main()
