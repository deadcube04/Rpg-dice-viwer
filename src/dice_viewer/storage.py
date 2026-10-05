"""Local feedback ledger with images in the configured MinIO bucket."""

from __future__ import annotations

import io
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3
from botocore.client import Config


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class FeedbackStore:
    def __init__(self) -> None:
        self.database = Path(os.environ.get("DICE_SQLITE_PATH", "runtime/feedback.db"))
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self.bucket = os.environ.get("DICE_IMAGE_BUCKET", "dice-feedback")
        self.s3 = boto3.client(
            "s3", endpoint_url=os.environ.get("S3_ENDPOINT", "http://127.0.0.1:9000"),
            aws_access_key_id=os.environ["S3_ACCESS_KEY"], aws_secret_access_key=os.environ["S3_SECRET_KEY"],
            region_name=os.environ.get("S3_REGION", "us-east-1"),
            config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
        )
        self._create_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _create_schema(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("""CREATE TABLE IF NOT EXISTS predictions (
                id TEXT PRIMARY KEY, image_key TEXT NOT NULL, image_sha256 TEXT NOT NULL,
                model_version TEXT NOT NULL, status TEXT NOT NULL, reason TEXT,
                sides INTEGER, value INTEGER, confidence REAL,
                created_at TEXT NOT NULL
            )""")
            connection.execute("""CREATE TABLE IF NOT EXISTS feedback (
                id TEXT PRIMARY KEY, prediction_id TEXT NOT NULL REFERENCES predictions(id),
                confirmed INTEGER NOT NULL, sides INTEGER NOT NULL, value INTEGER NOT NULL,
                created_at TEXT NOT NULL
            )""")

    def ready(self) -> bool:
        with self._connect() as connection:
            connection.execute("SELECT 1").fetchone()
        self.s3.head_bucket(Bucket=self.bucket)
        return True

    def save_prediction(self, image_bytes: bytes, image_sha256: str, extension: str,
                        model_version: str, status: str, reason: str | None,
                        sides: int | None, value: int | None, confidence: float | None) -> str:
        prediction_id = str(uuid.uuid4())
        key = f"predictions/{prediction_id}{extension}"
        self.s3.put_object(Bucket=self.bucket, Key=key, Body=io.BytesIO(image_bytes),
                           ContentType="image/jpeg" if extension == ".jpg" else "image/png",
                           Metadata={"sha256": image_sha256, "model-version": model_version})
        try:
            with self._connect() as connection:
                connection.execute("""INSERT INTO predictions
                    (id,image_key,image_sha256,model_version,status,reason,sides,value,confidence,created_at)
                    VALUES (?,?,?,?,?,?,?,?,?,?)""",
                    (prediction_id, key, image_sha256, model_version, status, reason, sides, value, confidence, utc_now()))
        except Exception:
            self.s3.delete_object(Bucket=self.bucket, Key=key)
            raise
        return prediction_id

    def add_feedback(self, prediction_id: str, confirmed: bool, sides: int, value: int) -> dict[str, Any]:
        feedback_id = str(uuid.uuid4())
        with self._connect() as connection:
            prediction = connection.execute("SELECT id,sides,value FROM predictions WHERE id=?", (prediction_id,)).fetchone()
            if prediction is None:
                raise KeyError(prediction_id)
            if confirmed and (prediction["sides"], prediction["value"]) != (sides, value):
                raise ValueError("Confirmed feedback must match the prediction")
            connection.execute("INSERT INTO feedback (id,prediction_id,confirmed,sides,value,created_at) VALUES (?,?,?,?,?,?)",
                               (feedback_id, prediction_id, int(confirmed), sides, value, utc_now()))
        return {"feedback_id": feedback_id, "prediction_id": prediction_id, "confirmed": confirmed,
                "sides": sides, "value": value}
