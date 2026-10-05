"""BentoML service with FastAPI upload, review, feedback, and health routes."""

from __future__ import annotations

import hashlib
import io
import json
import os
import time
from pathlib import Path
from typing import Literal

import bentoml
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image, ImageOps, UnidentifiedImageError
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field

from dice_viewer import SIDES
from dice_viewer.active_model import ACTIVE_MODEL_TAG
from dice_viewer.inference import InferenceEngine
from dice_viewer.storage import FeedbackStore

MAX_IMAGE_BYTES = 10 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 25_000_000
REQUESTS = Counter("dice_prediction_requests_total", "Prediction outcomes", ["status", "reason"])
FEEDBACK = Counter("dice_feedback_total", "Human feedback events", ["confirmed"])
LATENCY = Histogram("dice_prediction_seconds", "Full prediction request duration")
INFERENCE_LATENCY = Histogram("dice_inference_seconds", "Model inference duration")
CONFIDENCE = Histogram("dice_confidence", "Confidence for single-die predictions", buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0))
STORAGE_ERRORS = Counter("dice_storage_errors_total", "Prediction or feedback storage errors")
ACTIVE_MODEL = Gauge("dice_active_model", "Active model tag", ["version"])
app = FastAPI(title="RPG Die Recognition", version="1.0.0")


class FeedbackInput(BaseModel):
    confirmed: bool
    sides: Literal[6, 8, 10, 12, 20]
    value: int = Field(ge=1, le=20)


@bentoml.service(resources={"gpu": 1}, traffic={"timeout": 15})
@bentoml.asgi_app(app, path="/")
class DiceService:
    bundle = bentoml.models.BentoModel(ACTIVE_MODEL_TAG)

    def __init__(self) -> None:
        self.model_tag = ACTIVE_MODEL_TAG
        if os.environ.get("DICE_MODEL_TAG", ACTIVE_MODEL_TAG) != ACTIVE_MODEL_TAG:
            raise RuntimeError("Configured model tag differs from the Bento bundle")
        folder = Path(self.bundle.path)
        self.threshold = float(json.loads((folder / "threshold.json").read_text(encoding="utf-8"))["threshold"])
        self.engine = InferenceEngine(folder / "detector.pt", folder, require_gpu=True)
        self.store = FeedbackStore()
        ACTIVE_MODEL.labels(version=self.model_tag).set(1)

    @app.get("/health")
    def health(self) -> dict[str, object]:
        try:
            storage_ready = self.store.ready()
        except Exception as exc:
            raise HTTPException(status_code=503, detail="storage_unavailable") from exc
        if not torch.cuda.is_available():
            raise HTTPException(status_code=503, detail="gpu_unavailable")
        return {"status": "ready", "model_version": self.model_tag, "storage_ready": storage_ready, "gpu": torch.cuda.get_device_name(0)}

    @app.post("/v1/predictions")
    async def predict(self, file: UploadFile = File(...)) -> dict[str, object]:
        started = time.perf_counter()
        if file.content_type not in {"image/jpeg", "image/png"}:
            raise HTTPException(status_code=422, detail="unsupported_image_type")
        body = await file.read(MAX_IMAGE_BYTES + 1)
        if not body or len(body) > MAX_IMAGE_BYTES:
            raise HTTPException(status_code=422, detail="invalid_image_size")
        try:
            with Image.open(io.BytesIO(body)) as opened:
                if opened.format not in {"JPEG", "PNG"}:
                    raise HTTPException(status_code=422, detail="unsupported_image_type")
                if opened.width * opened.height > Image.MAX_IMAGE_PIXELS:
                    raise HTTPException(status_code=422, detail="image_too_large")
                image = ImageOps.exif_transpose(opened).convert("RGB")
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
            raise HTTPException(status_code=422, detail="invalid_image") from exc
        try:
            inference_started = time.perf_counter()
            predictions = self.engine.predict(image)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            inference_ms = (time.perf_counter() - inference_started) * 1000
            INFERENCE_LATENCY.observe(inference_ms / 1000)
        except Exception as exc:
            raise HTTPException(status_code=503, detail="inference_unavailable") from exc
        if len(predictions) == 0:
            status, reason, selected = "review_required", "no_die", None
        elif len(predictions) > 1:
            status, reason, selected = "review_required", "multiple_dice", None
        else:
            selected = predictions[0]
            status = "accepted" if selected.confidence >= self.threshold else "review_required"
            reason = None if status == "accepted" else "low_confidence"
        try:
            normalized_stream = io.BytesIO()
            image.save(normalized_stream, format="JPEG", quality=95)
            normalized_bytes = normalized_stream.getvalue()
            prediction_id = self.store.save_prediction(
                normalized_bytes, hashlib.sha256(normalized_bytes).hexdigest(), ".jpg", self.model_tag, status, reason,
                selected.sides if selected else None, selected.value if selected else None,
                selected.confidence if selected else None,
            )
        except Exception as exc:
            STORAGE_ERRORS.inc()
            raise HTTPException(status_code=503, detail="prediction_unavailable") from exc
        REQUESTS.labels(status=status, reason=reason or "none").inc()
        if selected is not None:
            CONFIDENCE.observe(selected.confidence)
        LATENCY.observe(time.perf_counter() - started)
        return {"prediction_id": prediction_id, "status": status, "reason": reason,
                "die_type": f"D{selected.sides}" if selected else None,
                "sides": selected.sides if selected else None, "value": selected.value if selected else None,
                "confidence": selected.confidence if selected else None, "model_version": self.model_tag,
                "inference_ms": inference_ms}

    @app.post("/v1/predictions/{prediction_id}/feedback")
    def feedback(self, prediction_id: str, feedback: FeedbackInput) -> dict[str, object]:
        if feedback.sides not in SIDES or feedback.value > feedback.sides:
            raise HTTPException(status_code=422, detail="invalid_die_result")
        try:
            result = self.store.add_feedback(prediction_id, feedback.confirmed, feedback.sides, feedback.value)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="prediction_not_found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="confirmation_mismatch") from exc
        except Exception as exc:
            STORAGE_ERRORS.inc()
            raise HTTPException(status_code=503, detail="feedback_unavailable") from exc
        FEEDBACK.labels(confirmed=str(feedback.confirmed).lower()).inc()
        return result

    @app.get("/v1/metrics", include_in_schema=False)
    def metrics(self) -> Response:
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
