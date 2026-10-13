"""Native BentoML APIs for prediction, feedback, and service readiness."""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import time
from pathlib import Path
from threading import Lock
from typing import Annotated, Literal

# Capture Pillow's decoder before Ultralytics installs its HEIF fallback.
# Invalid uploads must not trigger optional package installation.
from PIL import Image, ImageOps, UnidentifiedImageError
from PIL.Image import open as open_uploaded_image

import bentoml
import torch
from bentoml.exceptions import NotFound, ServiceUnavailable, UnprocessableEntity
from bentoml.validators import ContentType
from prometheus_client import Counter, Gauge, Histogram
from pydantic import BaseModel, Field, field_validator
from starlette.exceptions import HTTPException as MultipartParseError
from starlette.requests import Request

from dice_viewer import SIDES
from dice_viewer.active_model import ACTIVE_MODEL_TAG
from dice_viewer.inference import InferenceEngine
from dice_viewer.storage import FeedbackStore

MAX_IMAGE_BYTES = 10 * 1024 * 1024
logger = logging.getLogger(__name__)
Image.MAX_IMAGE_PIXELS = 25_000_000
REQUESTS = Counter("dice_prediction_requests_total", "Prediction outcomes", ["status", "reason"])
FEEDBACK = Counter("dice_feedback_total", "Human feedback events", ["confirmed"])
LATENCY = Histogram("dice_prediction_seconds", "Full prediction request duration")
INFERENCE_LATENCY = Histogram("dice_inference_seconds", "Model inference duration")
CONFIDENCE = Histogram("dice_confidence", "Confidence for single-die predictions", buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0))
STORAGE_ERRORS = Counter("dice_storage_errors_total", "Prediction or feedback storage errors")
ACTIVE_MODEL = Gauge("dice_active_model", "Active model tag", ["version"])


class PredictionInput(bentoml.IODescriptor):
    file: Annotated[Path, ContentType("image/*")]
    details: bool = False

    @classmethod
    async def from_http_request(cls, request: Request, serde: object) -> PredictionInput:
        # Accept uploads only; the default BentoML parser also downloads file URLs.
        if request.headers.get("content-type", "").split(";", 1)[0].lower() != "multipart/form-data":
            raise UnprocessableEntity("multipart_upload_required")
        try:
            async with request.form(max_files=1, max_fields=1) as form:
                uploads = form.getlist("file")
                if len(uploads) != 1 or isinstance(uploads[0], str):
                    raise UnprocessableEntity("image_upload_required")
                return cls.model_validate({"file": uploads[0], "details": form.get("details", False)})
        except MultipartParseError as exc:
            raise UnprocessableEntity("invalid_multipart") from exc

    @field_validator("file", mode="before")
    @classmethod
    def limit_upload(cls, value: object) -> object:
        # Multipart uploads expose a size computed by the parser, not a client header.
        size = value.stat().st_size if isinstance(value, Path) else getattr(value, "size", None)
        if not isinstance(size, int) or not 0 < size <= MAX_IMAGE_BYTES:
            raise ValueError("invalid_image_size")
        return value


class FeedbackInput(BaseModel):
    confirmed: bool
    sides: Literal[6, 8, 10, 12, 20]
    value: int = Field(ge=1, le=20)


@bentoml.service(resources={"gpu": 1}, traffic={"timeout": 15})
class DiceService:
    bundle = bentoml.models.BentoModel(ACTIVE_MODEL_TAG)

    def __init__(self) -> None:
        self.model_tag = ACTIVE_MODEL_TAG
        if os.environ.get("DICE_MODEL_TAG", ACTIVE_MODEL_TAG) != ACTIVE_MODEL_TAG:
            raise RuntimeError("Configured model tag differs from the Bento bundle")
        folder = Path(self.bundle.path)
        self.threshold = float(json.loads((folder / "threshold.json").read_text(encoding="utf-8"))["threshold"])
        self.engine = InferenceEngine(folder / "detector.pt", folder, require_gpu=True)
        self.inference_lock = Lock()
        self.store = FeedbackStore()
        self.store.ready()
        ACTIVE_MODEL.labels(version=self.model_tag).set(1)

    @bentoml.api(route="/v1/health")
    def health(self) -> dict[str, str | bool]:
        """Check the saved model, GPU, and feedback storage."""
        try:
            storage_ready = self.store.ready()
        except Exception as exc:
            raise ServiceUnavailable("storage_unavailable") from exc
        if not torch.cuda.is_available():
            raise ServiceUnavailable("gpu_unavailable")
        return {"status": "ready", "model_version": self.model_tag, "storage_ready": storage_ready,
                "storage_backend": self.store.backend, "gpu": torch.cuda.get_device_name(0)}

    @bentoml.api(route="/v1/predictions", input_spec=PredictionInput)
    def predict(
        self, file: Annotated[Path, ContentType("image/*")], details: bool = False,
    ) -> dict[str, str | int | float | None]:
        """Read one JPEG/PNG die photo; details enables detection and review metadata."""
        started = time.perf_counter()
        with file.open("rb") as stream:
            body = stream.read(MAX_IMAGE_BYTES + 1)
        if not body or len(body) > MAX_IMAGE_BYTES:
            raise UnprocessableEntity("invalid_image_size")
        try:
            with open_uploaded_image(io.BytesIO(body)) as opened:
                if opened.format not in {"JPEG", "PNG"}:
                    raise UnprocessableEntity("unsupported_image_type")
                if opened.width * opened.height > Image.MAX_IMAGE_PIXELS:
                    raise UnprocessableEntity("image_too_large")
                image = ImageOps.exif_transpose(opened).convert("RGB")
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
            raise UnprocessableEntity("invalid_image") from exc
        try:
            inference_started = time.perf_counter()
            with self.inference_lock:
                predictions = (self.engine.predict(image) if details
                               else [self.engine.predict_single(image)])
                torch.cuda.synchronize()
            inference_ms = (time.perf_counter() - inference_started) * 1000
            INFERENCE_LATENCY.observe(inference_ms / 1000)
        except Exception as exc:
            logger.exception("Prediction inference failed")
            REQUESTS.labels(status="error", reason="inference_unavailable").inc()
            raise ServiceUnavailable("inference_unavailable") from exc
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
            logger.exception("Prediction storage failed")
            STORAGE_ERRORS.inc()
            REQUESTS.labels(status="error", reason="storage_unavailable").inc()
            raise ServiceUnavailable("prediction_unavailable") from exc
        REQUESTS.labels(status=status, reason=reason or "none").inc()
        if selected is not None:
            CONFIDENCE.observe(selected.confidence)
        LATENCY.observe(time.perf_counter() - started)
        result = {"prediction_id": prediction_id, "status": status, "reason": reason,
                "die_type": f"D{selected.sides}" if selected else None,
                "sides": selected.sides if selected else None, "value": selected.value if selected else None,
                "confidence": selected.confidence if selected else None, "model_version": self.model_tag,
                "inference_ms": inference_ms}
        # The detailed response exposes confidence, review status and the feedback ID.
        if details:
            return result
        return {"die_type": result["die_type"], "value": result["value"]}

    @bentoml.api(route="/v1/feedback")
    def feedback(self, prediction_id: str, feedback: FeedbackInput) -> dict[str, object]:
        """Confirm or correct a prediction ID returned with details enabled."""
        if feedback.sides not in SIDES or feedback.value > feedback.sides:
            raise UnprocessableEntity("invalid_die_result")
        try:
            result = self.store.add_feedback(prediction_id, feedback.confirmed, feedback.sides, feedback.value)
        except KeyError as exc:
            raise NotFound("prediction_not_found") from exc
        except ValueError as exc:
            raise UnprocessableEntity("confirmation_mismatch") from exc
        except Exception as exc:
            logger.exception("Feedback storage failed")
            STORAGE_ERRORS.inc()
            raise ServiceUnavailable("feedback_unavailable") from exc
        FEEDBACK.labels(confirmed=str(feedback.confirmed).lower()).inc()
        return result
