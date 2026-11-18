"""Local, allowlisted JSONL events. No image bytes or exception messages."""

import asyncio
from contextvars import ContextVar
from datetime import datetime, timezone
import json
import logging
import math
import os
from pathlib import Path
import re
from threading import Lock
import time
import uuid

from .contract import MESSAGES
from .image_features import FEATURE_VERSION, extract_image_features
from .log_storage import AppendOnlyLog, LogPolicy

logger = logging.getLogger(__name__)
_current = ContextVar("recognition_observation", default=None)
_UUID = re.compile(r"[0-9a-f]{32}\Z")
_REASONS = {"low_detection_confidence", "low_type_confidence", "low_value_confidence",
            "inconsistent_type_value"}
_TYPES = {"d4", "d6", "d8", "d10", "d10_tens", "d12", "d20"}
_TIMINGS = ("decode", "persist", "detect", "classify", "total")


def current_observation():
    return _current.get()


def _diagnostic(code="observability_write_failed"):
    # Do not include paths, exception text, traceback or a failed payload.
    try:
        logger.error(code)
    except Exception:
        pass


def _number(value):
    return value if type(value) in (int, float) and math.isfinite(value) else None


def _timings(values):
    return {key: _number(values[key]) for key in _TIMINGS if key in values}


def _output(body):
    """Copy only public inference scalars; never serialize the supplied dict."""
    result = {}
    if body.get("status") in ("ok", "needs_review", "no_candidates"):
        result["status"] = body["status"]
    for key in ("dices_count", "request_requires_review"):
        if type(body.get(key)) in (bool, int):
            result[key] = body[key]
    if "timings_ms" in body:
        result["timings_ms"] = _timings(body["timings_ms"])
    return result


def _dice(body):
    results = []
    for index, item in enumerate(body.get("results", [])[:8]):
        confidence = item.get("confidence", {})
        bbox = item.get("bbox", ())
        results.append({
            "id": f"die-{index + 1}",
            "bbox": [_number(value) for value in bbox] if len(bbox) == 4 else None,
            "shape": item.get("shape") if item.get("shape") in _TYPES - {"d10_tens"} else None,
            "die_type": item.get("die_type") if item.get("die_type") in _TYPES else None,
            "face_value": _number(item.get("face_value")),
            "confidence": {key: _number(confidence.get(key)) for key in ("detection", "type", "value")},
            "auto_accepted": item.get("auto_accepted") is True,
            "review_reasons": [reason for reason in item.get("review_reasons", []) if reason in _REASONS],
        })
    return results


class JsonlWriter:
    """Serialize safe events and delegate append/rotation/expiration."""

    def __init__(self, path, policy=None):
        self.path = Path(path)
        self.storage = AppendOnlyLog(self.path, policy)

    @classmethod
    def from_environment(cls):
        root = Path(os.environ.get("DICE_PROJECT_ROOT", Path(__file__).resolve().parents[3]))
        state = Path(os.environ.get("DICE_STATE_DIR", root / "runtime"))
        default_directory = (state / "observability" if "DICE_STATE_DIR" in os.environ
                             else root / "logs/recognition/native")
        directory = Path(os.environ.get("DICE_OBSERVABILITY_DIR", default_directory))
        try:
            policy = LogPolicy(
                max_bytes=int(os.environ.get("DICE_LOG_MAX_BYTES", 10 * 1024 * 1024)),
                retention_days=int(os.environ.get("DICE_LOG_RETENTION_DAYS", 30)),
                max_total_bytes=int(os.environ.get("DICE_LOG_MAX_TOTAL_BYTES", 1024 ** 3)))
        except (ValueError, OverflowError):
            _diagnostic("observability_config_invalid")
            policy = LogPolicy()
        return cls(directory / "recognition.jsonl", policy)

    def write(self, events):
        try:
            payload = "".join(json.dumps(event, ensure_ascii=False, allow_nan=False,
                                         separators=(",", ":")) + "\n" for event in events).encode("utf-8")
            self.storage.append(payload)
        except Exception:
            _diagnostic()


class RequestObservation:
    def __init__(self, writer, content_type):
        self.writer = writer
        self.started = time.perf_counter()
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.request_id = uuid.uuid4().hex
        self.image_id = None
        self.content_type = content_type if content_type in ("image/png", "image/jpeg") else "other"
        self.received_bytes = 0
        self.image_features = {"version": FEATURE_VERSION, "status": "not_available",
                               "reason": "not_decoded_or_not_processed"}
        self.versions = None
        self.device = None
        self.output = {}
        self.dice = []
        self.error_code = None
        self.response_bound = False
        self.worker_pending = False
        self.worker_duration_ms = None
        self.worker_timings_ms = {}
        self.finished = False
        self.lock = Lock()

    def set_input(self, request_id):
        if isinstance(request_id, str) and _UUID.fullmatch(request_id):
            self.request_id = request_id

    def set_service(self, versions, device):
        try:
            self.versions = {key: versions[key] for key in ("detector", "classifier", "labels", "preprocessing")}
            self.device = "cpu" if device == "cpu" else "cuda" if device.startswith("cuda") else "other"
        except Exception:
            _diagnostic()

    def set_response(self, body):
        try:
            self.response_bound = True
            self.set_input(body.get("request_id"))
            image_id = body.get("image_id")
            self.image_id = image_id if isinstance(image_id, str) and _UUID.fullmatch(image_id) else None
            code = body.get("error", {}).get("code")
            self.error_code = code if code in MESSAGES else None
            self.output = _output(body)
            self.dice = _dice(body)
        except Exception:
            _diagnostic()

    def worker_started(self):
        with self.lock:
            self.worker_pending = True

    def observe_image(self, image):
        """Optional telemetry: feature failures must not replace inference results."""
        started = time.perf_counter()
        try:
            with self.lock:
                self.image_features = {"version": FEATURE_VERSION, "status": "pending"}
            values = extract_image_features(image)
            features = {**values, "version": FEATURE_VERSION, "status": "ok",
                        "extraction_ms": (time.perf_counter() - started) * 1000}
        except Exception:
            features = {"version": FEATURE_VERSION, "status": "error",
                        "extraction_ms": (time.perf_counter() - started) * 1000}
            _diagnostic("observability_feature_extraction_failed")
        with self.lock:
            self.image_features = features

    def worker_submission_failed(self):
        with self.lock:
            self.worker_pending = False

    def worker_finished(self, body, status, timings, duration):
        try:
            with self.lock:
                self.worker_pending = False
                self.worker_duration_ms = duration
                self.worker_timings_ms = _timings(timings)
                if self.finished:
                    code = body.get("error", {}).get("code")
                    image_id = body.get("image_id")
                    completed_image_id = image_id if isinstance(image_id, str) and _UUID.fullmatch(image_id) else None
                    self.writer.write([{**self._base("worker_completion"),
                                        "image_id": completed_image_id,
                                        "worker_http_status": status,
                                        "error_code": code if code in MESSAGES else None,
                                        "worker_duration_ms": duration,
                                        "worker_timings_ms": self.worker_timings_ms,
                                        "input": {"image_features": self.image_features}}])
        except Exception:
            _diagnostic()

    def _base(self, event):
        return {"schema_version": 1, "event": event,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "request_id": self.request_id, "image_id": self.image_id,
                "versions": self.versions, "device": self.device}

    def finish(self, status, completed, ended, failure=None):
        try:
            with self.lock:
                if self.finished:
                    return
                self.finished = True
                framework_error = None
                if not self.response_bound:
                    framework_error = ({429: "framework_rejected", 504: "framework_timeout"}.get(status)
                                       or ("framework_error" if status is not None and status >= 400 else None))
                request = {**self._base("request"), "started_at_utc": self.started_at,
                           "http_observation_version": 2,
                           "http_status": status, "response_completed": completed,
                           "http_duration_ms": (ended - self.started) * 1000,
                           "input": {"content_type": self.content_type,
                                     "received_bytes": self.received_bytes,
                                     "image_features": self.image_features},
                           "output": self.output, "error_code": failure or self.error_code or framework_error,
                           "worker_pending": self.worker_pending,
                           "worker_duration_ms": self.worker_duration_ms,
                           "worker_timings_ms": self.worker_timings_ms}
                events = [request]
                if completed:
                    events.extend({**self._base("die"), "result": die} for die in self.dice)
                self.writer.write(events)
        except Exception:
            _diagnostic()


class RecognitionLogging:
    """Observe complete inference HTTP, including framework-generated errors.

    No request/response payload is copied or buffered here.
    """

    def __init__(self, app, writer=None):
        self.app = app
        self.writer = writer if writer is not None else JsonlWriter.from_environment()

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["path"] != "/v1/recognize" or scope["method"] != "POST":
            return await self.app(scope, receive, send)
        # Compare bytes without decoding arbitrary client-controlled values.
        mime = next((value.split(b";")[0].strip().lower() for key, value in scope["headers"]
                     if key == b"content-type"), b"")
        content_type = {b"image/png": "image/png", b"image/jpeg": "image/jpeg"}.get(mime, "other")
        obs = RequestObservation(self.writer, content_type)
        token = _current.set(obs)
        status, completed, ended, failure = None, False, None, None

        async def observe_receive():
            message = await receive()
            if message["type"] == "http.request":
                obs.received_bytes += len(message.get("body", b""))
            return message

        async def observe_send(message):
            nonlocal status, completed, ended
            await send(message)
            if message["type"] == "http.response.start":
                status = message["status"]
            elif message["type"] == "http.response.body" and not message.get("more_body", False):
                completed, ended = True, time.perf_counter()

        try:
            await self.app(scope, observe_receive, observe_send)
        except asyncio.CancelledError:
            failure = "request_cancelled"
            raise
        except Exception:
            failure = "unhandled_exception"
            raise
        finally:
            obs.finish(status, completed, ended if ended is not None else time.perf_counter(), failure)
            _current.reset(token)
