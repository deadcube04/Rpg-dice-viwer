"""Exercise real descriptor/service/error flow with external inference/S3 doubles."""

from concurrent.futures import ThreadPoolExecutor
import asyncio
import io
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from threading import Event, Lock
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PIL import Image
from starlette.requests import Request

from dice_viewer.recognition.inputs import BinaryImage
from dice_viewer.recognition.image_features import extract_image_features
from dice_viewer.recognition.observability import JsonlWriter, RecognitionLogging
from dice_viewer.recognition.openapi import ContractDocumentation
from dice_viewer.recognition.service import DiceRecognitionService


class ServiceLoggingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "recognition.jsonl"
        self.service = DiceRecognitionService.inner.__new__(DiceRecognitionService.inner)
        self.service.model_version = "approved"
        self.service.versions = {"detector": "approved-detector", "classifier": "approved-classifier",
                                 "labels": "c" * 64, "preprocessing": "approved-preprocessing"}
        self.service.device = SimpleNamespace(resolved="cpu")
        self.service.busy = Lock()
        self.service.executor = ThreadPoolExecutor(max_workers=1)
        self.addCleanup(self.service.executor.shutdown, wait=True)
        self.service.store = SimpleNamespace(save=lambda *args: "b" * 32)
        self.service.engine = SimpleNamespace(infer=lambda source: {
            "technical_candidate_limit_reached": False, "status": "no_candidates", "items": [],
            "manual_intervention_required": True,
            "timings_seconds": {"detector": 0.002, "crop_classifier_policy": 0.001}})
        stream = io.BytesIO()
        Image.new("RGB", (32, 32)).save(stream, format="PNG")
        self.body = stream.getvalue()

    async def recognize(self, body=None, query=b""):
        async def endpoint(scope, receive, send):
            upload = await BinaryImage.from_http_request(Request(scope, receive), None)
            response = await self.service.recognize(upload)
            await response(scope, receive, send)

        app = RecognitionLogging(ContractDocumentation(endpoint), writer=JsonlWriter(self.path))
        scope = {"type": "http", "path": "/v1/recognize", "method": "POST",
                 "headers": [(b"content-type", b"image/png")], "query_string": query}
        messages = []

        async def receive():
            return {"type": "http.request", "body": self.body if body is None else body,
                    "more_body": False}

        async def send(message):
            messages.append(message)

        await app(scope, receive, send)
        response_body = json.loads(messages[-1]["body"])
        events = [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines()]
        return messages[0]["status"], response_body, events

    async def test_no_candidates_404_has_original_id_and_completed_worker(self):
        status, body, events = await self.recognize()
        self.assertEqual(status, 404)
        self.assertEqual(body["status"], "no_candidates")
        self.assertEqual(body["results"], [])
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["image_id"], "b" * 32)
        self.assertEqual(events[0]["request_id"], body["request_id"])
        self.assertEqual(events[0]["worker_timings_ms"]["detect"], 2)
        self.assertFalse(events[0]["worker_pending"])
        self.assertFalse(self.service.busy.locked())
        features = events[0]["input"]["image_features"]
        self.assertEqual(features["status"], "ok")
        self.assertEqual((features["width"], features["height"]), (32, 32))
        self.assertEqual(features["brightness_mean"], 0)
        self.assertGreaterEqual(features["extraction_ms"], 0)
        self.assertNotIn("image_features", body)
        self.assertEqual(set(body["timings_ms"]), {"decode", "persist", "detect", "classify", "total"})

    async def test_invalid_descriptor_records_generated_public_id(self):
        status, body, events = await self.recognize(query=b"token=PRIVATE_TOKEN")
        self.assertEqual(status, 400)
        self.assertEqual(body["error"]["code"], "invalid_request")
        self.assertEqual(events[0]["request_id"], body["request_id"])
        self.assertEqual(events[0]["error_code"], "invalid_request")
        self.assertNotIn("PRIVATE_TOKEN", self.path.read_text(encoding="utf-8"))
        self.assertEqual(events[0]["input"]["image_features"]["status"], "not_available")

    async def test_busy_service_503_does_not_claim_worker_started(self):
        self.service.busy.acquire()
        self.addCleanup(self.service.busy.release)
        status, body, events = await self.recognize()
        self.assertEqual(status, 503)
        self.assertEqual(body["error"]["code"], "model_unavailable")
        self.assertFalse(events[0]["worker_pending"])
        self.assertIsNone(events[0]["worker_duration_ms"])

    async def test_storage_error_is_sanitized_and_lock_released(self):
        def fail(*args):
            raise RuntimeError("S3_SECRET_KEY=PRIVATE_TOKEN")
        self.service.store = SimpleNamespace(save=fail)
        with self.assertLogs("dice_viewer.recognition.service", level="WARNING") as diagnostic:
            status, body, events = await self.recognize()
        self.assertEqual(status, 503)
        self.assertEqual(events[0]["error_code"], "storage_unavailable")
        self.assertIsNone(body["image_id"])
        self.assertFalse(self.service.busy.locked())
        self.assertNotIn("PRIVATE_TOKEN", str(diagnostic.output) + self.path.read_text(encoding="utf-8"))

    async def test_internal_error_does_not_log_dependency_exception(self):
        def fail(source):
            raise RuntimeError("Authorization: PRIVATE_TOKEN")
        self.service.engine = SimpleNamespace(infer=fail)
        with self.assertLogs("dice_viewer.recognition.service", level="ERROR") as diagnostic:
            status, body, events = await self.recognize()
        self.assertEqual(status, 500)
        self.assertEqual(body["error"]["code"], "internal_error")
        self.assertEqual(events[0]["error_code"], "internal_error")
        self.assertFalse(self.service.busy.locked())
        self.assertNotIn("PRIVATE_TOKEN", str(diagnostic.output) + self.path.read_text(encoding="utf-8"))

    async def test_http_timeout_then_late_persistence_keeps_separate_image_id(self):
        gate = Event()
        entered = Event()

        def slow_save(*args):
            entered.set()
            if not gate.wait(5):
                raise RuntimeError("test gate not released")
            return "b" * 32

        self.service.store = SimpleNamespace(save=slow_save)
        original_wait_for = asyncio.wait_for

        async def short_wait(future, timeout):
            # Keep the real shielded future and worker; shorten only HTTP wait.
            return await original_wait_for(future, timeout=0.05)

        try:
            with patch("dice_viewer.recognition.service.asyncio.wait_for", new=short_wait):
                status, body, events = await self.recognize()
            self.assertTrue(entered.is_set())
            self.assertEqual(status, 504)
            self.assertIsNone(body["image_id"])
            self.assertTrue(events[0]["worker_pending"])
            self.assertTrue(self.service.busy.locked())
        finally:
            gate.set()
            self.service.executor.submit(lambda: None).result(timeout=5)
        events = [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([e["event"] for e in events], ["request", "worker_completion"])
        self.assertIsNone(events[0]["image_id"])
        self.assertEqual(events[1]["image_id"], "b" * 32)
        self.assertEqual(events[1]["request_id"], body["request_id"])
        self.assertEqual(events[1]["input"]["image_features"]["status"], "ok")
        self.assertFalse(self.service.busy.locked())

    async def test_feature_failure_keeps_inference_response_and_sanitizes_diagnostic(self):
        with patch("dice_viewer.recognition.observability.extract_image_features",
                   side_effect=MemoryError("PRIVATE_TOKEN")):
            with self.assertLogs("dice_viewer.recognition.observability", level="ERROR") as diagnostic:
                status, body, events = await self.recognize()
        self.assertEqual(status, 404)
        self.assertEqual(body["status"], "no_candidates")
        features = events[0]["input"]["image_features"]
        self.assertEqual(features["status"], "error")
        self.assertNotIn("brightness_mean", features)
        self.assertGreaterEqual(features["extraction_ms"], 0)
        self.assertNotIn("PRIVATE_TOKEN", str(diagnostic.output) + self.path.read_text(encoding="utf-8"))

    async def test_features_use_exif_oriented_image_without_changing_inference_pixels(self):
        image = Image.new("RGB", (12, 8), (255, 0, 0))
        exif = image.getexif()
        exif[274] = 6  # Decoder rotates 90 degrees: the model sees 8 by 12.
        stream = io.BytesIO()
        image.save(stream, format="PNG", exif=exif)
        seen = []
        original_infer = self.service.engine.infer

        def inspect(source):
            with Image.open(source) as decoded:
                seen.append((decoded.size, decoded.getpixel((0, 0))))
            return original_infer(source)

        self.service.engine = SimpleNamespace(infer=inspect)
        status, body, events = await self.recognize(body=stream.getvalue())
        self.assertEqual(status, 404)
        self.assertEqual(seen, [((8, 12), (255, 0, 0))])
        features = events[0]["input"]["image_features"]
        self.assertEqual((features["width"], features["height"]), (8, 12))
        self.assertEqual(features["rgb_mean"], {"red": 1, "green": 0, "blue": 0})

    async def test_submission_failure_releases_lock_and_does_not_claim_running_worker(self):
        self.service.executor.shutdown(wait=True)
        with self.assertRaises(RuntimeError):
            await self.recognize()
        event = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(event["error_code"], "unhandled_exception")
        self.assertFalse(event["worker_pending"])
        self.assertFalse(self.service.busy.locked())

    async def test_cancellation_keeps_worker_alive_and_logs_completion_separately(self):
        gate, entered = Event(), Event()

        def slow_save(*args):
            entered.set()
            if not gate.wait(5):
                raise RuntimeError("test gate not released")
            return "b" * 32

        self.service.store = SimpleNamespace(save=slow_save)
        task = asyncio.create_task(self.recognize())
        try:
            self.assertTrue(await asyncio.to_thread(entered.wait, 5))
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertTrue(self.service.busy.locked())
        finally:
            gate.set()
            self.service.executor.submit(lambda: None).result(timeout=5)
        events = [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([e["event"] for e in events], ["request", "worker_completion"])
        self.assertEqual(events[0]["error_code"], "request_cancelled")
        self.assertFalse(events[0]["response_completed"])
        self.assertFalse(self.service.busy.locked())

    async def test_cancellation_during_extraction_keeps_pending_snapshot_and_late_features(self):
        gate, entered = Event(), Event()

        def slow_features(image):
            entered.set()
            if not gate.wait(5):
                raise RuntimeError("test gate not released")
            return extract_image_features(image)

        with patch("dice_viewer.recognition.observability.extract_image_features", side_effect=slow_features):
            task = asyncio.create_task(self.recognize())
            try:
                self.assertTrue(await asyncio.to_thread(entered.wait, 5))
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
            finally:
                gate.set()
                self.service.executor.submit(lambda: None).result(timeout=5)
        events = [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([e["event"] for e in events], ["request", "worker_completion"])
        self.assertEqual(events[0]["input"]["image_features"]["status"], "pending")
        self.assertEqual(events[1]["input"]["image_features"]["status"], "ok")
        self.assertEqual(events[1]["input"]["image_features"]["width"], 32)
        self.assertFalse(self.service.busy.locked())


if __name__ == "__main__":
    unittest.main()
