"""Behavioral tests for the stage-2 logging boundary; no model weights needed."""

import asyncio
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dice_viewer.recognition.contract import error_response
from dice_viewer.recognition.observability import (
    JsonlWriter, RecognitionLogging, current_observation,
)
from dice_viewer.recognition.openapi import ContractDocumentation


def recognition_body():
    return {
        "request_id": "a" * 32, "image_id": "b" * 32, "status": "needs_review",
        "dices_count": 1, "request_requires_review": True,
        "results": [{"id": "die-1", "bbox": [1, 2, 30, 40], "shape": "d6",
                     "die_type": "d6", "face_value": 3,
                     "confidence": {"detection": 0.99, "type": 0.9, "value": 0.7},
                     "auto_accepted": False, "review_reasons": ["low_value_confidence"]}],
        "thresholds": {"detection": 0.95, "type": 0.8, "value": 0.8},
        "versions": {"detector": "approved-detector", "classifier": "approved-classifier",
                     "labels": "c" * 64, "preprocessing": "approved-preprocessing"},
        "timings_ms": {"decode": 1, "persist": 2, "detect": 3, "classify": 4, "total": 12},
    }


async def invoke(app, body=b"image-bytes", mime=b"image/png"):
    scope = {"type": "http", "path": "/v1/recognize", "method": "POST",
             "headers": [(b"content-type", mime), (b"authorization", b"SECRET_TOKEN")],
             "query_string": b"", "state": {}}
    messages = []

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message):
        messages.append(message)

    await app(scope, receive, send)
    return messages


class LoggingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "recognition.jsonl"
        self.writer = JsonlWriter(self.path)

    def events(self):
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines()]

    async def test_success_keeps_response_and_projects_one_request_and_one_die(self):
        public = recognition_body()
        public["secret"] = "SECRET_TOKEN"
        public["results"][0]["image"] = "PRIVATE_IMAGE"
        wire = json.dumps(public).encode()

        async def app(scope, receive, send):
            await receive()
            obs = current_observation()
            obs.set_input(public["request_id"])
            obs.set_service(public["versions"], "cpu")
            obs.worker_finished(public, 200, public["timings_ms"], 12)
            obs.set_response(public)
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": wire})

        messages = await invoke(RecognitionLogging(app, writer=self.writer))
        self.assertEqual(messages[-1]["body"], wire)
        events = self.events()
        self.assertEqual([e["event"] for e in events], ["request", "die"])
        request, die = events
        self.assertEqual(request["http_status"], 200)
        self.assertEqual(request["request_id"], "a" * 32)
        self.assertEqual(request["input"]["content_type"], "image/png")
        self.assertEqual(request["input"]["received_bytes"], 11)
        self.assertEqual(request["input"]["image_features"]["status"], "not_available")
        self.assertEqual(request["output"]["status"], "needs_review")
        self.assertEqual(request["worker_timings_ms"]["persist"], 2)
        self.assertGreaterEqual(request["http_duration_ms"], 0)
        self.assertTrue(request["response_completed"])
        self.assertFalse(request["worker_pending"])
        self.assertEqual(die["request_id"], request["request_id"])
        self.assertEqual(die["result"]["face_value"], 3)
        self.assertEqual(die["result"]["confidence"]["value"], 0.7)
        self.assertTrue(request["timestamp_utc"].endswith("+00:00"))
        self.assertEqual(request["schema_version"], 1)
        text = self.path.read_text(encoding="utf-8")
        for secret in ("SECRET_TOKEN", "PRIVATE_IMAGE", "image-bytes", "authorization"):
            self.assertNotIn(secret, text)
        self.assertIsNone(current_observation())

    async def test_early_mime_rejection_is_logged_with_same_request_id(self):
        async def unreachable(*args):
            self.fail("MIME rejection must not reach the inference application")

        messages = await invoke(RecognitionLogging(ContractDocumentation(unreachable), writer=self.writer),
                                mime=b"application/json; secret=SECRET_TOKEN")
        body = json.loads(messages[-1]["body"])
        self.assertEqual(messages[0]["status"], 415)
        event, = self.events()
        self.assertEqual(event["request_id"], body["request_id"])
        self.assertEqual(event["error_code"], "unsupported_media_type")
        self.assertEqual(event["input"]["content_type"], "other")
        self.assertEqual(event["input"]["received_bytes"], 0)
        self.assertNotIn("SECRET_TOKEN", self.path.read_text(encoding="utf-8"))

    async def test_error_codes_keep_their_http_semantics(self):
        for code, status in (("invalid_request", 400), ("image_too_large", 413),
                             ("invalid_image", 422), ("storage_unavailable", 503),
                             ("model_unavailable", 503), ("internal_error", 500),
                             ("inference_timeout", 504)):
            async def app(scope, receive, send):
                body = error_response("d" * 32, None, code)
                current_observation().set_response(body)
                await send({"type": "http.response.start", "status": status, "headers": []})
                await send({"type": "http.response.body", "body": json.dumps(body).encode()})
            messages = await invoke(RecognitionLogging(app, writer=self.writer))
            self.assertEqual(messages[0]["status"], status)
            self.assertEqual(self.events()[-1]["error_code"], code)
        self.assertEqual(len(self.events()), 7)

    async def test_other_routes_are_not_logged(self):
        async def app(scope, receive, send):
            self.assertIsNone(current_observation())
            await send({"type": "http.response.start", "status": 200, "headers": []})

        async def send(message):
            pass

        await RecognitionLogging(app, writer=self.writer)(
            {"type": "http", "path": "/readyz", "method": "GET"}, None, send)
        self.assertFalse(self.path.exists())

    async def test_late_worker_has_separate_event_without_second_request(self):
        holder = []

        async def app(scope, receive, send):
            obs = current_observation()
            holder.append(obs)
            obs.worker_started()
            obs.set_response(error_response("e" * 32, None, "inference_timeout"))
            await send({"type": "http.response.start", "status": 504, "headers": []})
            await send({"type": "http.response.body", "body": b"{}"})

        await invoke(RecognitionLogging(app, writer=self.writer))
        self.assertTrue(self.events()[0]["worker_pending"])
        holder[0].worker_finished(error_response("e" * 32, None, "inference_timeout"),
                                  504, {"decode": 1}, 31_000)
        events = self.events()
        self.assertEqual([e["event"] for e in events], ["request", "worker_completion"])
        self.assertEqual(events[1]["request_id"], "e" * 32)
        self.assertEqual(events[1]["worker_duration_ms"], 31_000)

    async def test_unhandled_exception_is_not_swallowed_or_leaked(self):
        async def app(scope, receive, send):
            raise RuntimeError("SECRET_TOKEN")

        with self.assertRaisesRegex(RuntimeError, "SECRET_TOKEN"):
            await invoke(RecognitionLogging(app, writer=self.writer))
        event, = self.events()
        self.assertFalse(event["response_completed"])
        self.assertIsNone(event["http_status"])
        self.assertEqual(event["error_code"], "unhandled_exception")
        self.assertNotIn("SECRET_TOKEN", self.path.read_text(encoding="utf-8"))

    async def test_overlapping_requests_keep_independent_contexts(self):
        barrier = asyncio.Event()
        entered = 0

        async def app(scope, receive, send):
            nonlocal entered
            message = await receive()
            expected_id = message["body"].decode()
            current_observation().set_input(expected_id)
            entered += 1
            if entered == 2:
                barrier.set()
            await barrier.wait()
            self.assertEqual(current_observation().request_id, expected_id)
            await send({"type": "http.response.start", "status": 422, "headers": []})
            await send({"type": "http.response.body", "body": b"{}"})

        middleware = RecognitionLogging(app, writer=self.writer)
        await asyncio.gather(invoke(middleware, body=b"a" * 32), invoke(middleware, body=b"b" * 32))
        self.assertEqual({e["request_id"] for e in self.events()}, {"a" * 32, "b" * 32})
        self.assertIsNone(current_observation())

    async def test_late_worker_write_failure_is_contained(self):
        holder = []

        async def app(scope, receive, send):
            holder.append(current_observation())
            holder[0].worker_started()
            await send({"type": "http.response.start", "status": 504, "headers": []})
            await send({"type": "http.response.body", "body": b"{}"})

        await invoke(RecognitionLogging(app, writer=self.writer))
        self.path.unlink()
        self.path.mkdir()
        with self.assertLogs("dice_viewer.recognition.observability", level="ERROR") as logs:
            holder[0].worker_finished(error_response("a" * 32, None, "inference_timeout"), 504, {}, 31_000)
        self.assertIn("observability_write_failed", logs.output[0])
        self.assertNotIn(str(self.path), logs.output[0])

    async def test_unwritable_log_does_not_change_response(self):
        async def app(scope, receive, send):
            await send({"type": "http.response.start", "status": 503, "headers": []})
            await send({"type": "http.response.body", "body": b"unchanged"})

        self.path.mkdir()
        with self.assertLogs("dice_viewer.recognition.observability", level="ERROR") as logs:
            messages = await invoke(RecognitionLogging(app, writer=self.writer))
        self.assertEqual(messages[-1]["body"], b"unchanged")
        self.assertEqual(messages[0]["status"], 503)
        self.assertTrue(all("observability_write_failed" in entry for entry in logs.output))
        self.assertNotIn(str(self.path), "\n".join(logs.output))

    def test_append_and_threaded_writes_keep_all_complete_lines(self):
        self.writer.write([{"event": "sentinel"}])
        before = self.path.read_bytes()
        with ThreadPoolExecutor(max_workers=8) as executor:
            list(executor.map(lambda i: self.writer.write([{"event": "sample", "n": i}]), range(80)))
        self.assertTrue(self.path.read_bytes().startswith(before))
        events = self.events()
        self.assertEqual(len(events), 81)
        self.assertEqual({event["n"] for event in events[1:]}, set(range(80)))

    def test_default_directory_uses_existing_state_volume(self):
        with patch.dict(os.environ, {"DICE_STATE_DIR": self.directory.name}, clear=True):
            self.assertEqual(JsonlWriter.from_environment().path,
                             Path(self.directory.name) / "observability" / "recognition.jsonl")


if __name__ == "__main__":
    unittest.main()
