"""Exercise the configured BentoML exporter without loading model weights."""

import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prometheus_client.parser import text_string_to_metric_families
from bentoml._internal.container import BentoMLContainer
from bentoml._internal.server.metrics.prometheus import PrometheusClient
from dice_viewer.recognition.service import DiceRecognitionService


async def call(app, path, method="GET"):
    messages = []

    async def receive():
        return {"type": "http.request", "body": b"{}", "more_body": False}

    async def send(message):
        messages.append(message)

    await app({"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
               "scheme": "http", "method": method, "path": path, "raw_path": path.encode(),
               "root_path": "", "query_string": b"", "headers": [(b"content-type", b"application/json")],
               "server": ("127.0.0.1", 24000), "client": ("127.0.0.1", 40000)}, receive, send)
    return messages[0], b"".join(m.get("body", b"") for m in messages[1:])


class NativeMetricsTests(unittest.IsolatedAsyncioTestCase):
    async def test_native_exporter_counts_rejected_request_once_and_excludes_scrapes(self):
        # ASGI-only test has no Bento supervisor to configure multiprocess files.
        original = BentoMLContainer.metrics_client.get()
        BentoMLContainer.metrics_client.set(PrometheusClient(multiproc=False))
        self.addCleanup(BentoMLContainer.metrics_client.set, original)
        with TemporaryDirectory() as directory, patch.dict(os.environ, {"DICE_OBSERVABILITY_DIR": directory}):
            app = DiceRecognitionService.to_asgi()
            start, before = await call(app, "/metrics")
            self.assertEqual(start["status"], 200)
            self.assertIn(b"text/plain", dict(start["headers"])[b"content-type"])
            before_samples = [s for f in text_string_to_metric_families(before.decode()) for s in f.samples]
            start, body = await call(app, "/v1/recognize", "POST")
            self.assertEqual(start["status"], 415)
            self.assertEqual(json.loads(body)["error"]["code"], "unsupported_media_type")
            _, after = await call(app, "/metrics")
            unknown, _ = await call(app, "/unknown/PRIVATE_MARKER")
            self.assertEqual(unknown["status"], 404)
            _, repeat = await call(app, "/metrics")
            self.assertNotIn(b"PRIVATE_MARKER", repeat)
            samples = [s for f in text_string_to_metric_families(after.decode()) for s in f.samples]

            def rejected_total(values, suffix):
                return sum(s.value for s in values if s.name.endswith(suffix)
                           and s.labels.get("endpoint") == "/v1/recognize"
                           and s.labels.get("http_response_code") == "415")

            self.assertEqual(rejected_total(samples, "_request_total") -
                             rejected_total(before_samples, "_request_total"), 1)
            self.assertEqual(rejected_total(samples, "_request_duration_seconds_count") -
                             rejected_total(before_samples, "_request_duration_seconds_count"), 1)
            repeated = [s for f in text_string_to_metric_families(repeat.decode()) for s in f.samples]
            self.assertEqual(rejected_total(samples, "_request_total"),
                             rejected_total(repeated, "_request_total"))
            self.assertFalse(any(s.labels.get("endpoint") == "/metrics" for s in samples))
            self.assertFalse(any("request_id" in s.labels or "image_id" in s.labels for s in samples))
            lines = (Path(directory) / "recognition.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)


if __name__ == "__main__":
    unittest.main()
