"""Filter native exposition; do not register duplicate HTTP instruments."""

class InferenceMetrics:
    """Expose only the fixed inference path, never client-controlled URL paths.

    BentoML still owns instrumentation and internal series allocation.
    This middleware filters the exported representation only.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["path"] != "/metrics" or scope["method"] != "GET":
            return await self.app(scope, receive, send)
        messages = []

        async def capture(message):
            messages.append(message)

        await self.app(scope, receive, capture)
        if not messages or messages[0].get("status") != 200:
            for message in messages:
                await send(message)
            return
        # Bento's supervisor must configure multiprocess storage before this import.
        from prometheus_client import CollectorRegistry, generate_latest
        from prometheus_client.parser import text_string_to_metric_families

        raw = b"".join(m.get("body", b"") for m in messages[1:]).decode("utf-8")
        families = list(text_string_to_metric_families(raw))
        for family in families:
            family.samples = [sample for sample in family.samples
                              if "endpoint" not in sample.labels
                              or sample.labels["endpoint"] == "/v1/recognize"]

        class Collector:
            def collect(self):
                yield from (family for family in families if family.samples)

        registry = CollectorRegistry()
        registry.register(Collector())
        body = generate_latest(registry)
        start = dict(messages[0])
        start["headers"] = [(k, v) for k, v in start["headers"] if k != b"content-length"]
        start["headers"].append((b"content-length", str(len(body)).encode()))
        await send(start)
        await send({"type": "http.response.body", "body": body})
