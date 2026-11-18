"""Bento service extension placing JSONL observation outside framework errors."""

from _bentoml_sdk import Service

from .observability import RecognitionLogging


class ObservedService(Service):
    def to_asgi(self,is_main=True):
        # The complete app includes Bento admission/timeout and error handling.
        # No changes to routes, concurrency, deadlines or HTTP response bodies.
        return RecognitionLogging(super().to_asgi(is_main=is_main))
