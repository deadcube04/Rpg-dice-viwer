"""Align BentoML's generated documentation with the binary API responses."""

import json
import uuid

from starlette.responses import JSONResponse

from .contract import error_response
from .observability import current_observation


def complete_spec(spec):
    operation = spec["paths"]["/v1/recognize"]["post"]
    binary = {"schema": {"type": "string", "format": "binary"}}
    operation["requestBody"] = {"required": True, "content": {"image/jpeg": binary, "image/png": binary}}
    responses = operation["responses"]
    responses["404"] = {**responses["200"], "description": "No candidates; original persisted; empty recognition response"}
    error_schema = {"type": "object", "required": ["request_id", "image_id", "error"],
                    "additionalProperties": False,
                    "properties": {"request_id": {"type": "string"},
                                   "image_id": {"type": "string", "nullable": True},
                                   "error": {"type": "object", "additionalProperties": False,
                                             "required": ["code", "message", "retryable"],
                                             "properties": {"code": {"type": "string"},
                                                            "message": {"type": "string"},
                                                            "retryable": {"type": "boolean"}}}}}
    for status, description in {"400": "invalid_request", "413": "image_too_large",
                                "415": "unsupported_media_type", "422": "invalid_image / too_many_dice / scene_too_complex",
                                "500": "internal_error", "503": "storage_unavailable / model_unavailable",
                                "504": "inference_timeout"}.items():
        responses[status] = {"description": description, "content": {"application/json": {"schema": error_schema}}}
    return spec


class ContractDocumentation:
    """Only changes the generated /docs.json representation, not inference."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["path"] == "/v1/recognize" and scope["method"] == "POST":
            headers = dict(scope["headers"])
            media_type = headers.get(b"content-type", b"").split(b";")[0].strip().lower()
            if media_type not in (b"image/jpeg", b"image/png"):
                # Reject before BentoML's special pickle deserializer/checker.
                body = error_response(uuid.uuid4().hex, None, "unsupported_media_type")
                observation = current_observation()
                if observation is not None:
                    observation.set_response(body)
                response = JSONResponse(body, status_code=415)
                return await response(scope, receive, send)
        if scope["type"] != "http" or scope["path"] != "/docs.json":
            return await self.app(scope, receive, send)
        messages = []

        async def capture(message):
            messages.append(message)

        await self.app(scope, receive, capture)
        start = messages[0]
        if start["status"] != 200:
            for message in messages:
                await send(message)
            return
        original = b"".join(m.get("body", b"") for m in messages[1:])
        body = json.dumps(complete_spec(json.loads(original)), ensure_ascii=False).encode()
        start["headers"] = [(k, v) for k, v in start["headers"] if k != b"content-length"]
        start["headers"].append((b"content-length", str(len(body)).encode()))
        await send(start)
        await send({"type": "http.response.body", "body": body})
