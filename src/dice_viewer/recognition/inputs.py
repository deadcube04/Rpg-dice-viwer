"""Bounded binary root input for the pinned BentoML SDK."""

import asyncio
import io
import struct
import time
import uuid
from typing import ClassVar

import bentoml
from PIL import Image, ImageOps, UnidentifiedImageError
from PIL.JpegImagePlugin import JpegImageFile
from PIL.PngImagePlugin import PngImageFile

MAX_BYTES = 10_485_760


class InputError(Exception):
    def __init__(self, status: int, code: str):
        self.status, self.code = status, code


class BinaryImage(bentoml.IODescriptor):
    """Read the stream before Pillow; no multipart or JSON interpretation."""

    __root_input__: ClassVar[bool] = True
    media_type: ClassVar[str] = "image/jpeg"
    body: bytes = b""
    content_type: str = ""
    request_id: str
    error_status: int = 0
    error_code: str = ""

    @classmethod
    def model_json_schema(cls, *args, **kwargs):
        return {"type": "string", "format": "binary", "root_input": True}

    @classmethod
    async def from_http_request(cls, request, serde):
        value = cls(request_id=uuid.uuid4().hex,
                    content_type=request.headers.get("content-type", "").split(";")[0].strip().lower())
        try:
            if request.query_params or request.headers.get("content-encoding", "identity").lower() != "identity":
                raise InputError(400, "invalid_request")
            if value.content_type not in ("image/jpeg", "image/png"):
                raise InputError(415, "unsupported_media_type")
            length = request.headers.get("content-length")
            if length is not None:
                if not length.isdecimal():
                    raise InputError(400, "invalid_request")
                if int(length) > MAX_BYTES:
                    raise InputError(413, "image_too_large")
            body = bytearray()
            async with asyncio.timeout(45):
                async for chunk in request.stream():
                    if len(body) + len(chunk) > MAX_BYTES:
                        raise InputError(413, "image_too_large")
                    body.extend(chunk)
            value.body = bytes(body)
            if not body:
                raise InputError(422, "invalid_image")
        except TimeoutError:
            value.error_status, value.error_code = 400, "invalid_request"
        except InputError as exc:
            value.error_status, value.error_code = exc.status, exc.code
        return value


def decode_image(body: bytes, content_type: str):
    """Enforce cumulative header limits before full decode and orient once."""
    started = time.perf_counter()
    if body.startswith(b"\x89PNG\r\n\x1a\n"):
        decoder, actual_type = PngImageFile, "image/png"
    elif body.startswith(b"\xff\xd8\xff"):
        decoder, actual_type = JpegImageFile, "image/jpeg"
    else:
        raise InputError(422, "invalid_image")
    if actual_type != content_type:
        raise InputError(415, "unsupported_media_type")
    try:
        # Direct format plugins avoid Ultralytics' Image.open monkey patch,
        # whose generic fallback attempts to install pi-heif on corrupt input.
        with decoder(io.BytesIO(body)) as source:
            if max(source.size) > 4096 or source.width * source.height > 16_000_000:
                raise InputError(413, "image_too_large")
            if getattr(source, "n_frames", 1) != 1:
                raise InputError(422, "invalid_image")
            source.verify()
        with decoder(io.BytesIO(body)) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            image.load()
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise InputError(413, "image_too_large") from exc
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError,
            IndexError, EOFError, struct.error) as exc:
        raise InputError(422, "invalid_image") from exc
    return image, (time.perf_counter() - started) * 1000
