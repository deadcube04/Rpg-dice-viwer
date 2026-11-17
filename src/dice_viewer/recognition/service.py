"""BentoML V1 binary API with original-first persistence and bounded processing."""

import asyncio
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

import bentoml
import torch
from starlette.responses import JSONResponse

from .contract import RecognitionResponse, error_response
from .engine import DecodedImage, load_approved
from .inputs import BinaryImage, InputError, decode_image
from .store import OriginalStore
from .openapi import ContractDocumentation

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@bentoml.service(name="DiceRecognitionService", workers=1, threads=1,
                 traffic={"timeout": 90, "concurrency": 1}, metrics={"enabled": False})
class DiceRecognitionService:
    def __init__(self):
        root = Path(os.environ.get("DICE_PROJECT_ROOT", Path(__file__).resolve().parents[3])).resolve()
        self.engine, protocol, self.device = load_approved(root, os.environ.get("DICE_DEVICE", "cpu"))
        cfg = protocol["configuration"]
        self.model_version = cfg["candidate_id"]
        labels_hash = protocol["candidate"]["label_maps_sha256"]
        self.versions = {"detector": f"{self.model_version}-detector",
                         "classifier": f"{self.model_version}-classifier",
                         "labels": labels_hash,
                         "preprocessing": cfg["classifier"]["preprocessing"]}
        self.store = OriginalStore()
        if (root / "RELEASE.json").exists():
            from .operations import warmup_and_record
            warmup_and_record(self.engine, root, self.device)
        self.busy = Lock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="recognition")
        logger.info("Initial approved model loaded: %s device=%s reason=%s",
                    self.model_version, self.device.resolved, self.device.reason)

    @bentoml.on_shutdown
    def shutdown(self):
        self.executor.shutdown(wait=True, cancel_futures=True)
        self.store.client.close()

    @bentoml.api(route="/v1/recognize", input_spec=BinaryImage, output_spec=RecognitionResponse)
    async def recognize(self, image: BinaryImage, /):
        """Recognize a static JPEG/PNG binary body. No query parameters are accepted."""
        if image.error_status:
            return JSONResponse(error_response(image.request_id, None, image.error_code),
                                status_code=image.error_status)
        if not self.busy.acquire(blocking=False):
            return JSONResponse(error_response(image.request_id, None, "model_unavailable"), status_code=503)
        state = {"image_id": None, "deadline": time.perf_counter() + 30}
        try:
            future = self.executor.submit(self._process, image, state)
        except Exception:
            self.busy.release()
            raise
        try:
            # Shield keeps the worker/lock alive after timeout: a running Torch or
            # S3 call cannot be safely killed from another Python thread.
            body, status = await asyncio.wait_for(asyncio.shield(asyncio.wrap_future(future)), timeout=30)
        except TimeoutError:
            body, status = error_response(image.request_id, state["image_id"], "inference_timeout"), 504
        logger.info("recognition request_id=%s image_id=%s http=%s status=%s",
                    image.request_id, state["image_id"], status, body.get("status", body.get("error", {}).get("code")))
        return JSONResponse(body, status_code=status)

    def _process(self, upload, state):
        started = time.perf_counter()
        try:
            image, decode_ms = decode_image(upload.body, upload.content_type)
            self._deadline(state)
            persist_start = time.perf_counter()
            try:
                state["image_id"] = self.store.save(upload.body, upload.content_type,
                                                    upload.request_id, self.model_version)
            except Exception:
                logger.warning("Original persistence failed request_id=%s", upload.request_id)
                return error_response(upload.request_id, None, "storage_unavailable"), 503
            persist_ms = (time.perf_counter() - persist_start) * 1000
            self._deadline(state)
            with DecodedImage(image) as source:
                result = self.engine.infer(source)
            self._deadline(state)
            if result["technical_candidate_limit_reached"]:
                return error_response(upload.request_id, state["image_id"], "scene_too_complex"), 422
            if result["status"] == "too_many_candidates":
                return error_response(upload.request_id, state["image_id"], "too_many_dice"), 422
            if result["status"] == "invalid_candidate_geometry":
                raise RuntimeError("Invalid candidate geometry")
            items = [{"id": f"die-{index+1}", "bbox": item["bbox_xyxy"],
                      "shape": item["shape"], "die_type": item["die_type"],
                      "face_value": item["face_value"], "confidence": item["confidence"],
                      "auto_accepted": item["auto_accepted"], "review_reasons": item["review_reasons"]}
                     for index, item in enumerate(result["items"])]
            timings = result["timings_seconds"]
            body = RecognitionResponse(
                request_id=upload.request_id, image_id=state["image_id"], status=result["status"],
                results=items, dices_count=len(items), request_requires_review=result["manual_intervention_required"],
                thresholds={}, versions=self.versions,
                timings_ms={"decode": decode_ms, "persist": persist_ms,
                            "detect": timings["detector"] * 1000,
                            "classify": timings["crop_classifier_policy"] * 1000,
                            "total": (time.perf_counter() - started) * 1000},
            ).model_dump(mode="json")
            return body, 404 if result["status"] == "no_candidates" else 200
        except InputError as exc:
            return error_response(upload.request_id, state["image_id"], exc.code), exc.status
        except TimeoutError:
            return error_response(upload.request_id, state["image_id"], "inference_timeout"), 504
        except (torch.OutOfMemoryError, MemoryError):
            logger.warning("Recognition resources unavailable request_id=%s", upload.request_id)
            return error_response(upload.request_id, state["image_id"], "model_unavailable"), 503
        except Exception:
            logger.exception("Recognition failed request_id=%s", upload.request_id)
            return error_response(upload.request_id, state["image_id"], "internal_error"), 500
        finally:
            self.busy.release()

    @staticmethod
    def _deadline(state):
        if time.perf_counter() >= state["deadline"]:
            raise TimeoutError


DiceRecognitionService.add_asgi_middleware(ContractDocumentation)
