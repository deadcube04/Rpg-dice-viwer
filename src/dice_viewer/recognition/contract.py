"""Exact public V1 recognition fields; internal tokens never leave the service."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PublicModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Confidence(PublicModel):
    detection: float | None = Field(ge=0, le=1)
    type: float | None = Field(ge=0, le=1)
    value: float | None = Field(ge=0, le=1)


class DieResult(PublicModel):
    id: str
    bbox: tuple[float, float, float, float]
    shape: Literal["d4", "d6", "d8", "d10", "d12", "d20"] | None
    die_type: Literal["d4", "d6", "d8", "d10", "d10_tens", "d12", "d20"] | None
    face_value: int | None
    confidence: Confidence
    auto_accepted: bool
    review_reasons: list[str]


class Versions(PublicModel):
    detector: str
    classifier: str
    labels: str
    preprocessing: str


class Timings(PublicModel):
    decode: float = Field(ge=0)
    persist: float = Field(ge=0)
    detect: float = Field(ge=0)
    classify: float = Field(ge=0)
    total: float = Field(ge=0)


class Thresholds(PublicModel):
    detection: Literal[0.95] = 0.95
    type: Literal[0.8] = 0.8
    value: Literal[0.8] = 0.8


class RecognitionResponse(PublicModel):
    request_id: str
    image_id: str
    status: Literal["ok", "needs_review", "no_candidates"]
    results: list[DieResult] = Field(max_length=8)
    dices_count: int = Field(ge=0, le=8)
    request_requires_review: bool
    thresholds: Thresholds
    versions: Versions
    timings_ms: Timings


MESSAGES = {
    "invalid_request": "Requisição incompatível com o contrato.",
    "unsupported_media_type": "Envie uma imagem JPEG ou PNG em corpo binário direto.",
    "image_too_large": "A imagem excede os limites de bytes, dimensões ou pixels.",
    "invalid_image": "Não foi possível decodificar a imagem estática enviada.",
    "storage_unavailable": "Não foi possível confirmar a persistência da imagem.",
    "model_unavailable": "O modelo está temporariamente indisponível.",
    "inference_timeout": "O prazo de processamento foi excedido.",
    "too_many_dice": "Foram detectados mais de oito candidatos; simplifique a cena.",
    "scene_too_complex": "O limite técnico de candidatos foi atingido; simplifique a cena.",
    "internal_error": "Não foi possível concluir o processamento.",
}


def error_response(request_id, image_id, code):
    return {"request_id": request_id, "image_id": image_id,
            "error": {"code": code, "message": MESSAGES[code],
                      "retryable": code in {"storage_unavailable", "model_unavailable", "inference_timeout"}}}
