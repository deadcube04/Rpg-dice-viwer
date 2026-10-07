"""Local annotation review API. The original snapshot is never modified."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, model_validator
from PIL import Image, ImageOps, UnidentifiedImageError

from dice_viewer.dataset import image_path, load_rows, sha256, split_for

ROOT = Path(os.environ.get("DICE_AUDIT_ROOT", "data/raw")).resolve()
STATE = Path(os.environ.get("DICE_AUDIT_STATE", "runtime/audit.json")).resolve()
EXPORTS = Path(os.environ.get("DICE_AUDIT_EXPORTS", "data")).resolve()
UPLOADS = Path(os.environ.get("DICE_AUDIT_UPLOADS", "runtime/audit_uploads")).resolve()
LOCK = threading.RLock()
app = FastAPI(title="Dice dataset audit")
app.add_middleware(CORSMiddleware, allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
                   allow_methods=["GET", "PUT", "POST"], allow_headers=["Content-Type"])


class Box(BaseModel):
    x: float = Field(ge=0, lt=1)
    y: float = Field(ge=0, lt=1)
    w: float = Field(gt=0, le=1)
    h: float = Field(gt=0, le=1)

    @model_validator(mode="after")
    def within_image(self) -> "Box":
        if self.x + self.w > 1.02 or self.y + self.h > 1.02:
            raise ValueError("A caixa ultrapassa o limite aceito da imagem")
        return self


class Die(BaseModel):
    type: Literal["d4", "d6", "d8", "d10", "d12", "d20"]
    value: int | None = None
    box: Box

class Review(BaseModel):
    dice: list[Die]
    values_trusted: bool
    note: str = Field(default="", max_length=1000)

    @model_validator(mode="after")
    def complete_labels(self) -> "Review":
        if self.values_trusted:
            for die in self.dice:
                sides = int(die.type[1:])
                if die.value is None or not (die.value == 0 and sides == 10) and not 1 <= die.value <= sides:
                    raise ValueError("Corrija os valores ou marque a leitura como não confiável")
        return self


def source_rows() -> list[dict]:
    return load_rows(ROOT)


def read_state() -> dict:
    if not STATE.exists():
        return {"source_sha256": sha256(ROOT / "metadata.jsonl"), "reviews": {}, "uploads": {}}
    state = json.loads(STATE.read_text(encoding="utf-8"))
    if state["source_sha256"] != sha256(ROOT / "metadata.jsonl"):
        raise RuntimeError("O snapshot de origem mudou; a auditoria precisa de migração explícita")
    state.setdefault("uploads", {})
    return state


def write_state(state: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATE.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(STATE)


def all_rows(state: dict) -> list[dict]:
    return [*reversed(list(state["uploads"].values())), *source_rows()]


def find_frame(frame_id: str) -> dict:
    with LOCK:
        state = read_state()
    for row in all_rows(state):
        if str(row["id"]) == frame_id:
            return row
    raise HTTPException(status_code=404, detail="Imagem não encontrada")


def present_frame(row: dict, reviews: dict) -> dict:
    review = reviews.get(str(row["id"]))
    return {"id": str(row["id"]), "file_name": row["file_name"], "camera": row["camera"],
            "epoch": row["epoch"], "width": row["width"], "height": row["height"],
            "split": split_for(row), "reviewed": review is not None,
            "values_trusted": review["values_trusted"] if review else bool(row.get("values_trusted")),
            "note": review["note"] if review else "",
            "dice": review["dice"] if review else [{"type": die.get("type"), "value": die.get("value"), "box": die["box"]}
                                                   for die in row["dice"]]}


@app.get("/api/frames")
def frames(split: Literal["all", "train", "valid", "test"] = "train",
           status: Literal["all", "pending", "reviewed"] = "all", camera: str = "",
           offset: int = Query(0, ge=0), limit: int = Query(40, ge=1, le=100)) -> dict:
    with LOCK:
        state = read_state()
    rows = [present_frame(row, state["reviews"]) for row in all_rows(state)]
    filtered = [row for row in rows if (split == "all" or row["split"] == split)
                and (status == "all" or row["reviewed"] == (status == "reviewed"))
                and (not camera or camera.lower() in row["camera"].lower())]
    return {"items": filtered[offset:offset + limit], "total": len(filtered), "offset": offset,
            "cameras": sorted({row["camera"] for row in rows})}


@app.get("/api/frames/{frame_id}")
def frame(frame_id: str) -> dict:
    with LOCK:
        reviews = read_state()["reviews"]
    return present_frame(find_frame(frame_id), reviews)


@app.get("/api/frames/{frame_id}/image")
def image(frame_id: str) -> FileResponse:
    row = find_frame(frame_id)
    source = UPLOADS / f"{frame_id}.jpg" if row.get("label_source") == "audit_upload" else image_path(ROOT, row["file_name"])
    if not source.is_file():
        raise HTTPException(status_code=404, detail="Arquivo de imagem ausente")
    return FileResponse(source)


@app.post("/api/frames")
async def upload_frame(file: UploadFile = File(...), camera: str = Form(...),
                       split: Literal["train", "valid", "test"] = Form("train")) -> dict:
    camera = camera.strip()
    if not camera or len(camera) > 100:
        raise HTTPException(status_code=422, detail="Informe uma câmera com até 100 caracteres")
    if file.content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(status_code=422, detail="Use JPEG ou PNG")
    body = await file.read(10 * 1024 * 1024 + 1)
    if not body or len(body) > 10 * 1024 * 1024:
        raise HTTPException(status_code=422, detail="Imagem vazia ou maior que 10 MB")
    try:
        with Image.open(io.BytesIO(body)) as opened:
            if opened.format not in {"JPEG", "PNG"} or opened.width * opened.height > 25_000_000:
                raise ValueError("Imagem inválida ou grande demais")
            normalized = ImageOps.exif_transpose(opened).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise HTTPException(status_code=422, detail="Imagem inválida") from exc
    frame_id = f"audit-{uuid.uuid4().hex}"
    UPLOADS.mkdir(parents=True, exist_ok=True)
    destination = UPLOADS / f"{frame_id}.jpg"
    normalized.save(destination, format="JPEG", quality=95)
    row = {"id": frame_id, "file_name": f"uploads/{frame_id}.jpg", "camera": camera,
           "camera_role": "", "epoch": datetime.now(timezone.utc).date().isoformat(),
           "captured_at": datetime.now(timezone.utc).isoformat(), "width": normalized.width,
           "height": normalized.height, "label_source": "audit_upload", "values_trusted": False,
           "dice": [], "split_override": split}
    try:
        with LOCK:
            state = read_state()
            state["uploads"][frame_id] = row
            write_state(state)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    return present_frame(row, {})


@app.put("/api/frames/{frame_id}/review")
def save_review(frame_id: str, review: Review) -> dict:
    find_frame(frame_id)
    with LOCK:
        state = read_state()
        state["reviews"][frame_id] = {**review.model_dump(), "reviewed_at": datetime.now(timezone.utc).isoformat()}
        write_state(state)
    return frame(frame_id)


@app.post("/api/exports")
def export_snapshot() -> dict:
    with LOCK:
        state = read_state()
        rows = all_rows(state)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"-{uuid.uuid4().hex[:6]}"
        target = EXPORTS / f"audit-{stamp}"
        if target.exists():
            raise HTTPException(status_code=409, detail="Exportação com esse nome já existe")
        target.mkdir(parents=True)
        image_hashes: dict[str, str] = {}
        output_rows = []
        for row in rows:
            updated = dict(row)
            review = state["reviews"].get(str(row["id"]))
            if review:
                dice = []
                for edited in review["dice"]:
                    original = next((item for item in row["dice"] if item.get("type") == edited["type"]
                                     and item.get("box") == edited["box"]), None)
                    dice.append({**(original or {}), **edited})
                updated["dice"] = dice
                updated["values_trusted"] = review["values_trusted"]
                updated["label_source"] = "audit"
            source = UPLOADS / f"{row['id']}.jpg" if row.get("label_source") == "audit_upload" else image_path(ROOT, row["file_name"])
            destination = image_path(target, row["file_name"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(source, destination)
            except OSError:
                shutil.copy2(source, destination)
            image_hashes[row["file_name"]] = sha256(destination)
            output_rows.append(updated)
        manifest = target / "metadata.jsonl"
        manifest.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                                    for row in output_rows), encoding="utf-8")
        (target / "snapshot.json").write_text(json.dumps({"frames": len(output_rows),
            "manifest_sha256": sha256(manifest), "images": image_hashes}, indent=2), encoding="utf-8")
        provenance = {"source_manifest_sha256": state["source_sha256"],
                      "audit_state_sha256": hashlib.sha256(json.dumps(state, sort_keys=True).encode("utf-8")).hexdigest(),
                      "reviewed_frames": len(state["reviews"]), "uploaded_frames": len(state["uploads"]),
                      "exported_at": datetime.now(timezone.utc).isoformat(),
                      "dataset_source": "https://huggingface.co/datasets/G-G-Games/diecamera-frames"}
        (target / "audit_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
        return {"path": str(target), "frames": len(output_rows), "reviewed": len(state["reviews"]),
                "manifest_sha256": sha256(manifest)}


def main() -> None:
    import uvicorn
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
