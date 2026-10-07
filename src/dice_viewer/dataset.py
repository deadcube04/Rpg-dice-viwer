"""Audit and prepare the local dieCamera snapshot without mixing frames."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from dice_viewer import SIDES

TEST_CAMERAS = {"Triveni's iPhone (2) Camera", "Nintendo Switch Camera (057e:206d)"}
VALID_CAMERA = "HD USB Camera (05a3:9520)"
VALID_FROM = "2026-08-13"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_path(root: Path, file_name: str) -> Path:
    relative = Path(file_name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe image path: {file_name}")
    target = (root / relative).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError(f"Image escapes data root: {file_name}")
    return target


def split_for(row: dict[str, Any]) -> str:
    if row.get("split_override") in {"train", "valid", "test"}:
        return str(row["split_override"])
    if row["camera"] in TEST_CAMERAS:
        return "test"
    if row["camera"] == VALID_CAMERA and row["epoch"] >= VALID_FROM:
        return "valid"
    return "train"


def valid_value(die: dict[str, Any]) -> bool:
    die_type = str(die.get("type", ""))
    if die_type not in {f"d{sides}" for sides in SIDES}:
        return False
    value = die.get("value")
    if not isinstance(value, int):
        return False
    sides = int(die_type[1:])
    return value in range(0, 10) if sides == 10 else value in range(1, sides + 1)


def numeric_value(die: dict[str, Any]) -> int:
    value = int(die["value"])
    return 10 if die["type"] == "d10" and value == 0 else value


def normalized_box(box: dict[str, Any]) -> tuple[float, float, float, float]:
    values = tuple(float(box[key]) for key in ("x", "y", "w", "h"))
    x, y, w, h = values
    if not (0 <= x < 1 and 0 <= y < 1 and 0 < w <= 1 and 0 < h <= 1):
        raise ValueError(f"Invalid normalized box: {box}")
    if x + w > 1.02 or y + h > 1.02:
        raise ValueError(f"Box leaves image: {box}")
    return values


def pixel_box(box: dict[str, Any], width: int, height: int, padding: float = 0.0) -> tuple[int, int, int, int]:
    x, y, w, h = normalized_box(box)
    left = max(0, round((x - w * padding) * width))
    top = max(0, round((y - h * padding) * height))
    right = min(width, round((x + w * (1 + padding)) * width))
    bottom = min(height, round((y + h * (1 + padding)) * height))
    if right <= left or bottom <= top:
        raise ValueError(f"Empty crop: {box}")
    return left, top, right, bottom


def load_rows(raw: Path) -> list[dict[str, Any]]:
    manifest = raw / "metadata.jsonl"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("Empty metadata.jsonl")
    return rows


def snapshot(source: Path, target: Path) -> dict[str, Any]:
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"Snapshot target must be empty: {target}")
    rows = load_rows(source)
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / "metadata.jsonl", target / "metadata.jsonl")
    hashes: dict[str, str] = {}
    for row in rows:
        relative = row["file_name"]
        original = image_path(source, relative)
        if not original.is_file():
            raise FileNotFoundError(original)
        destination = image_path(target, relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, destination)
        hashes[relative] = sha256(destination)
    report = {"frames": len(rows), "manifest_sha256": sha256(target / "metadata.jsonl"), "images": hashes}
    (target / "snapshot.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return {"frames": len(rows), "images": len(hashes), "manifest_sha256": report["manifest_sha256"]}


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def prepare(raw: Path, out: Path) -> dict[str, Any]:
    rows = load_rows(raw)
    snapshot_report = json.loads((raw / "snapshot.json").read_text(encoding="utf-8"))
    if sha256(raw / "metadata.jsonl") != snapshot_report["manifest_sha256"]:
        raise ValueError("Snapshot manifest changed after capture")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Prepared target must be empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    crop_records: list[dict[str, Any]] = []
    frame_records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_files: set[str] = set()
    hash_split: dict[str, str] = {}
    issues: list[str] = []
    for row in rows:
        frame_id = str(row["id"])
        if frame_id in seen_ids:
            raise ValueError(f"Duplicate frame id: {frame_id}")
        seen_ids.add(frame_id)
        if row["file_name"] in seen_files:
            raise ValueError(f"Duplicate image path: {row['file_name']}")
        seen_files.add(row["file_name"])
        source = image_path(raw, row["file_name"])
        digest = sha256(source)
        if digest != snapshot_report["images"].get(row["file_name"]):
            raise ValueError(f"Snapshot image changed after capture: {row['file_name']}")
        split = split_for(row)
        if digest in hash_split and hash_split[digest] != split:
            raise ValueError(f"Exact duplicate image spans splits: {row['file_name']}")
        hash_split[digest] = split
        with Image.open(source) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        if image.size != (int(row["width"]), int(row["height"])):
            raise ValueError(f"Dimensions disagree with manifest: {row['file_name']}")
        frame_file = out / "detector" / split / "images" / f"{frame_id}.jpg"
        link_or_copy(source, frame_file)
        labels: list[str] = []
        for index, die in enumerate(row["dice"]):
            try:
                x, y, w, h = normalized_box(die["box"])
                pixel_box(die["box"], *image.size)
            except (KeyError, TypeError, ValueError) as exc:
                issues.append(f"{row['file_name']} die {index}: {exc}")
                continue
            labels.append(f"0 {x + w / 2:.8f} {y + h / 2:.8f} {w:.8f} {h:.8f}")
            if not row.get("values_trusted") or not valid_value(die):
                if row.get("values_trusted") and die.get("type") in {f"d{s}" for s in SIDES}:
                    issues.append(f"Excluded invalid reading: {row['file_name']} die {index} {die.get('type')}={die.get('value')}")
                continue
            crop = image.crop(pixel_box(die["box"], *image.size, padding=0.15))
            value = numeric_value(die)
            crop_rel = Path("reader") / split / die["type"] / str(value) / f"{frame_id}-{index}.jpg"
            crop_file = out / crop_rel
            crop_file.parent.mkdir(parents=True, exist_ok=True)
            crop.save(crop_file, quality=95)
            crop_records.append({"path": crop_rel.as_posix(), "source_file": row["file_name"], "frame_id": frame_id,
                                 "camera": row["camera"], "epoch": row["epoch"], "split": split,
                                 "type": die["type"], "value": value, "image_sha256": digest})
            counts[f"reader_{split}"] += 1
        label_file = out / "detector" / split / "labels" / f"{frame_id}.txt"
        label_file.parent.mkdir(parents=True, exist_ok=True)
        label_file.write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")
        frame_records.append({"frame_id": frame_id, "source_file": row["file_name"], "camera": row["camera"],
                              "epoch": row["epoch"], "split": split, "values_trusted": bool(row.get("values_trusted")),
                              "image_sha256": digest, "labels": len(labels)})
        counts[f"frames_{split}"] += 1
    (out / "reader_manifest.jsonl").write_text("".join(json.dumps(item) + "\n" for item in crop_records), encoding="utf-8")
    (out / "frame_manifest.jsonl").write_text("".join(json.dumps(item) + "\n" for item in frame_records), encoding="utf-8")
    (out / "detector" / "data.yaml").write_text(
        "path: .\ntrain: train/images\nval: valid/images\ntest: test/images\nnames:\n  0: die\n", encoding="utf-8"
    )
    report = {"counts": dict(counts), "issues": issues, "source_manifest_sha256": sha256(raw / "metadata.jsonl"),
              "reader_manifest_sha256": sha256(out / "reader_manifest.jsonl"), "frame_manifest_sha256": sha256(out / "frame_manifest.jsonl")}
    (out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    snap = commands.add_parser("snapshot")
    snap.add_argument("--source", type=Path, required=True)
    snap.add_argument("--target", type=Path, default=Path("data/raw"))
    prep = commands.add_parser("prepare")
    prep.add_argument("--raw", type=Path, default=Path("data/raw"))
    prep.add_argument("--out", type=Path, default=Path("data/processed"))
    args = parser.parse_args()
    result = snapshot(args.source, args.target) if args.command == "snapshot" else prepare(args.raw, args.out)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
