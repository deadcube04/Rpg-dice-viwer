"""Prepare Stage 2 artifacts, without loading models or changing audit state.

Run from the repository with PYTHONPATH=src and python -m dice_viewer.data.prepare.
Only an explicitly approved snapshot can be used. Output directories are never reused.
"""

import argparse
import csv
import hashlib
import io
import json
import math
import platform
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import PIL
from PIL import Image, ImageOps

from .grouping import assign_splits, build_groups, fingerprint
from .schema import LEGAL_TOKENS, TOKENS, TYPES, canonical_instance, image_role


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_image(archive, row):
    content = archive.read(row["path"])
    if hashlib.sha256(content).hexdigest() != row["sha256"]:
        raise ValueError(f"Image does not match approved snapshot: {row['image_id']}")
    with Image.open(io.BytesIO(content)) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        image.load()
    if image.size != (row["width"], row["height"]):
        raise ValueError(f"Oriented size mismatch: {row['image_id']}")
    return image


def make_crop(image, box, margin=.1, size=224):
    x1, y1, x2, y2 = box
    dx, dy = (x2 - x1) * margin, (y2 - y1) * margin
    bounds = (max(0, math.floor(x1 - dx)), max(0, math.floor(y1 - dy)),
              min(image.width, math.ceil(x2 + dx)), min(image.height, math.ceil(y2 + dy)))
    crop = image.crop(bounds)
    scale = size / max(crop.size)
    resized = (max(1, round(crop.width * scale)), max(1, round(crop.height * scale)))
    offset = ((size - resized[0]) // 2, (size - resized[1]) // 2)
    padded = Image.new("RGB", (size, size), (0, 0, 0))
    padded.paste(crop.resize(resized, Image.Resampling.BILINEAR), offset)
    return padded, {"source_crop_xyxy": bounds, "resized_size": resized, "padding_left_top": offset}


def export_assets(archive, rows, output):
    crops = []
    assets = []
    for index, row in enumerate(rows):
        split = row["split"]
        if split not in ("train", "val", "e2", "negative"):
            continue
        image = read_image(archive, row)
        image_relative = Path("yolo") / "images" / split / (row["image_id"] + ".png")
        image_path = output / image_relative
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(image_path)
        row["prepared_image"] = image_relative.as_posix()
        row["prepared_image_sha256"] = sha256(image_path)
        labels = []
        for inst in row["instances"]:
            if not inst["detection_known"]:
                raise ValueError("Incomplete detection supervision reached YOLO export")
            x1, y1, x2, y2 = inst["bbox_xyxy"]
            labels.append(f"0 {(x1+x2)/2/row['width']:.12f} {(y1+y2)/2/row['height']:.12f} "
                          f"{(x2-x1)/row['width']:.12f} {(y2-y1)/row['height']:.12f}")
            if not inst["crop_eligible"]:
                continue
            crop, transform = make_crop(image, inst["bbox_xyxy"])
            # A content-derived ID avoids unsafe or platform-specific instance names.
            crop_id = hashlib.sha256((row["image_id"] + ":" + inst["instance_id"]).encode()).hexdigest()[:24]
            crop_relative = Path("classifier") / "crops" / split / (crop_id + ".png")
            crop_path = output / crop_relative
            crop_path.parent.mkdir(parents=True, exist_ok=True)
            crop.save(crop_path)
            crops.append({
                "crop_id": crop_id, "image_id": row["image_id"], "instance_id": inst["instance_id"],
                "group_id": row["group_id"], "split": split, "path": crop_relative.as_posix(),
                "sha256": sha256(crop_path), "image_sha256": row["sha256"],
                "bbox_xyxy": inst["bbox_xyxy"], "die_type": inst["die_type"],
                "face_token": inst["face_token"], "face_value": inst["face_value"],
                "type_id": inst["type_id"], "token_id": inst["token_id"],
                "type_mask": inst["type_known"], "value_mask": inst["value_known"],
                "modality": inst.get("modality", "unknown"), "convention": inst.get("convention", ""),
                "transform": transform, "preprocessing_version": "body-margin10-letterbox224-v1",
            })
        label_relative = Path("yolo") / "labels" / split / (row["image_id"] + ".txt")
        label_path = output / label_relative
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")
        assets.extend([
            {"path": image_relative.as_posix(), "sha256": row["prepared_image_sha256"]},
            {"path": label_relative.as_posix(), "sha256": sha256(label_path)},
        ])
        if index % 500 == 0:
            print(f"Export progress {index}/{len(rows)}", flush=True)
    for crop in crops:
        assets.append({"path": crop["path"], "sha256": crop["sha256"]})
    for split in ("train", "val", "e2", "negative"):
        # Relative entries are resolved from each list's parent by Ultralytics.
        selected = [row for row in rows if row["split"] == split]
        (output / "yolo").mkdir(exist_ok=True)
        (output / "yolo" / f"{split}.txt").write_text(
            "".join("./images/" + split + "/" + row["image_id"] + ".png\n" for row in selected), encoding="utf-8")
        write_jsonl(output / "classifier" / f"{split}.jsonl", (c for c in crops if c["split"] == split))
    yolo_root = (output / "yolo").resolve().as_posix()
    # This config reserves E2 as test; training must never evaluate this split for selection.
    (output / "yolo" / "dataset.yaml").write_text(
        f"path: {json.dumps(yolo_root)}\ntrain: train.txt\nval: val.txt\ntest: e2.txt\nnames:\n  0: die\n", encoding="utf-8")
    write_jsonl(output / "manifests" / "assets.jsonl", assets)
    return crops


def create_reports(rows, all_rows, crops, groups, split_recipe, output):
    image_counts = Counter(row["split"] for row in all_rows)
    group_partitions = defaultdict(set)
    pixel_partitions = defaultdict(set)
    byte_partitions = defaultdict(set)
    for row in rows:
        if row["split"] in ("train", "val", "e2"):
            group_partitions[row["group_id"]].add(row["split"])
            pixel_partitions[row["pixel_sha256"]].add(row["split"])
            byte_partitions[row["sha256"]].add(row["split"])
    leakage = {name: {key: sorted(value) for key, value in mapping.items() if len(value) > 1}
               for name, mapping in (("groups", group_partitions), ("pixels", pixel_partitions), ("bytes", byte_partitions))}
    if any(leakage.values()):
        raise ValueError(f"Cross-partition overlap: {leakage}")
    write_json(output / "reports" / "leakage.json", {
        "known_overlap": leakage, "known_overlap_count": 0,
        "scope": "byte/pixel equality, inferred filenames and sessions, conservative perceptual candidates",
        "not_proven": "physical die identity, unknown-camera sessions and semantic independence beyond the declared heuristics",
    })
    coverage = Counter()
    modalities = Counter()
    for row in all_rows:
        for inst in row["instances"]:
            coverage[(row["split"], inst["die_type"] or "unknown", inst["face_token"] or "unknown")] += 1
            modalities[(row["split"], inst.get("modality", "unknown"))] += 1
    write_csv(output / "reports" / "coverage.csv", ["split", "die_type", "face_token", "instances"],
              [dict(zip(("split", "die_type", "face_token", "instances"), (*key, n))) for key, n in sorted(coverage.items())])
    write_csv(output / "reports" / "modalities.csv", ["split", "modality", "instances"],
              [dict(zip(("split", "modality", "instances"), (*key, n))) for key, n in sorted(modalities.items())])
    breakdown = Counter((r["split"], r["source"], len(r["instances"])) for r in all_rows)
    write_csv(output / "reports" / "sources_and_counts.csv", ["split", "source", "dice_count", "images"],
              [dict(zip(("split", "source", "dice_count", "images"), (*key, n))) for key, n in sorted(breakdown.items())])
    usable = Counter((c["split"], c["die_type"], c["face_token"]) for c in crops if c["type_mask"] and c["value_mask"])
    support = []
    for kind in TYPES:
        for token in LEGAL_TOKENS[kind]:
            support.append({"die_type": kind, "face_token": token,
                            **{split: usable[(split, kind, token)] for split in ("train", "val", "e2")}})
    write_csv(output / "reports" / "joint_support.csv", ["die_type", "face_token", "train", "val", "e2"], support)
    missing_train = [f"{r['die_type']}:{r['face_token']}" for r in support if not r["train"]]
    missing_val = [f"{r['die_type']}:{r['face_token']}" for r in support if not r["val"]]
    missing_e2 = [f"{r['die_type']}:{r['face_token']}" for r in support if not r["e2"]]
    reasons = Counter(r["reason"] for r in all_rows if r["split"] not in ("train", "val", "e2"))
    group_sizes = Counter(r["group_id"] for r in rows)
    near = [edge for edge in groups.edges if edge["reason"] == "potential_near_duplicate"]
    crop_counts = Counter(c["split"] for c in crops)
    unknown_modality = sum(1 for r in rows if r["split"] in ("train", "val", "e2")
                           for i in r["instances"] if i.get("modality", "unknown") == "unknown")
    summary = {
        "stage": 2, "status": "prepared_awaiting_user_review", "prepared_dataset_approved": False,
        "images_by_partition": dict(sorted(image_counts.items())),
        "classifier_crops_by_partition": dict(sorted(crop_counts.items())),
        "e1_validation_fraction": image_counts["val"] / (image_counts["train"] + image_counts["val"]),
        "groups": {"total_active": len(group_sizes), "largest": group_sizes.most_common(12),
                   "evidence_counts": dict(Counter(e["reason"] for e in groups.edges)),
                   "potential_near_duplicate_edges": len(near)},
        "split_recipe": split_recipe, "exclusion_reasons": dict(reasons),
        "missing_joint_training_support": missing_train,
        "missing_joint_validation_support": missing_val,
        "missing_joint_e2_support": missing_e2,
        "instances_with_unknown_modality_in_export": unknown_modality,
        "limitations": [
            "Perceptual links are potential duplicates and conservatively kept together; no semantic duplicate confirmation is claimed.",
            "Capture sessions inferred from filenames/timestamps with 30-minute maximum adjacent gap; camera and physical die identity may be unknown.",
            "Whole inferred sequence families are grouped conservatively; 80/20 is approximate and grouped balance takes priority.",
            "E1 members related to E2 are quarantined; no crops from group scenes enter E1 training or validation.",
            "Conflicting exact-pixel annotations are quarantined; the approved review is not silently edited.",
            "Unknown modality remains unknown; d6 pips coverage cannot be inferred from the type alone.",
            "Missing joint support and separate type/value masks must be respected in training and reporting.",
            "No model has been trained or evaluated. Dataset preparation does not authorize Stage 3.",
            "Generalization to unseen physical dice or collections is not established by these splits.",
        ],
    }
    write_json(output / "reports" / "summary.json", summary)
    lines = ["ETAPA 2 — DATASET PREPARADO PARA CONFERENCIA", "",
             "Snapshot de auditoria aprovado pelo usuario; dataset preparado ainda aguarda aprovacao.",
             "Particoes: " + json.dumps(summary["images_by_partition"], ensure_ascii=False),
             "Recortes: " + json.dumps(summary["classifier_crops_by_partition"], ensure_ascii=False),
             f"Validacao E1: {summary['e1_validation_fraction']:.2%} das imagens E1 exportadas.",
             "Sobreposicoes conhecidas entre treino/validacao/E2: 0.",
             "Combinacoes tipo:token sem supervisao conjunta no treino: " + ", ".join(missing_train),
             "", "LIMITACOES", *summary["limitations"], "",
             "Arquivos: canonical.jsonl; splits/*.jsonl; reports/*; yolo/dataset.yaml; classifier/*.jsonl.",
             "Uso local. Creditos e licencas das fontes permanecem no ZIP original; dados nao foram publicados."]
    (output / "reports" / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def prepare(args):
    snapshot = Path(args.snapshot).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise ValueError("Output already exists; use a new version to preserve prior artifacts")
    manifest_bytes = (snapshot / "revisions.jsonl").read_bytes()
    meta_bytes = (snapshot / "snapshot.json").read_bytes()
    meta = json.loads(meta_bytes)
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    if manifest_hash != meta["manifest_sha256"]:
        raise ValueError("Snapshot manifest hash mismatch")
    all_rows = [json.loads(line) for line in manifest_bytes.splitlines() if line.strip()]
    if len(all_rows) != meta["frames"] or len({r["image_id"] for r in all_rows}) != len(all_rows):
        raise ValueError("Snapshot count mismatch or duplicate image IDs")
    archive_hash = sha256(args.archive)
    if {r["archive_sha256"] for r in all_rows} != {archive_hash}:
        raise ValueError("Original archive hash mismatch")
    output.mkdir(parents=True)
    provenance = output / "provenance"
    provenance.mkdir()
    (provenance / "revisions.jsonl").write_bytes(manifest_bytes)
    (provenance / "snapshot.json").write_bytes(meta_bytes)
    write_json(provenance / "approval.json", {
        "approved": True, "scope": "source_audit_snapshot_and_execution_of_stage_2_only",
        "approved_by": args.approved_by, "approval_reference": args.approval_reference,
        "approval_date": "2026-11-17", "manifest_sha256": manifest_hash,
        "original_snapshot_sha256": hashlib.sha256(meta_bytes).hexdigest(),
        "excluded_sources": sorted(args.exclude_source), "archive_sha256": archive_hash,
        "source_snapshot": str(snapshot), "prepared_dataset_approved": False,
    })
    source_copy = provenance / "preparation_code"
    source_copy.mkdir()
    for path in Path(__file__).parent.glob("*.py"):
        shutil.copyfile(path, source_copy / path.name)
    recipe = {
        "schema_version": "1.0", "seed": args.seed, "e1_target_train_val": [.8, .2],
        "excluded_sources": sorted(args.exclude_source), "archive": str(Path(args.archive).resolve()),
        "session_max_adjacent_gap_seconds": 1800, "potential_near_duplicates": {"dhash_bits": 128, "max_hamming": 6, "rgb32_max_mae": 12},
        "preprocessing": {"version": "body-margin10-letterbox224-v1", "crop_size": [224, 224],
                          "margin_per_side": .1, "clip_to_image": True, "padding_rgb": [0, 0, 0],
                          "interpolation": "bilinear", "channels": "RGB", "output_dtype": "uint8",
                          "normalization": "none; tensor/ImageNet normalization belongs to Stage 3"},
        "software": {"python": platform.python_version(), "pillow": PIL.__version__},
        "invocation": sys.argv, "started_at": datetime.now(timezone.utc).isoformat(),
        "approved_source_hash": manifest_hash,
    }
    write_json(provenance / "recipe.json", recipe)
    fingerprints = {}
    rows = []
    with zipfile.ZipFile(args.archive) as archive:
        for index, row in enumerate(all_rows):
            if row["source"] in set(args.exclude_source):
                row.update(role="excluded_source", original_role="excluded_source", split="excluded_source",
                           reason="source_excluded_by_user", group_id="excluded-" + row["image_id"])
                continue
            row["instances"] = [canonical_instance(i, row["width"], row["height"]) for i in row["instances"]]
            row["role"], row["reason"] = image_role(row, set(args.exclude_source))
            row["original_role"] = row["role"]
            image = read_image(archive, row)
            fingerprints[row["image_id"]] = fingerprint(image)
            row["pixel_sha256"] = fingerprints[row["image_id"]][0]
            row["dhash128"] = f"{fingerprints[row['image_id']][1]:032x}"
            row["approved_source_manifest_sha256"] = manifest_hash
            rows.append(row)
            if index % 500 == 0:
                print(f"Fingerprint progress {index}/{len(all_rows)}", flush=True)
        print("Grouping all active images", flush=True)
        groups = build_groups(rows, fingerprints)
        split_recipe = assign_splits(rows, args.seed)
        write_jsonl(output / "manifests" / "group_evidence.jsonl", groups.edges)
        # Partitions are recorded BEFORE any crop or training-format export.
        for split in sorted({r["split"] for r in all_rows}):
            write_jsonl(output / "splits" / f"{split}.jsonl", (r for r in all_rows if r["split"] == split))
        print("Partitions frozen; exporting images, YOLO and classifier crops", flush=True)
        crops = export_assets(archive, rows, output)
    write_jsonl(output / "manifests" / "canonical.jsonl", all_rows)
    for split in sorted({r["split"] for r in all_rows}):
        write_jsonl(output / "splits" / f"{split}.jsonl", (r for r in all_rows if r["split"] == split))
    write_json(output / "manifests" / "label_maps.json", {
        "detector": {"0": "die"}, "types": {str(i): x for i, x in enumerate(TYPES)},
        "tokens": {str(i): x for i, x in enumerate(TOKENS)}, "legal_tokens_by_type": LEGAL_TOKENS,
        "zero_normalization": {"0": 10, "00": 10}, "unknown_target": None,
        "loss_masks": "independent type_mask and value_mask; do not train on null labels",
    })
    summary = create_reports(rows, all_rows, crops, groups, split_recipe, output)
    # Content-address every deliverable except this index itself.
    digests = [{"path": p.relative_to(output).as_posix(), "sha256": sha256(p), "bytes": p.stat().st_size}
               for p in sorted(output.rglob("*")) if p.is_file()]
    write_jsonl(output / "checksums.jsonl", digests)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--approved-by", required=True)
    parser.add_argument("--approval-reference", required=True)
    parser.add_argument("--exclude-source", action="append", default=[])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare(args)


if __name__ == "__main__":
    main()
