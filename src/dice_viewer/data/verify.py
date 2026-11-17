"""Read-only integrity inspection of a real prepared dataset (no model execution)."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from .prepare import sha256


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def verify(root):
    checksums = read_jsonl(root / "checksums.jsonl")
    indexed = {item["path"] for item in checksums}
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if actual - {"checksums.jsonl"} != indexed or len(indexed) != len(checksums):
        raise ValueError("Checksum index has missing, extra or duplicate paths")
    for item in checksums:
        path = root / item["path"]
        if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise ValueError(f"Content hash mismatch: {path}")
    source = root / "provenance" / "revisions.jsonl"
    approval = json.loads((root / "provenance" / "approval.json").read_text(encoding="utf-8"))
    if not approval["approved"] or sha256(source) != approval["manifest_sha256"]:
        raise ValueError("Source approval/hash mismatch")
    rows = read_jsonl(root / "manifests" / "canonical.jsonl")
    original = read_jsonl(source)
    if {r["image_id"] for r in rows} != {r["image_id"] for r in original}:
        raise ValueError("Canonical inventory differs from approved source")
    by_id = {r["image_id"]: r for r in rows}
    maps = json.loads((root / "manifests" / "label_maps.json").read_text(encoding="utf-8"))
    if len(by_id) != len(rows):
        raise ValueError("Duplicate canonical image ID")
    groups = defaultdict(set)
    source_hashes = defaultdict(set)
    pixels = defaultdict(set)
    for row in rows:
        split = row["split"]
        if split not in ("train", "val", "e2"):
            continue
        if row["source"] in approval["excluded_sources"]:
            raise ValueError("Excluded source reached a model partition")
        count = len(row["instances"])
        if (split in ("train", "val") and count != 1) or (split == "e2" and not 2 <= count <= 8):
            raise ValueError("E1/E2 instance count violation")
        groups[row["group_id"]].add(split)
        source_hashes[row["sha256"]].add(split)
        pixels[row["pixel_sha256"]].add(split)
        for inst in row["instances"]:
            for mask, target_id, target, mapping in (("type_known", "type_id", "die_type", "types"),
                                                       ("value_known", "token_id", "face_token", "tokens")):
                if inst[mask] and maps[mapping].get(str(inst[target_id])) != inst[target]:
                    raise ValueError("Canonical target ID differs from label map")
        image_path = root / row["prepared_image"]
        with Image.open(image_path) as image:
            if image.size != (row["width"], row["height"]) or image.mode != "RGB":
                raise ValueError("Prepared image size/mode mismatch")
        label_path = root / "yolo" / "labels" / split / (row["image_id"] + ".txt")
        labels = label_path.read_text(encoding="utf-8").splitlines()
        if len(labels) != count:
            raise ValueError("YOLO instance count mismatch")
        for label, inst in zip(labels, row["instances"]):
            values = label.split()
            x1, y1, x2, y2 = inst["bbox_xyxy"]
            expected = ((x1+x2)/2/row["width"], (y1+y2)/2/row["height"],
                        (x2-x1)/row["width"], (y2-y1)/row["height"])
            if len(values) != 5 or values[0] != "0" or any(abs(float(v) - e) > 1e-10 for v, e in zip(values[1:], expected)):
                raise ValueError("YOLO coordinate/class mismatch")
    if any(len(v) > 1 for mapping in (groups, source_hashes, pixels) for v in mapping.values()):
        raise ValueError("Known overlap between model partitions")
    crop_count = 0
    for split in ("train", "val", "e2", "negative"):
        split_rows = read_jsonl(root / "splits" / (split + ".jsonl"))
        listed_images = (root / "yolo" / (split + ".txt")).read_text(encoding="utf-8").splitlines()
        expected_images = ["./images/" + split + "/" + r["image_id"] + ".png" for r in split_rows]
        if listed_images != expected_images:
            raise ValueError("YOLO image list differs from partition manifest")
        if {r["image_id"] for r in split_rows} != {r["image_id"] for r in rows if r["split"] == split}:
            raise ValueError("Partition manifest mismatch")
        for row in split_rows:
            if row != by_id[row["image_id"]]:
                raise ValueError("Partition annotations differ from canonical manifest")
        crop_rows = read_jsonl(root / "classifier" / (split + ".jsonl"))
        expected_crops = {(r["image_id"], inst["instance_id"])
                          for r in split_rows for inst in r["instances"] if inst["crop_eligible"]}
        observed_crops = [(c["image_id"], c["instance_id"]) for c in crop_rows]
        if len(observed_crops) != len(set(observed_crops)) or set(observed_crops) != expected_crops:
            raise ValueError("Missing, extra or duplicate classifier crops")
        for crop in crop_rows:
            parent = by_id[crop["image_id"]]
            inst = next(i for i in parent["instances"] if i["instance_id"] == crop["instance_id"])
            if crop["split"] != parent["split"] or crop["group_id"] != parent["group_id"]:
                raise ValueError("Crop partition/group mismatch")
            if (crop["type_mask"], crop["value_mask"]) != (inst["type_known"], inst["value_known"]):
                raise ValueError("Crop supervision mask mismatch")
            for field in ("die_type", "face_token", "face_value", "type_id", "token_id"):
                if crop[field] != inst[field]:
                    raise ValueError(f"Crop target differs from canonical annotation: {field}")
            with Image.open(root / crop["path"]) as image:
                if image.size != (224, 224) or image.mode != "RGB":
                    raise ValueError("Crop size/mode mismatch")
            crop_count += 1
    return {"status": "integrity_verified", "files_hashed": len(checksums),
            "canonical_images": len(rows), "images_by_partition": dict(Counter(r["split"] for r in rows)),
            "crops_inspected": crop_count, "known_cross_partition_overlap": 0,
            "source_manifest_sha256": sha256(source), "checksum_index_sha256": sha256(root / "checksums.jsonl"),
            "verifier_sha256": sha256(Path(__file__)),
            "scope": "Real exported data integrity and format inspection; no automated software tests or model evaluation"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.dataset), indent=2))
