"""Conservative grouping before any split or crop is generated."""

import hashlib
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import PurePosixPath

from PIL import Image, ImageChops, ImageStat


class Groups:
    def __init__(self, ids):
        self.parent = {key: key for key in ids}
        self.edges = []

    def root(self, key):
        while self.parent[key] != key:
            self.parent[key] = self.parent[self.parent[key]]
            key = self.parent[key]
        return key

    def link(self, left, right, reason, **details):
        a, b = self.root(left), self.root(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)
        self.edges.append({"left": left, "right": right, "reason": reason, **details})

    def link_buckets(self, buckets, reason):
        for key, ids in sorted(buckets.items()):
            for other in ids[1:]:
                self.link(ids[0], other, reason, key=key)


def fingerprint(image: Image.Image) -> tuple[str, int, Image.Image]:
    """Exact oriented RGB pixels plus 128-bit horizontal/vertical difference hash."""
    digest = hashlib.sha256(
        f"{image.width}x{image.height}:RGB:".encode() + image.tobytes()
    ).hexdigest()
    small = image.convert("L").resize((9, 9), Image.Resampling.LANCZOS)
    pixels = list(small.get_flattened_data())
    bits = 0
    for y in range(8):
        for x in range(8):
            bits = (bits << 1) | (pixels[y * 9 + x] > pixels[y * 9 + x + 1])
            bits = (bits << 1) | (pixels[y * 9 + x] > pixels[(y + 1) * 9 + x])
    return digest, bits, image.resize((32, 32), Image.Resampling.LANCZOS)


def filename_keys(row):
    """Metadata inferred from names stays marked inferred, never physical identity."""
    name = PurePosixPath(row["path"]).stem
    if row["source"].startswith("roboflow/"):
        original = re.split(r"\.rf\.", name)[0]
        original = re.sub(r"^[0-9a-f]{8}-", "", original, flags=re.I)
        original = re.sub(r"_(jpg|jpeg|png)$", "", original, flags=re.I)
        original = re.sub(r"_(\d+)$", "", original) if re.match(r"^\d{8}_\d{6}", original) else original
        result = {"augmentation_family": original}
        timestamp = re.match(r"^(\d{8}_\d{6})", original)
        if timestamp:
            result["session_clock"] = datetime.strptime(timestamp[1], "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc).timestamp()
            result["session_domain"] = "roboflow_timestamp_unknown_camera"
        else:
            sequence = re.match(r"^(.*?)[_-]?(\d+)$", original)
            if sequence:
                result["sequence_family"] = sequence[1].rstrip("_-").lower()
        return result
    timestamp = re.match(r"^[a-z]?(\d{13})", name)
    result = {"capture_frame": name}
    if timestamp:
        result.update(session_clock=int(timestamp[1]) / 1000,
                      session_domain="local_capture_unknown_camera")
    return result


def build_groups(rows, fingerprints, session_gap=1800, max_hamming=6, max_mae=12):
    groups = Groups(r["image_id"] for r in rows)
    buckets = {kind: defaultdict(list) for kind in
               ("byte_sha256", "pixel_sha256", "augmentation_family", "sequence_family", "capture_frame")}
    sessions = defaultdict(list)
    for row in rows:
        image_id = row["image_id"]
        keys = filename_keys(row)
        row["inferred_group_metadata"] = keys
        buckets["byte_sha256"][row["sha256"]].append(image_id)
        buckets["pixel_sha256"][fingerprints[image_id][0]].append(image_id)
        for kind in ("augmentation_family", "sequence_family", "capture_frame"):
            if keys.get(kind):
                buckets[kind][keys[kind]].append(image_id)
        if "session_clock" in keys:
            sessions[keys["session_domain"]].append((keys["session_clock"], image_id))
    for reason, values in buckets.items():
        groups.link_buckets(values, reason)
    for domain, values in sorted(sessions.items()):
        values.sort()
        session_id = None
        previous = None
        for timestamp, image_id in values:
            if previous and timestamp - previous[0] <= session_gap:
                groups.link(previous[1], image_id, "inferred_capture_session", domain=domain,
                            gap_seconds=timestamp - previous[0])
            else:
                session_id = f"{domain}:{int(timestamp)}"
            next(r for r in rows if r["image_id"] == image_id)["capture_session_id"] = session_id
            previous = timestamp, image_id
    # Search all active pairs; no source/split restriction can hide cross-source duplicates.
    ids = sorted(fingerprints)
    for index, left in enumerate(ids):
        left_pixel, left_hash, left_thumb = fingerprints[left]
        for right in ids[index + 1:]:
            right_pixel, right_hash, right_thumb = fingerprints[right]
            if left_pixel == right_pixel:
                continue
            distance = (left_hash ^ right_hash).bit_count()
            if distance <= max_hamming:
                mae = sum(ImageStat.Stat(ImageChops.difference(left_thumb, right_thumb)).mean) / 3
                if mae <= max_mae:
                    groups.link(left, right, "potential_near_duplicate",
                                hamming=distance, thumbnail_rgb_mae=round(mae, 5))
    for row in rows:
        row["group_id"] = "group-" + groups.root(row["image_id"])
    return groups


def annotation_signature(row):
    """Ignore instance order and provenance for exact-pixel annotation conflicts."""
    return (row["decision"], tuple(sorted(
        (tuple(round(v, 3) for v in it["bbox_xyxy"]), it["box_kind"],
         it["die_type"] or "", it["face_token"] or "", it["detection_known"],
         it["type_known"], it["value_known"], bool(it.get("result_ambiguous")))
        for it in row["instances"])))


def assign_splits(rows, seed=42):
    """Reserve mixed E1/E2 groups for E2, keeping their E1 members out of training."""
    pixel_buckets = defaultdict(list)
    for row in rows:
        pixel_buckets[row["pixel_sha256"]].append(row)
    for bucket in pixel_buckets.values():
        if len(bucket) > 1:
            if len({annotation_signature(row) for row in bucket}) > 1:
                for row in bucket:
                    row.update(role="quarantine", reason="conflicting_exact_pixel_annotations")
            else:
                for row in sorted(bucket, key=lambda r: r["image_id"])[1:]:
                    row.update(role="duplicate", reason="exact_pixel_duplicate")
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["group_id"]].append(row)
    for bucket in grouped.values():
        if any(row["original_role"] == "e2" for row in bucket):
            for row in bucket:
                if row["role"] == "e1":
                    row.update(role="quarantine", reason="related_to_reserved_e2")
    candidates = {group: [r for r in bucket if r["role"] == "e1"]
                  for group, bucket in grouped.items()}
    candidates = {group: bucket for group, bucket in candidates.items() if bucket}
    if len(candidates) < 2:
        raise ValueError("Fewer than two independent E1 groups; cannot prepare train/val")
    features = {}
    total = Counter()
    for key, bucket in candidates.items():
        counts = Counter({"images": len(bucket)})
        for row in bucket:
            it = row["instances"][0]
            if it["type_known"]:
                counts["type:" + it["die_type"]] += 1
            if it["value_known"]:
                counts["token:" + it["face_token"]] += 1
            if it["type_known"] and it["value_known"]:
                counts[f"pair:{it['die_type']}:{it['face_token']}"] += 1
        features[key] = counts
        total.update(counts)
    rng = random.Random(seed)
    best = None
    # Deterministic grouped candidate search uses annotation balance only, no model results.
    for _ in range(512):
        order = sorted(candidates)
        rng.shuffle(order)
        counts = Counter()
        selected = []
        for key in order[:-1]:
            before = abs(counts["images"] - total["images"] * .2)
            after = abs(counts["images"] + features[key]["images"] - total["images"] * .2)
            if after < before or not selected:
                selected.append(key)
                counts.update(features[key])
        score = sum((5 if name == "images" else 1) * ((counts[name] / number) - .2) ** 2
                    for name, number in total.items())
        item = score, tuple(sorted(selected))
        if best is None or item < best:
            best = item
    validation = set(best[1])
    for row in rows:
        role = row["role"]
        row["split"] = ("val" if row["group_id"] in validation else "train") if role == "e1" else role
    return {"seed": seed, "target_validation_fraction": .2, "search_candidates": 512,
            "objective": best[0], "e1_groups": len(candidates), "validation_groups": len(validation)}
