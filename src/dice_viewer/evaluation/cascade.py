"""Offline detector → body crops → shared multihead → fixed acceptance policy."""

import json
import math
import time
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import load_file

from ..data.prepare import sha256
from ..models.classifier import MultiheadResNet18
from ..models.detector import build_detector
from ..models.labels import LabelMaps
from ..models.preprocessing import crop_body
from ..training.timing import synchronize


class FrozenCascade:
    def __init__(self, root: Path, protocol, device):
        self.device, self.protocol = device, protocol
        dataset = root / protocol["dataset_root"]
        candidate = root / "artifacts" / protocol["candidate_id"]
        manifest = json.loads((candidate / "candidate.json").read_text(encoding="utf-8"))
        labels_path = dataset / "manifests/label_maps.json"
        if sha256(labels_path) != manifest["label_maps_sha256"] or sha256(root / "config/preprocessing.yaml") != manifest["preprocessing_sha256"]:
            raise ValueError("Candidate map/preprocessing differs from the frozen training inputs")
        self.labels = LabelMaps.load(labels_path)
        self.legal = json.loads(labels_path.read_text(encoding="utf-8"))["legal_tokens_by_type"]
        pretrained = json.loads((root / "reports/stage3-20261117/pretrained-weights.json").read_text(encoding="utf-8"))["detector"]
        self.detector = build_detector(root / "config/yolo26n-die.yaml", root / pretrained["path"], pretrained["sha256"])
        self.classifier = MultiheadResNet18(self.labels)
        for component, model in (("detector", self.detector.model), ("classifier", self.classifier)):
            path = candidate / manifest["checkpoints"][component]
            if sha256(path) != manifest["results"][component]["best_checkpoint_sha256"]:
                raise ValueError(f"Trained candidate hash mismatch: {component}")
            model.load_state_dict(load_file(str(path), device="cpu"), strict=True)
        self.detector.model.model[-1].end2end = True
        self.classifier.float().eval().to(device)

    def infer(self, path: Path):
        synchronize(self.device)
        start = time.perf_counter()
        with Image.open(path) as source:
            image = source.convert("RGB")
            image.load()
        decoded = time.perf_counter()
        cfg = self.protocol["detector"]
        result = self.detector.predict(source=image, imgsz=cfg["imgsz"], device=self.device,
                                       nms=cfg["nms"], quantize=cfg["quantize"], conf=cfg["candidate_threshold"],
                                       max_det=cfg["technical_max_candidates"], verbose=False, save=False)[0]
        synchronize(self.device)
        detected = time.perf_counter()
        candidates = []
        for index, row in enumerate(result.boxes.data.detach().cpu().tolist()):
            x1, y1, x2, y2, confidence, label = row
            if not all(math.isfinite(number) for number in row):
                raise RuntimeError("Non-finite detector output")
            if int(label) != 0:
                raise ValueError("Single-class detector returned another class")
            candidates.append({"candidate_index": index, "bbox_xyxy": [x1, y1, x2, y2],
                               "valid_geometry": 0 <= x1 < x2 <= image.width and 0 <= y1 < y2 <= image.height,
                               "model_index": index, "confidence": {"detection": confidence}})
        candidates.sort(key=lambda candidate: (-candidate["confidence"]["detection"],
                                               *candidate["bbox_xyxy"], candidate["model_index"]))
        for index, candidate in enumerate(candidates):
            candidate["candidate_index"] = index
        policy = self.protocol["policy"]
        if any(not p["valid_geometry"] for p in candidates):
            status, items = "invalid_candidate_geometry", []
        elif len(candidates) > policy["max_scene_candidates"]:
            status, items = "too_many_candidates", []
        elif not candidates:
            status, items = "no_candidates", []
        else:
            items = []
            batch_size = self.protocol["classifier"]["crop_batch_size"]
            with torch.inference_mode():
                for offset in range(0, len(candidates), batch_size):
                    chunk = candidates[offset:offset + batch_size]
                    crops = torch.stack([crop_body(image, candidate["bbox_xyxy"])[0] for candidate in chunk]).to(self.device)
                    outputs = self.classifier(crops)
                    if not all(bool(torch.isfinite(tensor).all()) for tensor in outputs.values()):
                        raise RuntimeError("Non-finite classifier output")
                    type_scores, type_ids = outputs["type_logits"].softmax(1).max(1)
                    value_scores, token_ids = outputs["value_logits"].softmax(1).max(1)
                    for candidate, type_id, token_id, type_score, value_score in zip(
                        chunk, type_ids.tolist(), token_ids.tolist(), type_scores.tolist(), value_scores.tolist()
                    ):
                        kind, token = self.labels.types[type_id], self.labels.tokens[token_id]
                        confidence = {**candidate["confidence"], "type": type_score, "value": value_score}
                        reasons = []
                        if confidence["detection"] < policy["detection_threshold"]:
                            reasons.append("low_detection_confidence")
                        if type_score < policy["type_threshold"]:
                            reasons.append("low_type_confidence")
                        if value_score < policy["value_threshold"]:
                            reasons.append("low_value_confidence")
                        if token not in self.legal[kind]:
                            reasons.append("inconsistent_type_value")
                        items.append({**candidate, "die_type": kind, "shape": "d10" if kind == "d10_tens" else kind,
                                      "face_token": token, "face_value": 10 if token in ("0", "00") else int(token),
                                      "confidence": confidence, "auto_accepted": not reasons,
                                      "requires_review": bool(reasons), "review_reasons": reasons})
            status = "needs_review" if any(item["requires_review"] for item in items) else "ok"
        synchronize(self.device)
        finished = time.perf_counter()
        return {"status": status, "candidates": candidates, "items": items,
                "manual_intervention_required": status != "ok",
                "technical_candidate_limit_reached": len(candidates) >= cfg["technical_max_candidates"],
                "timings_seconds": {"decode": decoded - start, "detector": detected - decoded,
                                    "crop_classifier_policy": finished - detected, "total": finished - start}}
