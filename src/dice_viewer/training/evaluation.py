"""Validation-only classifier metrics with explicit task supports."""

import csv
import json
from collections import defaultdict
from pathlib import Path

import torch


def evaluate_classifier(model, loader, criterion, device, labels, destination: Path | None = None):
    model.eval()
    counts = defaultdict(int)
    sums = defaultdict(float)
    type_confusion = torch.zeros((len(labels.types), len(labels.types)), dtype=torch.long)
    value_confusion = torch.zeros((len(labels.tokens), len(labels.tokens)), dtype=torch.long)
    strata = defaultdict(lambda: defaultdict(int))
    normalized = torch.tensor([10 if token in ("0", "00") else int(token) for token in labels.tokens], device=device)
    predictions = []
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            tt = batch["type_target"].to(device)
            vt = batch["value_target"].to(device)
            tm = batch["type_mask"].to(device)
            vm = batch["value_mask"].to(device)
            outputs = model(images)
            losses = criterion(outputs, tt, vt, tm, vm)
            if not torch.isfinite(losses["loss"]):
                raise RuntimeError("Non-finite classifier validation loss")
            tp, vp = outputs["type_logits"].argmax(1), outputs["value_logits"].argmax(1)
            joint = tm & vm
            correct_type, correct_token = tp == tt, vp == vt
            correct_normalized = normalized[vp] == normalized[vt.clamp_min(0)]
            counts["type"] += int(tm.sum())
            counts["value"] += int(vm.sum())
            counts["joint"] += int(joint.sum())
            counts["type_correct"] += int((correct_type & tm).sum())
            counts["token_correct"] += int((correct_token & vm).sum())
            counts["normalized_correct"] += int((correct_normalized & vm).sum())
            counts["joint_correct"] += int((correct_type & correct_token & joint).sum())
            counts["joint_normalized_correct"] += int((correct_type & correct_normalized & joint).sum())
            sums["type_loss"] += float(losses["type_loss"]) * losses["type_count"]
            sums["value_loss"] += float(losses["value_loss"]) * losses["value_count"]
            for truth, prediction in zip(tt[tm].cpu(), tp[tm].cpu()):
                type_confusion[truth, prediction] += 1
            for truth, prediction in zip(vt[vm].cpu(), vp[vm].cpu()):
                value_confusion[truth, prediction] += 1
            for position, index in enumerate(batch["index"].tolist()):
                record = loader.dataset.records[index]
                key = (record["die_type"] or "unknown", record.get("modality", "unknown"))
                strata[key]["instances"] += 1
                if record["type_mask"]:
                    strata[key]["type_support"] += 1
                    strata[key]["type_correct"] += int(correct_type[position])
                if record["value_mask"]:
                    strata[key]["value_support"] += 1
                    strata[key]["value_correct"] += int(correct_token[position])
                if record["type_mask"] and record["value_mask"]:
                    strata[key]["joint_support"] += 1
                    strata[key]["joint_correct"] += int(correct_type[position] & correct_token[position])
                if destination is not None:
                    predictions.append({"crop_id": record["crop_id"], "image_id": record["image_id"],
                                        "truth_type": record["die_type"], "truth_token": record["face_token"],
                                        "predicted_type": labels.types[int(tp[position])],
                                        "predicted_token": labels.tokens[int(vp[position])],
                                        "type_known": record["type_mask"], "value_known": record["value_mask"]})
    if not counts["joint"]:
        raise ValueError("Validation has no jointly labeled examples for checkpoint selection")
    metrics = {
        "type_loss": sums["type_loss"] / max(1, counts["type"]),
        "value_loss": sums["value_loss"] / max(1, counts["value"]),
        "type_accuracy": counts["type_correct"] / max(1, counts["type"]),
        "token_accuracy": counts["token_correct"] / max(1, counts["value"]),
        "normalized_value_accuracy": counts["normalized_correct"] / max(1, counts["value"]),
        "joint_accuracy": counts["joint_correct"] / counts["joint"],
        "joint_normalized_accuracy": counts["joint_normalized_correct"] / counts["joint"],
        "type_support": counts["type"], "value_support": counts["value"], "joint_support": counts["joint"],
    }
    metrics["loss"] = criterion.type_weight * metrics["type_loss"] + criterion.value_weight * metrics["value_loss"]
    if destination is not None:
        destination.mkdir(parents=True, exist_ok=True)
        for name, matrix, vocabulary in (("type", type_confusion, labels.types), ("token", value_confusion, labels.tokens)):
            with (destination / f"confusion-{name}.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["truth/prediction", *vocabulary])
                for label, values in zip(vocabulary, matrix.tolist()):
                    writer.writerow([label, *values])
        (destination / "strata.json").write_text(json.dumps([
            {"die_type": key[0], "modality": key[1], **dict(value)} for key, value in sorted(strata.items())], indent=2) + "\n", encoding="utf-8")
        (destination / "predictions.jsonl").write_text("".join(json.dumps(value) + "\n" for value in predictions), encoding="utf-8")
    return metrics
