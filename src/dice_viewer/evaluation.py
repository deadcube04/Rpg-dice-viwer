"""Freeze validation choice, then evaluate the held-out cameras once."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageOps

from dice_viewer.dataset import load_rows, numeric_value, pixel_box, split_for, valid_value
from dice_viewer.inference import InferenceEngine
from dice_viewer.reader import ARCHITECTURES, CropDataset, classify


def iou(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_left = (left[2] - left[0]) * (left[3] - left[1])
    area_right = (right[2] - right[0]) * (right[3] - right[1])
    return intersection / (area_left + area_right - intersection) if area_left + area_right > intersection else 0.0


def wilson(correct: int, total: int) -> list[float] | None:
    if total == 0:
        return None
    z = 1.96
    p = correct / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def oracle_reader(engine: InferenceEngine, processed: Path, split: str) -> dict[str, Any]:
    dataset = CropDataset(processed, split)
    correct = 0
    by_type: Counter[str] = Counter()
    correct_by_type: Counter[str] = Counter()
    for row in dataset.rows:
        with Image.open(processed / row["path"]) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        sides, value, _, _ = classify(engine.reader, image, engine.device)
        truth = (int(row["type"][1:]), int(row["value"]))
        matched = (sides, value) == truth
        correct += int(matched)
        by_type[row["type"]] += 1
        correct_by_type[row["type"]] += int(matched)
    return {"correct": correct, "total": len(dataset), "joint_accuracy": correct / len(dataset),
            "by_type": {name: {"correct": correct_by_type[name], "total": count} for name, count in sorted(by_type.items())}}


def evaluate_split(engine: InferenceEngine, raw: Path, split: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    timings: list[float] = []
    frame_categories: Counter[str] = Counter()
    detection_counts: Counter[str] = Counter()
    single_frame_correct = 0
    single_frame_total = 0
    for row in load_rows(raw):
        if split_for(row) != split:
            continue
        with Image.open(raw / row["file_name"]) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        start = time.perf_counter()
        predictions = engine.predict(image)
        if engine.device.type == "cuda":
            torch.cuda.synchronize()
        timings.append((time.perf_counter() - start) * 1000)
        if not row["dice"]:
            category = "no_annotated_die"
        elif len(row["dice"]) > 1:
            category = "multiple_annotated_dice"
        elif not valid_value(row["dice"][0]):
            category = "outside_reading_scope"
        elif not row.get("values_trusted"):
            category = "untrusted_reading"
        else:
            category = "single_trusted_die"
        frame_categories[category] += 1
        detection_counts[f"{category}:{len(predictions)}"] += 1
        if not row.get("values_trusted"):
            continue
        truths = [die for die in row["dice"] if valid_value(die)]
        used: set[int] = set()
        frame_results: list[bool] = []
        for die in truths:
            target = pixel_box(die["box"], *image.size)
            scores = [(iou(target, item.box), index) for index, item in enumerate(predictions) if index not in used]
            overlap, chosen = max(scores, default=(0.0, -1))
            prediction = predictions[chosen] if overlap >= 0.5 else None
            if prediction is not None:
                used.add(chosen)
            expected = (int(die["type"][1:]), numeric_value(die))
            correct = prediction is not None and (prediction.sides, prediction.value) == expected
            frame_results.append(correct)
            records.append({"frame_id": row["id"], "source_file": row["file_name"], "expected_sides": expected[0],
                            "expected_value": expected[1], "predicted_sides": prediction.sides if prediction else None,
                            "predicted_value": prediction.value if prediction else None,
                            "confidence": prediction.confidence if prediction else 0.0,
                            "detected": prediction is not None, "correct": correct, "iou": overlap,
                            "single_die_frame": len(row["dice"]) == 1 and len(truths) == 1,
                            "detection_count": len(predictions)})
        if len(row["dice"]) == 1 and len(truths) == 1:
            single_frame_total += 1
            single_frame_correct += int(len(predictions) == 1 and frame_results[0])
    correct = sum(bool(item["correct"]) for item in records)
    detected = sum(bool(item["detected"]) for item in records)
    total = len(records)
    by_type = {f"d{sides}": {"correct": sum(bool(r["correct"]) for r in records if r["expected_sides"] == sides),
                           "total": sum(r["expected_sides"] == sides for r in records)} for sides in (6, 8, 10, 12, 20)}
    type_confusion: dict[str, dict[str, int]] = {}
    value_confusion: dict[str, dict[str, dict[str, int]]] = {}
    class_counts: Counter[str] = Counter()
    for record in records:
        expected_type = f'd{record["expected_sides"]}'
        predicted_type = f'd{record["predicted_sides"]}' if record["predicted_sides"] else "missing"
        expected_value = str(record["expected_value"])
        predicted_value = str(record["predicted_value"]) if record["predicted_value"] else "missing"
        type_row = type_confusion.setdefault(expected_type, {})
        type_row[predicted_type] = type_row.get(predicted_type, 0) + 1
        value_row = value_confusion.setdefault(expected_type, {}).setdefault(expected_value, {})
        value_row[predicted_value] = value_row.get(predicted_value, 0) + 1
        class_counts[f"{expected_type}={expected_value}"] += 1
    return ({"correct": correct, "total": total, "joint_accuracy": correct / total if total else None,
             "joint_wilson_95": wilson(correct, total), "detector_recall_iou50": detected / total if total else None,
             "single_die_full_frame": {"correct": single_frame_correct, "total": single_frame_total,
                                       "joint_accuracy": single_frame_correct / single_frame_total if single_frame_total else None,
                                       "wilson_95": wilson(single_frame_correct, single_frame_total)},
             "inference_p50_ms": percentile(timings, 0.5), "inference_p95_ms": percentile(timings, 0.95),
             "by_type": by_type, "type_confusion": type_confusion, "value_confusion_by_type": value_confusion,
             "examples_by_class": dict(sorted(class_counts.items())), "frame_categories": dict(frame_categories),
             "detection_counts": dict(detection_counts)}, records)


def calibrate(records: list[dict[str, Any]]) -> tuple[float, dict[str, Any]]:
    eligible = [row for row in records if row["single_die_frame"] and row["detection_count"] == 1 and row["detected"]]
    total_images = sum(row["single_die_frame"] for row in records)
    ordered = sorted(eligible, key=lambda row: row["confidence"], reverse=True)
    candidates = sorted({float(row["confidence"]) for row in ordered}, reverse=True)
    best_threshold = 1.01
    best_accepted = 0
    best_errors = 0
    for threshold in candidates:
        accepted = [row for row in ordered if row["confidence"] >= threshold and row["detected"]]
        if len(accepted) < 20:
            continue
        errors = sum(not row["correct"] for row in accepted)
        upper = wilson(errors, len(accepted))[1]
        if upper < 0.05 and len(accepted) > best_accepted:
            best_threshold, best_accepted, best_errors = threshold, len(accepted), errors
    return best_threshold, {"accepted": best_accepted, "errors": best_errors,
                            "coverage": best_accepted / total_images if total_images else 0.0,
                            "observed_false_accept": best_errors / best_accepted if best_accepted else None,
                            "false_accept_wilson_95": wilson(best_errors, best_accepted),
                            "calibration_supported": best_accepted > 0}


def apply_threshold(records: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    single = [row for row in records if row["single_die_frame"]]
    accepted = [row for row in single if row["detected"] and row["detection_count"] == 1 and row["confidence"] >= threshold]
    errors = sum(not row["correct"] for row in accepted)
    return {"accepted": len(accepted), "errors": errors, "coverage": len(accepted) / len(single) if single else 0.0,
            "false_accept": errors / len(accepted) if accepted else None,
            "false_accept_wilson_95": wilson(errors, len(accepted))}


def select(raw: Path, processed: Path, artifacts: Path) -> None:
    selection = artifacts / "selection.json"
    if selection.exists():
        raise FileExistsError(f"Selection already frozen: {selection}")
    candidates: list[dict[str, Any]] = []
    for architecture in ARCHITECTURES:
        reader_folder = artifacts / "readers" / architecture
        engine = InferenceEngine(artifacts / "detector" / "best.pt", reader_folder)
        summary, records = evaluate_split(engine, raw, "valid")
        threshold, calibration = calibrate(records)
        candidates.append({"architecture": architecture, "reader_folder": str(reader_folder), "validation": summary,
                           "oracle_reader": oracle_reader(engine, processed, "valid"),
                           "threshold": threshold, "calibration": calibration})
    candidates.sort(key=lambda item: (item["validation"]["single_die_full_frame"]["joint_accuracy"] or 0,
                                      -(item["calibration"]["observed_false_accept"] if item["calibration"]["observed_false_accept"] is not None else 1),
                                      item["calibration"]["coverage"],
                                      item["validation"]["joint_accuracy"] or 0,
                                      -(item["validation"]["inference_p95_ms"] or float("inf"))), reverse=True)
    result = {"selected": candidates[0]["architecture"], "threshold": candidates[0]["threshold"],
              "candidates": candidates, "detector_path": str(artifacts / "detector" / "best.pt"),
              "selection_order": ["single_die_image_joint_accuracy", "observed_false_accept", "auto_accept_coverage",
                                  "per_die_joint_accuracy", "inference_p95_ms"]}
    selection.parent.mkdir(parents=True, exist_ok=True)
    selection.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


def final_test(raw: Path, processed: Path, artifacts: Path) -> None:
    target = artifacts / "evaluation.json"
    if target.exists():
        raise FileExistsError(f"Held-out evaluation already exists: {target}")
    selection = json.loads((artifacts / "selection.json").read_text(encoding="utf-8"))
    engine = InferenceEngine(Path(selection["detector_path"]), artifacts / "readers" / selection["selected"])
    summary, records = evaluate_split(engine, raw, "test")
    report = {"end_to_end": summary, "reader_with_true_crops": oracle_reader(engine, processed, "test"),
              "automatic_acceptance": apply_threshold(records, float(selection["threshold"])),
              "threshold": selection["threshold"], "selected": selection["selected"], "rows": records}
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("select", "test"))
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    select(args.raw, args.processed, args.artifacts) if args.phase == "select" else final_test(args.raw, args.processed, args.artifacts)


if __name__ == "__main__":
    main()
