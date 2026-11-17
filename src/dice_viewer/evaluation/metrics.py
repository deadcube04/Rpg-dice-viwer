"""Operational E2 metrics with spatial matching and explicit evaluable supports."""

from collections import Counter, defaultdict

from .matching import match_boxes


def rate(numerator, denominator):
    return numerator / denominator if denominator else None


def joint_known(reference):
    return bool(reference["type_known"] and reference["value_known"])


def joint_correct(prediction, reference):
    return joint_known(reference) and prediction["die_type"] == reference["die_type"] and prediction["face_token"] == reference["face_token"]


def score_scene(row, result, threshold=.5):
    references, candidates, items = row["instances"], result["candidates"], result["items"]
    spatial = match_boxes(candidates, references, threshold)
    paired = match_boxes(items, references, threshold)
    high = [p for p in candidates if p["confidence"]["detection"] >= .95]
    high_pairs = match_boxes(high, references, threshold)
    accepted = [p for p in items if p["auto_accepted"]]
    accepted_pairs = match_boxes(accepted, references, threshold)
    counts = Counter(images=1, references=len(references), predictions=len(candidates),
                     spatial_tp=len(spatial), spatial_fp=len(candidates) - len(spatial), spatial_fn=len(references) - len(spatial),
                     high_predictions=len(high), high_tp=len(high_pairs), high_fp=len(high) - len(high_pairs),
                     high_fn=len(references) - len(high_pairs), returned_items=len(items),
                     joint_references=sum(joint_known(g) for g in references),
                     accepted=len(accepted), review_items=len(items) - len(accepted),
                     manual_images=int(result["manual_intervention_required"]),
                     count_exact_images=int(len(candidates) == len(references)),
                     absolute_count_error=abs(len(candidates) - len(references)),
                     exact_eligible_images=int(all(joint_known(g) for g in references)),
                     technical_limit_images=int(result["technical_candidate_limit_reached"]))
    counts["status_" + result["status"]] += 1
    counts["incompatible_items"] = sum("inconsistent_type_value" in p["review_reasons"] for p in items)
    counts["recognition_evaluable_predictions"] = len(items)
    confusion = []
    returned_by_reference = {pair["reference"]: items[pair["prediction"]] for pair in paired}
    localized_refs = {pair["reference"] for pair in spatial}
    accepted_correct_refs = set()
    for pair in accepted_pairs:
        p, g = accepted[pair["prediction"]], references[pair["reference"]]
        if not joint_known(g):
            counts["accepted_not_evaluable"] += 1
        elif joint_correct(p, g):
            counts["accepted_correct"] += 1
            accepted_correct_refs.add(pair["reference"])
    counts["accepted_evaluable"] = len(accepted) - counts["accepted_not_evaluable"]
    for pair in paired:
        p, g = items[pair["prediction"]], references[pair["reference"]]
        if g["type_known"]:
            counts["paired_type_support"] += 1
            counts["paired_type_correct"] += int(p["die_type"] == g["die_type"])
            confusion.append(("type", g["die_type"], p["die_type"]))
        if g["value_known"]:
            counts["paired_value_support"] += 1
            counts["paired_token_correct"] += int(p["face_token"] == g["face_token"])
            counts["paired_value_correct"] += int(p["face_value"] == g["face_value"])
            confusion.append(("token", g["face_token"], p["face_token"]))
        if joint_known(g):
            counts["paired_joint_support"] += 1
            counts["joint_correct"] += int(joint_correct(p, g))
            counts["joint_normalized_correct"] += int(p["die_type"] == g["die_type"] and p["face_value"] == g["face_value"])
        else:
            counts["recognition_evaluable_predictions"] -= 1
    counts["exact_correct_images"] = int(counts["exact_eligible_images"] and len(items) == len(references)
                                         and counts["joint_correct"] == len(references))
    objects = []
    for index, g in enumerate(references):
        p = returned_by_reference.get(index)
        objects.append({"image_id": row["image_id"], "instance_id": g["instance_id"], "source": row["source"],
                        "dice_count": len(references), "die_type": g["die_type"] or "unknown",
                        "face_token": g["face_token"], "modality": g.get("modality", "unknown"),
                        "occluded": g.get("occluded", False), "truncated": g.get("truncated", False),
                        "references": 1, "localized": int(index in localized_refs),
                        "type_references": int(g["type_known"]), "value_references": int(g["value_known"]),
                        "joint_references": int(joint_known(g)),
                        "paired_type_support": int(p is not None and g["type_known"]),
                        "paired_type_correct": int(p is not None and g["type_known"] and p["die_type"] == g["die_type"]),
                        "paired_value_support": int(p is not None and g["value_known"]),
                        "paired_token_correct": int(p is not None and g["value_known"] and p["face_token"] == g["face_token"]),
                        "joint_correct": int(p is not None and joint_correct(p, g)),
                        "accepted_correct": int(index in accepted_correct_refs)})
    return {"image_id": row["image_id"], "source": row["source"], "dice_count": len(references),
            "counts": dict(counts), "spatial_pairs": spatial, "returned_pairs": paired, "accepted_pairs": accepted_pairs,
            "count_error": len(candidates) - len(references), "objects": objects, "confusion": confusion,
            "timings_seconds": result["timings_seconds"], "status": result["status"]}


def summarize_counts(counts):
    c = Counter(counts)
    precision, recall = rate(c["spatial_tp"], c["predictions"]), rate(c["spatial_tp"], c["references"])
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else (0. if c["references"] else None)
    return {"counts": dict(c), "detector_precision_at_025": precision, "detector_recall_at_025": recall,
            "detector_f1_at_025": f1, "detector_precision_at_095": rate(c["high_tp"], c["high_predictions"]),
            "detector_recall_at_095": rate(c["high_tp"], c["references"]),
            "type_accuracy_on_matched_items": rate(c["paired_type_correct"], c["paired_type_support"]),
            "token_accuracy_on_matched_items": rate(c["paired_token_correct"], c["paired_value_support"]),
            "normalized_value_accuracy_on_matched_items": rate(c["paired_value_correct"], c["paired_value_support"]),
            "joint_accuracy_on_matched_items": rate(c["joint_correct"], c["paired_joint_support"]),
            "complete_recognition_precision": rate(c["joint_correct"], c["recognition_evaluable_predictions"]),
            "complete_recognition_recall": rate(c["joint_correct"], c["joint_references"]),
            "scene_exact_match": rate(c["exact_correct_images"], c["exact_eligible_images"]),
            "scene_count_accuracy": rate(c["count_exact_images"], c["images"]),
            "mean_absolute_count_error": rate(c["absolute_count_error"], c["images"]),
            "candidate_acceptance_rate": rate(c["accepted"], c["returned_items"]),
            "candidate_review_rate": rate(c["review_items"], c["returned_items"]),
            "accepted_precision": rate(c["accepted_correct"], c["accepted_evaluable"]),
            "correct_automatic_coverage": rate(c["accepted_correct"], c["joint_references"]),
            "manual_intervention_image_rate": rate(c["manual_images"], c["images"]),
            "incompatible_item_rate": rate(c["incompatible_items"], c["returned_items"])}


def aggregate(scored, labels):
    overall = Counter()
    scene_groups = {"count": defaultdict(Counter), "source": defaultdict(Counter)}
    object_groups = {"type": defaultdict(Counter), "modality": defaultdict(Counter), "type_modality": defaultdict(Counter)}
    confusion = {"type": Counter(), "token": Counter()}
    count_errors = Counter()
    for scene in scored:
        overall.update(scene["counts"])
        scene_groups["count"][str(scene["dice_count"])].update(scene["counts"])
        scene_groups["source"][scene["source"]].update(scene["counts"])
        count_errors[scene["count_error"]] += 1
        for task, truth, predicted in scene["confusion"]:
            confusion[task][(truth, predicted)] += 1
        for obj in scene["objects"]:
            numeric = {key: value for key, value in obj.items() if isinstance(value, int) and not isinstance(value, bool) and key != "dice_count"}
            for dimension, key in (("type", obj["die_type"]), ("modality", obj["modality"]),
                                   ("type_modality", obj["die_type"] + ":" + obj["modality"])):
                object_groups[dimension][key].update(numeric)
    macro = {}
    matrices = {}
    for task, vocabulary in (("type", labels.types), ("token", labels.tokens)):
        matrix = [[confusion[task][(truth, pred)] for pred in vocabulary] for truth in vocabulary]
        classes = []
        supported = []
        for i, label in enumerate(vocabulary):
            support = sum(matrix[i])
            tp = matrix[i][i]
            fp = sum(row[i] for row in matrix) - tp
            fn = support - tp
            f1 = rate(2 * tp, 2 * tp + fp + fn) if support else None
            classes.append({"label": label, "support": support, "f1": f1})
            if f1 is not None:
                supported.append(f1)
        macro[task] = {"macro_f1_supported_reference_classes": rate(sum(supported), len(supported)), "classes": classes}
        matrices[task] = {"labels": vocabulary, "matrix": matrix}
    return {"overall": summarize_counts(overall),
            "by_count": {key: summarize_counts(value) for key, value in sorted(scene_groups["count"].items())},
            "by_source": {key: summarize_counts(value) for key, value in sorted(scene_groups["source"].items())},
            "object_strata": {dimension: {key: dict(counts) for key, counts in sorted(values.items())} for dimension, values in object_groups.items()},
            "conditional_classification": macro, "confusion_matrices": matrices,
            "count_error_distribution": {str(key): value for key, value in sorted(count_errors.items())}}
