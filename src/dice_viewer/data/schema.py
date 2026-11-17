"""Canonical V1 labels and independent supervision masks."""

import math

TYPES = ("d4", "d6", "d8", "d10", "d10_tens", "d12", "d20")
TOKENS = ("0", "00", *(str(n) for n in range(1, 21)),
          *(str(n) for n in range(30, 100, 10)))
LEGAL_TOKENS = {
    kind: tuple(str(n) for n in range(1, sides + 1))
    for kind, sides in (("d4", 4), ("d6", 6), ("d8", 8),
                        ("d10", 10), ("d12", 12), ("d20", 20))
}
LEGAL_TOKENS["d10"] += ("0",)
LEGAL_TOKENS["d10_tens"] = ("00", *(str(n) for n in range(10, 100, 10)))


def canonical_instance(instance: dict, width: int, height: int) -> dict:
    """Preserve reviewed labels, but never turn unusable labels into targets."""
    result = dict(instance)
    result["reviewed_labels"] = {
        key: instance.get(key)
        for key in ("die_type", "face_token", "face_value", "type_known", "value_known")
    }
    reasons = []
    box = instance.get("bbox_xyxy")
    valid_box = (isinstance(box, list) and len(box) == 4
                 and all(isinstance(x, (float, int)) and math.isfinite(x) for x in box)
                 and 0 <= box[0] < box[2] <= width
                 and 0 <= box[1] < box[3] <= height)
    body = valid_box and instance.get("box_kind") == "body"
    detection = body and instance.get("detection_known") is True
    kind = instance.get("die_type")
    token = instance.get("face_token")
    type_mask = instance.get("type_known") is True and kind in TYPES
    value_mask = (instance.get("value_known") is True
                  and isinstance(token, str) and token in TOKENS
                  and not instance.get("result_ambiguous", False))
    if not valid_box:
        reasons.append("invalid_bbox")
    if not body:
        reasons.append("not_body_bbox")
    if instance.get("type_known") and kind not in TYPES:
        reasons.append("unknown_type_vocabulary")
    if instance.get("value_known") and token not in TOKENS:
        reasons.append("unknown_token_vocabulary")
    if type_mask and value_mask and token not in LEGAL_TOKENS[kind]:
        value_mask = False
        reasons.append("incompatible_type_token")
    if value_mask:
        value = 10 if token in ("0", "00") else int(token)
        if instance.get("face_value") != value:
            reasons.append("normalized_value_recomputed_from_token")
    else:
        value = None
    result.update(
        die_type=kind if type_mask else None,
        face_token=token if value_mask else None,
        face_value=value,
        type_known=type_mask,
        value_known=value_mask,
        detection_known=detection,
        type_id=TYPES.index(kind) if type_mask else None,
        token_id=TOKENS.index(token) if value_mask else None,
        masks={"detection": detection, "type": type_mask, "value": value_mask},
        preparation_issues=reasons,
        crop_eligible=body and (type_mask or value_mask),
    )
    return result


def image_role(row: dict, excluded_sources: set[str]) -> tuple[str, str]:
    if row["source"] in excluded_sources:
        return "excluded_source", "source_excluded_by_user"
    decision = row.get("decision")
    if decision == "excluded":
        return "excluded", row.get("reason") or "excluded_in_review"
    if decision == "negative":
        if row["instances"]:
            raise ValueError(f"Negative image has instances: {row['image_id']}")
        return "negative", "confirmed_negative_reserved_separately"
    if decision != "reviewed":
        raise ValueError(f"Unfinished active review: {row['image_id']}")
    count = len(row["instances"])
    if count == 0:
        return "quarantine", "empty_without_negative_decision"
    if count > 8:
        return "out_of_scope", "above_eight_instances"
    if not all(it["detection_known"] for it in row["instances"]):
        return "quarantine", "incomplete_body_detection_supervision"
    return ("e1", "single_die") if count == 1 else ("e2", "two_to_eight_dice")
