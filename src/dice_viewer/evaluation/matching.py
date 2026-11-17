"""Exact spatial assignment for scenes with at most eight reference dice."""


def box_iou(left, right):
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0., x2 - x1) * max(0., y2 - y1)
    left_area = max(0., left[2] - left[0]) * max(0., left[3] - left[1])
    right_area = max(0., right[2] - right[0]) * max(0., right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.


def match_boxes(predictions, references, threshold=.5):
    """DP over reference masks: maximize pair count, then summed IoU, then tie order.

    Each prediction is processed exactly once. For the same reference mask and
    processed prefix, only the highest-IoU assignment can improve a future state.
    Pairing never reads type, token, confidence or acceptance labels.
    """
    if len(references) > 8:
        raise ValueError("Operational E2 matching supports at most eight references")
    states = {0: (0., ())}
    matrix = [[box_iou(p["bbox_xyxy"], g["bbox_xyxy"]) if p.get("valid_geometry", True) else 0.
               for g in references] for p in predictions]
    for prediction_index, overlaps in enumerate(matrix):
        following = dict(states)
        eligible = [(index, value) for index, value in enumerate(overlaps) if value >= threshold]
        for mask, (total, pairs) in states.items():
            for reference_index, overlap in eligible:
                bit = 1 << reference_index
                if mask & bit:
                    continue
                next_mask = mask | bit
                candidate = total + overlap, pairs + ((prediction_index, reference_index),)
                previous = following.get(next_mask)
                if previous is None or candidate[0] > previous[0] or (candidate[0] == previous[0] and candidate[1] < previous[1]):
                    following[next_mask] = candidate
        states = following
    mask, (score, pairs) = min(states.items(), key=lambda item: (-item[0].bit_count(), -item[1][0], item[1][1]))
    return [{"prediction": p, "reference": g, "iou": matrix[p][g]} for p, g in pairs]
