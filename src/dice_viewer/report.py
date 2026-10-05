"""Record the four business targets without treating missing evidence as success."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_optional(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    evaluation = json.loads((args.artifacts / "evaluation.json").read_text(encoding="utf-8"))
    benchmark = read_optional(args.artifacts / "benchmark.json")
    study = read_optional(args.artifacts / "study-comparison.json")
    end_to_end = evaluation["end_to_end"]["single_die_full_frame"]
    acceptance = evaluation["automatic_acceptance"]
    accuracy = end_to_end["joint_accuracy"]
    false_accept = acceptance["false_accept"]
    upper = acceptance["false_accept_wilson_95"][1] if acceptance["false_accept_wilson_95"] else None
    latency = benchmark["inference_p95_ms"] if benchmark else None
    reduction = study["time_reduction"] if study else None
    report = {
        "joint_image_accuracy": {"observed": accuracy, "target": 0.95, "met": accuracy is not None and accuracy >= 0.95,
                                 "correct": end_to_end["correct"], "total": end_to_end["total"], "wilson_95": end_to_end["wilson_95"]},
        "false_accept": {"observed": false_accept, "target_strictly_below": 0.05,
                         "accepted": acceptance["accepted"], "errors": acceptance["errors"],
                         "wilson_upper_95": upper, "met_with_evidence": upper is not None and upper < 0.05},
        "inference_p95_ms": {"observed": latency, "target_at_most": 1000,
                             "met": latency is not None and latency <= 1000},
        "median_time_reduction": {"observed": reduction, "target_at_least": 0.5,
                                  "met": reduction is not None and reduction >= 0.5},
        "external_dataset_requirement_met": False,
        "evaluation_scope": "Reserved cameras and dates within dieCamera only",
    }
    target = args.artifacts / "target-report.json"
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
