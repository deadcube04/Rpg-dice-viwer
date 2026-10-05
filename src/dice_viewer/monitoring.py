"""Summarize confirmed feedback against the trusted training distribution."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=Path("runtime/feedback.db"))
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/feedback-report.json"))
    args = parser.parse_args()
    if not args.database.is_file():
        raise FileNotFoundError(args.database)
    training = [json.loads(line) for line in (args.processed / "reader_manifest.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    baseline = Counter(row["type"] for row in training if row["split"] == "train")
    baseline_classes = Counter(f'{row["type"]}={row["value"]}' for row in training if row["split"] == "train")
    with sqlite3.connect(args.database) as connection:
        rows = connection.execute("""SELECT f.sides, f.value, f.confirmed, p.model_version, p.status
            FROM feedback f JOIN predictions p ON p.id=f.prediction_id""").fetchall()
    observed = Counter(f"d{sides}" for sides, _, _, _, _ in rows)
    observed_classes = Counter(f"d{sides}={value}" for sides, value, _, _, _ in rows)
    corrections = Counter(f"d{sides}" for sides, _, confirmed, _, _ in rows if not confirmed)
    total_baseline, total_observed = sum(baseline.values()), sum(observed.values())
    by_type = {}
    for die_type in sorted(set(baseline) | set(observed)):
        reference = baseline[die_type] / total_baseline if total_baseline else 0.0
        actual = observed[die_type] / total_observed if total_observed else 0.0
        by_type[die_type] = {"training_count": baseline[die_type], "feedback_count": observed[die_type],
                             "correction_events": corrections[die_type], "training_share": reference,
                             "feedback_share": actual, "share_difference": actual - reference}
    by_class = {}
    for die_class in sorted(set(baseline_classes) | set(observed_classes)):
        by_class[die_class] = {"training_count": baseline_classes[die_class], "feedback_count": observed_classes[die_class],
                               "training_share": baseline_classes[die_class] / total_baseline if total_baseline else 0.0,
                               "feedback_share": observed_classes[die_class] / total_observed if total_observed else 0.0}
    report = {"feedback_events": total_observed, "by_type": by_type, "by_class": by_class,
              "interpretation": "Voluntary feedback is selected by users and does not estimate the real error rate."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
