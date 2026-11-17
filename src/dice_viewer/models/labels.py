"""Load the approved numeric label order, independently of JSON key ordering."""

import json
from dataclasses import dataclass
from pathlib import Path


def _ordered(mapping: dict[str, str]) -> tuple[str, ...]:
    ids = sorted(int(key) for key in mapping)
    if ids != list(range(len(ids))):
        raise ValueError("Label IDs must be contiguous from zero")
    labels = tuple(mapping[str(index)] for index in ids)
    if not labels or len(labels) != len(set(labels)):
        raise ValueError("Label vocabulary is empty or has duplicates")
    return labels


@dataclass(frozen=True)
class LabelMaps:
    types: tuple[str, ...]
    tokens: tuple[str, ...]

    @classmethod
    def load(cls, path: Path) -> "LabelMaps":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw["detector"] != {"0": "die"}:
            raise ValueError("Detector vocabulary must be exactly {0: die}")
        return cls(_ordered(raw["types"]), _ordered(raw["tokens"]))
