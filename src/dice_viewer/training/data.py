"""Read-only classification crops and an isolated YOLO training view."""

import json
import os
import shutil
from pathlib import Path

import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter

from ..models.preprocessing import tensor_from_prepared_crop


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


class CropDataset(Dataset):
    def __init__(self, dataset: Path, split: str, augmentation=None):
        if split not in ("train", "val"):
            raise ValueError("E1 training only permits train and val crop manifests")
        self.root = dataset
        self.records = read_jsonl(dataset / f"classifier/{split}.jsonl")
        self.augment = None
        if split == "train" and augmentation:
            self.augment = ColorJitter(brightness=augmentation["brightness_delta"],
                                      contrast=augmentation["contrast_delta"], saturation=augmentation["saturation_delta"],
                                      hue=augmentation["hue_delta"])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with Image.open(self.root / record["path"]) as image:
            image = image.convert("RGB")
            if self.augment:
                image = self.augment(image)
            tensor = tensor_from_prepared_crop(image)
        return {"image": tensor, "type_target": record["type_id"] if record["type_mask"] else -100,
                "value_target": record["token_id"] if record["value_mask"] else -100,
                "type_mask": record["type_mask"], "value_mask": record["value_mask"], "index": index}


def create_detector_view(dataset: Path, destination: Path):
    """Keep Ultralytics caches outside the frozen approved dataset; no E2 paths."""
    counts, methods = {}, {"hardlink": 0, "copy": 0}
    for split in ("train", "val"):
        rows = read_jsonl(dataset / f"splits/{split}.jsonl")
        counts[split] = len(rows)
        for kind, suffix in (("images", ".png"), ("labels", ".txt")):
            target_directory = destination / kind / split
            target_directory.mkdir(parents=True)
            for row in rows:
                source = dataset / "yolo" / kind / split / (row["image_id"] + suffix)
                target = target_directory / source.name
                # Images are read-only inputs; label copies also isolate any label-side cache behavior.
                try:
                    if kind == "labels":
                        raise OSError("Label files use independent copies")
                    os.link(source, target)
                    methods["hardlink"] += 1
                except OSError:
                    shutil.copyfile(source, target)
                    methods["copy"] += 1
    path = destination / "dataset.yaml"
    path.write_text(yaml.safe_dump({"path": str(destination.resolve()), "train": "images/train", "val": "images/val",
                                    "names": {0: "die"}}, sort_keys=False), encoding="utf-8")
    return path, {"images": counts, "storage": methods, "e2_included": False,
                  "cache_policy": "Ultralytics caches are written only in this run-local view"}
