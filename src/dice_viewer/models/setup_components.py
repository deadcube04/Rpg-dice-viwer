"""Provision official initial weights and record Stage 3 decisions; never train."""

import argparse
import ctypes
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import urllib.request
import winreg
from pathlib import Path

import torch
import yaml
from torchvision.models import ResNet18_Weights
from ultralytics.cfg import DEFAULT_CFG_DICT
from ultralytics.nn.tasks import yaml_model_load

from ..data.prepare import sha256, write_json
from .preprocessing import MEAN, STD, VERSION

ROOT = Path(__file__).resolve().parents[3]
DATASET = ROOT / "data/datasets/v1-20261117-7b23477513de"
EXPECTED_INDEX = "5ecbe03114abb68790e2319e6e7b26f745c422735d48895a86d2761b6c0d1a73"
YOLO_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt"
YOLO_SHA256 = "9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef"
RESNET_SHA256 = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"


class MemoryStatus(ctypes.Structure):
    _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                *[(name, ctypes.c_ulonglong) for name in
                  ("total_physical", "available_physical", "total_page_file", "available_page_file",
                   "total_virtual", "available_virtual", "available_extended_virtual")]]


def hardware_inventory():
    memory = MemoryStatus()
    memory.length = ctypes.sizeof(memory)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
        raise OSError("GlobalMemoryStatusEx failed")
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
        cpu = winreg.QueryValueEx(key, "ProcessorNameString")[0]
    inventory = {"platform": platform.platform(), "python": sys.version, "cpu": cpu,
                 "logical_processors": os.cpu_count(), "ram_bytes": memory.total_physical,
                 "available_ram_bytes": memory.available_physical,
                 "cuda_available": torch.cuda.is_available(), "torch_cuda_runtime": torch.version.cuda,
                 "torch_supported_architectures": torch.cuda.get_arch_list()}
    if torch.cuda.is_available():
        prop = torch.cuda.get_device_properties(0)
        inventory["gpu"] = {"name": prop.name, "vram_bytes": prop.total_memory,
                            "compute_capability": [prop.major, prop.minor]}
        inventory["nvidia_smi"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.free", "--format=csv,noheader"],
            capture_output=True, text=True, check=True).stdout.strip()
    return inventory


def download_official(url: str, destination: Path, expected_sha256: str):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        partial = destination.with_suffix(destination.suffix + ".part")
        print(f"Downloading {url}", flush=True)
        request = urllib.request.Request(url, headers={"User-Agent": "dice-viewer-stage3/1.0"})
        with urllib.request.urlopen(request, timeout=120) as source, partial.open("wb") as output:
            while block := source.read(1024 * 1024):
                output.write(block)
        partial.replace(destination)
    digest = sha256(destination)
    if digest != expected_sha256:
        raise ValueError(f"Pinned official weight hash mismatch: {destination}")
    return {"url": url, "path": destination.relative_to(ROOT).as_posix(), "sha256": digest,
            "bytes": destination.stat().st_size, "purpose": "pretrained_initialization_only_not_a_trained_candidate"}


def write_yaml_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        yaml.safe_dump(value, handle, sort_keys=False, allow_unicode=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports/stage3-20261117")
    parser.add_argument("--weights-dir", type=Path, default=ROOT / "artifacts/pretrained-v1")
    args = parser.parse_args()
    report = args.report_dir
    if (report / "setup-status.json").exists() or any((ROOT / "config" / name).exists() for name in
        ("dataset.yaml", "detector.yaml", "classifier.yaml", "preprocessing.yaml", "yolo26n-die.yaml")):
        raise FileExistsError("Stage 3 is already provisioned; preserve its configs and use a new version")
    report.mkdir(parents=True, exist_ok=True)
    if sha256(DATASET / "checksums.jsonl") != EXPECTED_INDEX:
        raise ValueError("Prepared dataset no longer matches the version approved by the user")
    approval = {"approved": True, "approval_date": "2026-11-17", "approved_by": "user",
                "reference": "ok, aprovado — approval of Stage 2 dataset and limitations in this conversation",
                "dataset": str(DATASET), "checksum_index_sha256": EXPECTED_INDEX,
                "authorized_next_stage": 3, "training_authorized": False}
    write_json(ROOT / "data/preparation/dataset-approval.json", approval)
    write_json(report / "dataset-approval.json", approval)
    write_json(report / "hardware.json", hardware_inventory())
    packages = {dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()}
    write_json(report / "installed-packages.json", dict(sorted(packages.items(), key=lambda x: x[0].lower())))
    write_json(report / "documentation.json", {
        "context7": "Attempted library resolution for ultralytics and torchvision; both failed during MCP initialization",
        "fallback": "Official documentation and inspection of the installed source of the pinned packages",
        "references": ["https://docs.ultralytics.com/models/yolo26/",
                       "https://docs.ultralytics.com/reference/nn/tasks/",
                       "https://docs.ultralytics.com/reference/nn/modules/head/",
                       "https://docs.ultralytics.com/modes/predict/",
                       "https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html",
                       "https://pytorch.org/get-started/previous-versions/"],
        "api_decision": "In installed Ultralytics 8.4.173, nms=False selects AutoBackend end2end=True; quantize=32 keeps FP32",
    })
    weights = {
        "detector": download_official(YOLO_URL, args.weights_dir / "yolo26n.pt", YOLO_SHA256),
        "classifier": download_official(ResNet18_Weights.IMAGENET1K_V1.url,
                                        args.weights_dir / "resnet18-f37072fd.pth", RESNET_SHA256),
    }
    write_json(report / "pretrained-weights.json", weights)
    config = ROOT / "config"
    architecture = yaml_model_load("yolo26n.yaml")
    architecture.update(nc=1, scale="n")
    architecture.pop("yaml_file", None)
    write_yaml_new(config / "yolo26n-die.yaml", architecture)
    write_yaml_new(config / "dataset.yaml", {
        "version": DATASET.name, "root": DATASET.relative_to(ROOT).as_posix(),
        "checksum_index_sha256": EXPECTED_INDEX, "approval": "data/preparation/dataset-approval.json",
        "detector_dataset": (DATASET / "yolo/dataset.yaml").relative_to(ROOT).as_posix(),
        "label_maps": (DATASET / "manifests/label_maps.json").relative_to(ROOT).as_posix(),
        "classifier_manifests": {split: (DATASET / f"classifier/{split}.jsonl").relative_to(ROOT).as_posix()
                                 for split in ("train", "val", "e2")},
        "e2_use": "reserved_final_evaluation_only; never training or checkpoint selection",
    })
    write_yaml_new(config / "preprocessing.yaml", {
        "version": VERSION, "orientation": "EXIF transpose before applying canonical boxes",
        "detector": {"imgsz": 640, "channels": "RGB", "tensor_scale": "0..1", "dtype": "float32",
                     "geometry": "Ultralytics pinned-version letterbox; reverse to original oriented pixels"},
        "classifier": {"crop_size": [224, 224], "margin_per_side": .1, "clip_to_image": True,
                       "preserve_aspect_ratio": True, "padding_rgb": [0, 0, 0], "interpolation": "bilinear",
                       "channels": "RGB", "tensor_scale": "0..1", "mean": list(MEAN), "std": list(STD),
                       "prepared_crops": "normalize only; no second crop/resize", "center_crop": False},
    })
    write_yaml_new(config / "detector.yaml", {
        "architecture": "config/yolo26n-die.yaml", "pretrained": weights["detector"],
        "class_names": {0: "die"}, "default_device": "cpu", "precision": "FP32",
        "prediction": {"imgsz": 640, "nms": False, "quantize": 32, "conf": .25, "max_det": 64},
        "training": {"epochs": 100, "patience": 15, "optimizer": "SGD", "lr0": .01, "lrf": .01,
                     "momentum": .937, "weight_decay": .0005, "seed": 42, "deterministic": True,
                     "batch_by_device": {"cuda": 8, "cpu": 2}, "imgsz": 640, "amp": False,
                     "workers": 0, "mosaic": 0.0, "mixup": 0.0, "cutmix": 0.0, "copy_paste": 0.0,
                     "degrees": 0.0, "fliplr": 0.0, "flipud": 0.0, "translate": 0.0, "scale": 0.0,
                     "shear": 0.0, "perspective": 0.0, "hsv_h": 0.0, "hsv_s": .1, "hsv_v": .1,
                     "close_mosaic": 0, "multi_scale": False, "cos_lr": False, "warmup_epochs": 3.0,
                     "cache": False, "rect": False, "fraction": 1.0,
                     "checkpoint_selection": "validation_AP50_95", "on_out_of_memory": "stop; require a new recorded batch configuration"},
        "remaining_library_defaults": "reports/stage3-20261117/ultralytics-defaults.json",
    })
    write_yaml_new(config / "classifier.yaml", {
        "backbone": "resnet18", "pretrained_enum": "ResNet18_Weights.IMAGENET1K_V1",
        "pretrained": weights["classifier"], "heads": "independent linear type and global token heads; no type conditioning",
        "label_maps": (DATASET / "manifests/label_maps.json").relative_to(ROOT).as_posix(),
        "default_device": "cpu", "precision": "FP32", "seed": 42,
        "training": {"epochs_total": 40, "head_only_epochs": 5, "full_finetune_max_epochs": 35,
                     "freeze_backbone_batchnorm_during_head_phase": True,
                     "optimizer": "AdamW", "head_lr": .001, "backbone_lr": .0001,
                     "betas": [.9, .999], "eps": 1e-8, "weight_decay": .0001,
                     "scheduler": "constant", "gradient_accumulation": 1,
                     "optimizer_phase_transition": "preserve head optimizer state; add backbone parameter group at epoch 6",
                     "batch_by_device": {"cuda": 16, "cpu": 4}, "workers": 0,
                     "loss": "independent masked cross_entropy on raw logits",
                     "loss_weights": {"type": 1.0, "value": 1.0},
                     "empty_task": "graph-connected zero; no mean over an empty set",
                     "checkpoint_selection": ["max_joint_type_token_accuracy_on_jointly_labeled_validation_crops",
                                              "min_validation_loss", "earliest_epoch"],
                     "early_stopping_patience_full_finetune": 8,
                     "augmentation": {"brightness_delta": .1, "contrast_delta": .1, "saturation_delta": .1,
                                      "hue_delta": 0.0, "flips": False, "rotation": 0.0,
                                      "training_only": True, "before_normalization": True},
                     "on_out_of_memory": "stop; require a new recorded batch configuration"},
        "scores": "separate softmax per head at inference; uncalibrated",
    })
    write_json(report / "ultralytics-defaults.json", DEFAULT_CFG_DICT)
    write_json(report / "setup-status.json", {"stage": 3, "status": "provisioned_awaiting_component_inspection",
                                              "training_started": False, "weights": weights})
    print("Official weights and versioned component configuration are provisioned.", flush=True)


if __name__ == "__main__":
    main()
