"""Train one die detector and two conditional readers with MLflow records."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from pathlib import Path

import mlflow
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from ultralytics import YOLO

from dice_viewer.provenance import git_revision, source_digest
from dice_viewer.reader import ARCHITECTURES, CropDataset, DieReader, save_reader


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def common_tags(processed: Path) -> dict[str, str]:
    report = json.loads((processed / "report.json").read_text(encoding="utf-8"))
    return {"source_sha256": report["source_manifest_sha256"], "split_sha256": report["frame_manifest_sha256"],
            "code_sha256": source_digest(), "git_revision": git_revision() or "uncommitted"}


def train_detector(processed: Path, output: Path) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for the planned detector training")
    seed_everything()
    output.mkdir(parents=True, exist_ok=True)
    dataset_yaml = output / "detector-data.yaml"
    dataset_yaml.write_text(f"path: {((processed / 'detector').resolve()).as_posix()}\ntrain: train/images\nval: valid/images\ntest: test/images\nnames:\n  0: die\n", encoding="utf-8")
    with mlflow.start_run(run_name="detector-yolo11n") as run:
        mlflow.set_tags({"stage": "detector", **common_tags(processed)})
        mlflow.log_params({"architecture": "yolo11n", "seed": 42, "epochs": 100, "patience": 15, "image_size": 640, "batch": 8, "amp": False})
        detector = YOLO("yolo11n.pt")
        results = detector.train(data=str(dataset_yaml), epochs=100, imgsz=640, batch=8, patience=15,
                                 device=0, seed=42, workers=0, amp=False, project=str(output.resolve()), name="run", exist_ok=True)
        best = Path(results.save_dir) / "weights" / "best.pt"
        if not best.is_file():
            raise FileNotFoundError(best)
        destination = output / "best.pt"
        shutil.copy2(best, destination)
        metrics = {key.replace("(", "_").replace(")", "").replace("/", "_"): float(value)
                   for key, value in results.results_dict.items() if isinstance(value, (int, float))}
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(Path(results.save_dir) / "results.csv"), artifact_path="detector")
        mlflow.log_artifact(str(destination), artifact_path="detector")
        (output / "run.json").write_text(json.dumps({"mlflow_run_id": run.info.run_id, **common_tags(processed)}, indent=2), encoding="utf-8")
        return destination


def epoch_pass(model: DieReader, loader: DataLoader, device: torch.device,
               optimizer: torch.optim.Optimizer | None) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    total = 0
    for images, types, values in loader:
        images, types, values = images.to(device), types.to(device), values.to(device)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
                type_logits, value_logits = model(images, types)
                loss = nn.functional.cross_entropy(type_logits, types) + nn.functional.cross_entropy(value_logits, values)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        total_loss += float(loss.detach()) * len(images)
        if not training:
            with torch.no_grad():
                pred_types, pred_values = model(images)
                correct += int(((pred_types.argmax(1) == types) & (pred_values.argmax(1) == values)).sum())
        total += len(images)
    return total_loss / total, correct / total if not training else 0.0


def train_reader(processed: Path, output: Path, architecture: str, batch: int = 16) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for the planned reader training")
    seed_everything()
    os.environ.setdefault("TORCH_HOME", str(Path(".cache/torch").resolve()))
    device = torch.device("cuda:0")
    train_loader = DataLoader(CropDataset(processed, "train", training=True), batch_size=batch, shuffle=True, num_workers=0)
    valid_loader = DataLoader(CropDataset(processed, "valid"), batch_size=batch, shuffle=False, num_workers=0)
    model = DieReader(architecture, pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)
    best_accuracy = -1.0
    best_loss = float("inf")
    stale = 0
    folder = output / architecture
    with mlflow.start_run(run_name=f"reader-{architecture}") as run:
        mlflow.set_tags({"stage": "reader", **common_tags(processed)})
        mlflow.log_params({"architecture": architecture, "seed": 42, "epochs_max": 40, "patience": 6,
                           "batch": batch, "lr": 3e-4, "weight_decay": 1e-4, "image_size": 224})
        for epoch in range(1, 41):
            train_loss, _ = epoch_pass(model, train_loader, device, optimizer)
            valid_loss, valid_accuracy = epoch_pass(model, valid_loader, device, None)
            scheduler.step()
            mlflow.log_metrics({"train_loss": train_loss, "valid_loss": valid_loss, "valid_joint_accuracy": valid_accuracy}, step=epoch)
            if (valid_accuracy, -valid_loss) > (best_accuracy, -best_loss):
                best_accuracy, best_loss = valid_accuracy, valid_loss
                stale = 0
                save_reader(model, folder, {"mlflow_run_id": run.info.run_id, **common_tags(processed), "best_epoch": epoch})
            else:
                stale += 1
                if stale >= 6:
                    break
        mlflow.log_artifacts(str(folder), artifact_path="reader")
    return folder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("component", choices=("detector", "readers"))
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()
    if args.component == "detector":
        result = train_detector(args.processed, args.output / "detector")
        print(result)
    else:
        for architecture in ARCHITECTURES:
            print(train_reader(args.processed, args.output / "readers", architecture, args.batch))


if __name__ == "__main__":
    main()
