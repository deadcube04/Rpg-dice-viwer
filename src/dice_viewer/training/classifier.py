"""Two-phase ResNet18 training under the approved independent-mask recipe."""

import os
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader

from ..data.prepare import sha256
from ..models.checkpoints import load_initial_state
from ..models.classifier import MultiheadResNet18
from ..models.labels import LabelMaps
from ..models.losses import MaskedMultitaskLoss
from ..models.runtime import seed_components
from .data import CropDataset
from .evaluation import evaluate_classifier
from .timing import atomic_json, epoch_observation


def save_state(path, model):
    if any(not bool(torch.isfinite(value).all()) for value in model.state_dict().values() if value.is_floating_point()):
        raise RuntimeError("Non-finite model state; candidate serialization is prohibited")
    temporary = path.with_suffix(path.suffix + ".tmp")
    save_file({key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}, str(temporary))
    os.replace(temporary, path)


def train_classifier(root: Path, dataset: Path, artifact: Path, report: Path, configuration, clock):
    beginning = clock.mark()
    device = clock.device
    recipe = configuration["training"]
    seed_components(configuration["seed"])
    batch_size = recipe["batch_by_device"]["cuda" if device.startswith("cuda") else "cpu"]
    labels_path = dataset / "manifests/label_maps.json"
    labels = LabelMaps.load(labels_path)
    model = MultiheadResNet18(labels)
    load_initial_state(model, root / "artifacts/initial-components-v1", "classifier", labels_path, root / "config/yolo26n-die.yaml")
    model.float().to(device)
    model.freeze_backbone(True)
    criterion = MaskedMultitaskLoss(**{key + "_weight": value for key, value in recipe["loss_weights"].items()})
    train_data = CropDataset(dataset, "train", recipe["augmentation"])
    val_data = CropDataset(dataset, "val")
    generator = torch.Generator().manual_seed(configuration["seed"])
    options = {"batch_size": batch_size, "num_workers": recipe["workers"], "pin_memory": device.startswith("cuda")}
    train_loader = DataLoader(train_data, shuffle=True, generator=generator, **options)
    val_loader = DataLoader(val_data, shuffle=False, **options)
    optimizer = torch.optim.AdamW([
        {"params": [*model.type_head.parameters(), *model.value_head.parameters()], "lr": recipe["head_lr"]}
    ], lr=recipe["head_lr"], betas=tuple(recipe["betas"]), eps=recipe["eps"], weight_decay=recipe["weight_decay"])
    output = artifact / "classifier"
    output.mkdir(parents=True)
    best_rank = (-1., float("-inf"))
    best_epoch = None
    stale = 0
    best_metrics = None
    epoch_rows = []
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    setup_seconds = clock.mark() - beginning
    for epoch in range(1, recipe["epochs_total"] + 1):
        phase = "heads" if epoch <= recipe["head_only_epochs"] else "full_finetune"
        epoch_start = clock.mark()
        if epoch == recipe["head_only_epochs"] + 1:
            model.freeze_backbone(False)
            optimizer.add_param_group({"params": model.backbone.parameters(), "lr": recipe["backbone_lr"]})
            stale = 0
        model.train()
        sums = {"type": 0., "value": 0.}
        counts = {"type": 0, "value": 0}
        for index, batch in enumerate(train_loader):
            images = batch["image"].to(device, non_blocking=True)
            targets = [batch[key].to(device) for key in ("type_target", "value_target", "type_mask", "value_mask")]
            optimizer.zero_grad(set_to_none=True)
            losses = criterion(model(images), *targets)
            if not torch.isfinite(losses["loss"]):
                raise RuntimeError("Non-finite classifier training loss; no automatic recovery")
            losses["loss"].backward()
            optimizer.step()
            for task in ("type", "value"):
                count = losses[task + "_count"]
                counts[task] += count
                sums[task] += float(losses[task + "_loss"].detach()) * count
            if index % 25 == 0:
                clock.progress("classifier", phase=phase, epoch=epoch, batch=index + 1,
                               batches=len(train_loader), epoch_elapsed_seconds=time.perf_counter() - epoch_start)
        train_seconds = clock.mark() - epoch_start
        val_start = clock.mark()
        metrics = evaluate_classifier(model, val_loader, criterion, device, labels)
        validation_seconds = clock.mark() - val_start
        metrics.update({"train_type_loss": sums["type"] / max(1, counts["type"]),
                        "train_value_loss": sums["value"] / max(1, counts["value"])})
        rank = metrics["joint_accuracy"], -metrics["loss"]
        checkpoint_start = clock.mark()
        if rank > best_rank:
            best_rank, best_epoch, best_metrics = rank, epoch, dict(metrics)
            stale = 0
            save_state(output / "best.safetensors", model)
            atomic_json(output / "best.json", {"epoch": epoch, "phase": phase, "metrics": metrics,
                                                "sha256": sha256(output / "best.safetensors"),
                                                "status": "candidate_not_approved"})
        elif phase == "full_finetune":
            stale += 1
        save_state(output / "last.safetensors", model)
        state_path = output / "last-training-state.pt"
        torch.save({"epoch": epoch, "phase": phase, "optimizer": optimizer.state_dict(),
                    "torch_rng": torch.get_rng_state(), "sampler_rng": generator.get_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if device.startswith("cuda") else [],
                    "best_epoch": best_epoch, "best_rank": best_rank, "full_phase_stale_epochs": stale},
                   state_path.with_suffix(".tmp"))
        os.replace(state_path.with_suffix(".tmp"), state_path)
        checkpoint_seconds = clock.mark() - checkpoint_start
        row = epoch_observation(clock, "classifier", phase, epoch, epoch_start, train_seconds,
                                validation_seconds, checkpoint_seconds, len(train_data), len(val_data), batch_size, metrics)
        epoch_rows.append(row)
        clock.epoch(row)
        if phase == "full_finetune" and stale >= recipe["early_stopping_patience_full_finetune"]:
            break
    final_start = clock.mark()
    model.load_state_dict(load_file(str(output / "best.safetensors"), device="cpu"), strict=True)
    final_metrics = evaluate_classifier(model, val_loader, criterion, device, labels, report / "classifier-validation")
    final_validation_seconds = clock.mark() - final_start
    result = {"component": "classifier", "epochs_completed": len(epoch_rows), "selected_epoch": best_epoch,
              "selected_metrics": best_metrics, "final_validation_metrics": final_metrics,
              "setup_seconds": setup_seconds, "training_seconds": sum(r["train_seconds"] for r in epoch_rows),
              "validation_seconds": sum(r["validation_seconds"] for r in epoch_rows),
              "checkpoint_seconds": sum(r["checkpoint_seconds"] for r in epoch_rows),
              "final_validation_seconds": final_validation_seconds,
              "component_wall_seconds": clock.mark() - beginning, "batch_size": batch_size,
              "stop_reason": "early_stopping" if len(epoch_rows) < recipe["epochs_total"] else "epoch_budget",
              "best_checkpoint_sha256": sha256(output / "best.safetensors"),
              "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.startswith("cuda") else None}
    atomic_json(report / "classifier-result.json", result)
    return result
