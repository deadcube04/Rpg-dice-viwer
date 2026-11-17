"""Pinned Ultralytics training with explicit timing and no silent OOM recovery."""

import json
import math
from copy import deepcopy
from pathlib import Path

import torch
from safetensors.torch import load_file
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import callbacks

from ..data.prepare import sha256
from ..models.checkpoints import load_initial_state
from ..models.detector import build_detector
from ..models.runtime import seed_components
from .classifier import save_state
from .data import create_detector_view
from .timing import atomic_json, epoch_observation


class StrictEarlyStopping:
    """Equal AP, including zero AP, does not reset the approved patience."""
    def __init__(self, patience):
        self.patience = patience
        self.best_fitness = float("-inf")
        self.best_epoch = 0
        self.possible_stop = False

    def __call__(self, epoch, fitness):
        if fitness > self.best_fitness:
            self.best_fitness, self.best_epoch = fitness, epoch
        stale = epoch - self.best_epoch
        self.possible_stop = stale >= self.patience - 1
        return stale >= self.patience


class RecordedDetectionTrainer(DetectionTrainer):
    """API behavior is pinned to Ultralytics 8.4.173 and recorded in source snapshots."""
    def __init__(self, *, clock, report, artifact, **kwargs):
        super().__init__(**kwargs)
        # Keep this academic run local; use default no-op events and our local ledger.
        self.callbacks = callbacks.get_default_callbacks()
        self.clock, self.report, self.artifact = clock, report, artifact
        self.records = []
        self.selected_fitness = -1.
        self.selected_epoch = None
        self.final_validation_seconds = 0.
        self.add_callback("on_train_start", self._start)
        self.add_callback("on_train_epoch_start", self._epoch_start)
        self.add_callback("on_train_batch_end", self._batch_end)
        self.add_callback("on_train_epoch_end", self._train_end)
        self.add_callback("on_fit_epoch_end", self._epoch_end)

    def _start(self, trainer):
        self.stopper = StrictEarlyStopping(self.args.patience)
        self.optimizer_started = self.clock.mark()

    def _epoch_start(self, trainer):
        self.epoch_started = self.clock.mark()
        self.validation_seconds = 0.
        self.native_checkpoint_seconds = 0.
        self.batch_index = 0
        # _do_train checks this before its first-epoch automatic batch-halving branch.
        # A completed batch loop resets it to zero; OOM raises immediately instead.
        self._oom_retries = 3

    def _batch_end(self, trainer):
        if not bool(torch.isfinite(self.loss).all()):
            raise RuntimeError("Non-finite detector loss; stopping without automatic recovery")
        self.batch_index += 1
        if self.batch_index % 20 == 0:
            self.clock.progress("detector", phase="finetune", epoch=self.epoch + 1,
                                batch=self.batch_index, batches=len(self.train_loader))

    def _train_end(self, trainer):
        self.train_seconds = self.clock.mark() - self.epoch_started

    def validate(self):
        start = self.clock.mark()
        metrics = self.validator(self)
        metrics.pop("fitness", None)
        fitness = float(metrics["metrics/mAP50-95(B)"])
        if not math.isfinite(fitness):
            raise RuntimeError("Non-finite detector validation AP")
        if self.best_fitness is None or fitness > self.best_fitness:
            self.best_fitness = fitness
        self.validation_seconds = self.clock.mark() - start
        return metrics, fitness

    def _handle_nan_recovery(self, epoch):
        if not torch.isfinite(self.loss) or not math.isfinite(self.fitness):
            raise RuntimeError("NaN/Inf detector state; automatic checkpoint recovery is disabled")
        return False

    def save_model(self):
        start = self.clock.mark()
        for name, model in (("model", self.model), ("EMA", self.ema.ema)):
            if any(not bool(torch.isfinite(value).all()) for value in model.state_dict().values() if value.is_floating_point()):
                raise RuntimeError(f"Non-finite detector {name}; checkpoint repair is prohibited")
        result = super().save_model()
        self.native_checkpoint_seconds += self.clock.mark() - start
        return result

    def _epoch_end(self, trainer):
        checkpoint_start = self.clock.mark()
        if float(self.fitness) > self.selected_fitness:
            self.selected_fitness = float(self.fitness)
            self.selected_epoch = self.epoch + 1
            # Preserve full FP32 EMA weights; the library's .pt exports may store FP16.
            save_state(self.artifact / "best.safetensors", self.ema.ema.float())
            atomic_json(self.artifact / "best.json", {"epoch": self.selected_epoch, "metrics": self.metrics,
                                                       "sha256": sha256(self.artifact / "best.safetensors"),
                                                       "status": "candidate_not_approved", "weights": "FP32 EMA"})
        save_state(self.artifact / "last.safetensors", self.ema.ema.float())
        checkpoint_seconds = self.native_checkpoint_seconds + self.clock.mark() - checkpoint_start
        row = epoch_observation(self.clock, "detector", "finetune", self.epoch + 1, self.epoch_started,
                                self.train_seconds, self.validation_seconds, checkpoint_seconds,
                                len(self.train_loader.dataset), len(self.test_loader.dataset), self.batch_size,
                                {key: float(value) for key, value in self.metrics.items()})
        self.records.append(row)
        self.clock.epoch(row)

    def final_eval(self):
        # Evaluate the selected FP32 state, not the library's quantized convenience export.
        start = self.clock.mark()
        selected = deepcopy(self.ema.ema).float()
        selected.load_state_dict(load_file(str(self.artifact / "best.safetensors"), device="cpu"), strict=True)
        selected.model[-1].end2end = True
        self.validator.args.quantize = 32
        self.validator.args.nms = False
        self.metrics = self.validator(model=selected)
        self.metrics.pop("fitness", None)
        self.final_validation_seconds = self.clock.mark() - start


def train_detector(root: Path, dataset: Path, artifact: Path, report: Path, configuration, clock):
    beginning = clock.mark()
    seed_components(42)
    recipe = dict(configuration["training"])
    batch_size = recipe.pop("batch_by_device")["cuda" if clock.device.startswith("cuda") else "cpu"]
    recipe.pop("checkpoint_selection")
    recipe.pop("on_out_of_memory")
    output = artifact / "detector"
    output.mkdir(parents=True)
    data_path, data_inventory = create_detector_view(dataset, report / "detector-data")
    atomic_json(report / "detector-data-view.json", data_inventory)
    pretrained = configuration["pretrained"]
    model = build_detector(root / configuration["architecture"], root / pretrained["path"], pretrained["sha256"])
    load_initial_state(model.model, root / "artifacts/initial-components-v1", "detector",
                       dataset / "manifests/label_maps.json", root / configuration["architecture"])
    args = {**recipe, "model": str(root / configuration["architecture"]), "data": str(data_path),
            "device": clock.device, "batch": batch_size, "project": str(report), "name": "detector-fit",
            "exist_ok": False, "save": True, "plots": True, "val": True, "resume": False,
            "pretrained": False, "nms": False, "quantize": 32, "verbose": False}
    trainer = RecordedDetectionTrainer(clock=clock, report=report, artifact=output, overrides=args)
    trainer.model = model.model  # Use the exact approved initial state; no hidden reinitialization.
    if clock.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(clock.device)
    trainer.train()
    result = {"component": "detector", "epochs_completed": len(trainer.records), "selected_epoch": trainer.selected_epoch,
              "selected_AP50_95": trainer.selected_fitness,
              "final_validation_metrics": {key: float(value) for key, value in trainer.metrics.items()},
              "setup_seconds": trainer.optimizer_started - beginning,
              "training_seconds": sum(r["train_seconds"] for r in trainer.records),
              "validation_seconds": sum(r["validation_seconds"] for r in trainer.records),
              "checkpoint_seconds": sum(r["checkpoint_seconds"] for r in trainer.records),
              "final_validation_seconds": trainer.final_validation_seconds,
              "component_wall_seconds": clock.mark() - beginning, "batch_size": batch_size,
              "stop_reason": "early_stopping" if len(trainer.records) < recipe["epochs"] else "epoch_budget",
              "best_checkpoint_sha256": sha256(output / "best.safetensors"),
              "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(clock.device) if clock.device.startswith("cuda") else None}
    atomic_json(report / "detector-result.json", result)
    return result
