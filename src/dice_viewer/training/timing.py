"""Monotonic wall-clock measurements, with explicit CUDA synchronization."""

import csv
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import torch


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def synchronize(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def append_jsonl(path, value):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()


def append_csv(path, value):
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(value))
        if not exists:
            writer.writeheader()
        writer.writerow(value)


class RunClock:
    def __init__(self, directory: Path, device: str):
        self.directory, self.device = directory, device
        self.start = time.perf_counter()
        self.started_at = utc_now()
        self.on_epoch = None

    def mark(self):
        synchronize(self.device)
        return time.perf_counter()

    def progress(self, component, **fields):
        atomic_json(self.directory / "progress.json", {
            "status": "running", "updated_at_utc": utc_now(), "started_at_utc": self.started_at,
            "wall_elapsed_seconds": time.perf_counter() - self.start, "device": self.device,
            "component": component, **fields,
        })

    def epoch(self, value):
        append_jsonl(self.directory / "epochs.jsonl", value)
        append_csv(self.directory / "timings.csv", {key: value[key] for key in (
            "component", "phase", "epoch", "train_seconds", "validation_seconds", "checkpoint_seconds",
            "other_seconds", "epoch_wall_seconds", "run_elapsed_seconds", "train_samples", "validation_samples", "batch_size")})
        self.progress(value["component"], phase=value["phase"], completed_epoch=value["epoch"],
                      last_epoch_seconds=value["epoch_wall_seconds"], metrics=value["metrics"])
        print(json.dumps({"event": "epoch_completed", **value}, ensure_ascii=False), flush=True)
        if self.on_epoch is not None:
            self.on_epoch()


def epoch_observation(clock, component, phase, epoch, start, train_seconds, validation_seconds,
                      checkpoint_seconds, train_samples, validation_samples, batch_size, metrics):
    end = clock.mark()
    duration = end - start
    return {"component": component, "phase": phase, "epoch": epoch, "finished_at_utc": utc_now(),
            "train_seconds": train_seconds, "validation_seconds": validation_seconds,
            "checkpoint_seconds": checkpoint_seconds,
            "other_seconds": max(0., duration - train_seconds - validation_seconds - checkpoint_seconds),
            "epoch_wall_seconds": duration, "run_elapsed_seconds": end - clock.start,
            "train_samples": train_samples, "validation_samples": validation_samples,
            "batch_size": batch_size, "metrics": metrics}
