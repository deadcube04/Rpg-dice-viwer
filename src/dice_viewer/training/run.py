"""Execute the explicitly approved Stage 4 recipe and keep time-comparison records."""

import argparse
import gc
import importlib.metadata
import json
import os
import shutil
import sys
import time
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

from ..data.prepare import sha256
from ..data.verify import verify
from ..models.runtime import select_device
from ..models.setup_components import hardware_inventory
from .classifier import train_classifier
from .detector import train_detector
from .reporting import append_comparison, plot_curves, write_report
from .timing import RunClock, atomic_json, utc_now

ROOT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--run-id", default="v1-e1-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    args = parser.parse_args()
    if not args.run_id or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for char in args.run_id):
        raise ValueError("Run ID must contain only letters, digits, hyphens and underscores")
    torch.set_num_threads(4)
    selection = select_device(args.device, allow_cpu_fallback=True)
    training_root = ROOT / "reports/training"
    training_root.mkdir(parents=True, exist_ok=True)
    active = training_root / "LATEST.json"
    if active.exists() and json.loads(active.read_text(encoding="utf-8")).get("status") == "running":
        raise RuntimeError("Another run is recorded as active; inspect it before starting another")
    report = training_root / args.run_id
    report.mkdir()
    artifact = ROOT / "artifacts" / args.run_id
    artifact.mkdir()
    clock = RunClock(report, selection.resolved)
    initial_status = {"run_id": args.run_id, "status": "running", "pid": os.getpid(),
                      "started_at_utc": clock.started_at, "report": str(report), "artifact": str(artifact)}
    metadata = {"run_id": args.run_id, "started_at_utc": clock.started_at, "device": asdict(selection),
                "dataset_version": "unresolved", "dataset_checksum_index": "unresolved", "recipe_hashes": {},
                "hardware": {}, "versions": {}}
    results = {}
    status, error = "running", None
    try:
        atomic_json(active, initial_status)
        dataset_cfg = yaml.safe_load((ROOT / "config/dataset.yaml").read_text(encoding="utf-8"))
        dataset = ROOT / dataset_cfg["root"]
        configurations = {name: yaml.safe_load((ROOT / f"config/{name}.yaml").read_text(encoding="utf-8")) for name in ("detector", "classifier")}
        metadata.update(
            dataset_version=dataset.name, dataset_checksum_index=dataset_cfg["checksum_index_sha256"],
            recipe_hashes={name: sha256(ROOT / f"config/{name}.yaml") for name in configurations},
            hardware=hardware_inventory(), torch_threads=torch.get_num_threads(),
            versions={name: importlib.metadata.version(name) for name in ("torch", "torchvision", "ultralytics", "pillow", "numpy")},
            approval_reference="ok, aprovado, documente o tempo de treinamente tambem, para eu usar de comparação futuramente",
            approval_scope="Stage 4 recipe and timing documentation; no Stage 5 or model promotion",
            training_configurations=configurations, e2_model_execution=False,
            initial_states=json.loads((ROOT / "artifacts/initial-components-v1/manifest.json").read_text(encoding="utf-8")),
            time_semantics="monotonic synchronized wall-clock seconds; not GPU-kernel-only time",
            comparison_constraints="same dataset/groups, recipes, epochs, stop criterion, device, precision, versions and timing scope")
        atomic_json(report / "run.json", metadata)
        write_report(report, metadata, {}, "running")
        clock.on_epoch = lambda: write_report(report, metadata, results, "running")
        if metadata["versions"]["ultralytics"] != "8.4.173":
            raise ValueError("Training adapter requires the approved pinned Ultralytics version")
        previous = json.loads((ROOT / "reports/stage3-20261117/delivery-checksums.json").read_text(encoding="utf-8"))
        for path, expected in previous["config_files"].items():
            if sha256(ROOT / path) != expected:
                raise ValueError(f"Approved recipe/config changed: {path}")
        environment = json.loads((ROOT / "reports/stage3-20261117/environment.json").read_text(encoding="utf-8"))
        if sha256(ROOT / "uv.lock") != environment["lockfile_sha256"]:
            raise ValueError("Approved lockfile changed")
        source_approval = json.loads((ROOT / "data/preparation/dataset-approval.json").read_text(encoding="utf-8"))
        if not source_approval["approved"] or sha256(dataset / "checksums.jsonl") != source_approval["checksum_index_sha256"]:
            raise ValueError("Approved prepared dataset hash mismatch")
        atomic_json(report / "training-approval.json", {
            "approved": True, "reference": metadata["approval_reference"], "scope": metadata["approval_scope"],
            "dataset_checksum_index": metadata["dataset_checksum_index"], "recipe_hashes": metadata["recipe_hashes"],
            "registered_at_utc": utc_now(), "model_release_approved": False})
        clock.progress("preflight", phase="dataset_integrity")
        atomic_json(report / "dataset-integrity-before.json", verify(dataset))
        provenance = report / "provenance"
        provenance.mkdir()
        for name, paths in (("training", (ROOT / "src/dice_viewer/training").glob("*.py")),
                            ("models", (ROOT / "src/dice_viewer/models").glob("*.py")),
                            ("config", (ROOT / "config").glob("*.yaml")),
                            ("environment", [ROOT / "uv.lock", ROOT / "pyproject.toml", ROOT / ".python-version"])):
            target = provenance / name
            target.mkdir()
            for path in paths:
                shutil.copyfile(path, target / path.name)
        atomic_json(report / "provenance-checksums.json", {p.relative_to(provenance).as_posix(): sha256(p)
                                                          for p in provenance.rglob("*") if p.is_file()})
        clock.progress("detector", phase="setup")
        results["detector"] = train_detector(ROOT, dataset, artifact, report, configurations["detector"], clock)
        write_report(report, metadata, results, "running")
        gc.collect()
        if selection.resolved.startswith("cuda"):
            torch.cuda.empty_cache()
        clock.progress("classifier", phase="setup")
        results["classifier"] = train_classifier(ROOT, dataset, artifact, report, configurations["classifier"], clock)
        clock.progress("finalization", phase="reports_and_dataset_integrity")
        atomic_json(report / "dataset-integrity-after.json", verify(dataset))
        plot_curves(report)
        status = "completed"
    except BaseException as exception:
        status = "interrupted" if isinstance(exception, KeyboardInterrupt) else "failed"
        error = f"{type(exception).__name__}: {exception}"
        (report / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
        print(traceback.format_exc(), file=sys.stderr, flush=True)
    finally:
        elapsed = time.perf_counter() - clock.start
        final = {**initial_status, "status": status, "finished_at_utc": utc_now(),
                 "run_wall_seconds": elapsed, "components": results, "error": error,
                 "next_stage_authorized": False, "candidate_approved": False}
        atomic_json(report / "status.json", final)
        atomic_json(active, final)
        atomic_json(report / "progress.json", final)
        atomic_json(artifact / "candidate.json", {"candidate_id": args.run_id, "status": "candidate" if status == "completed" else "incomplete",
                                                   "training_status": status, "approved": False, "results": results,
                                                   "dataset_checksum_index": metadata["dataset_checksum_index"]})
        write_report(report, metadata, results, status, error)
        append_comparison(training_root, metadata, results, elapsed, status)
        print(json.dumps(final, ensure_ascii=False, indent=2), flush=True)
    if status != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
