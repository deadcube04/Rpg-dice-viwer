"""Freeze the E2 protocol and execute a device-specific, read-only evaluation."""

import argparse
import gc
import importlib.metadata
import json
import os
import shutil
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import yaml
from ultralytics.utils import SETTINGS

from ..data.prepare import sha256
from ..data.verify import verify
from ..models.labels import LabelMaps
from ..models.runtime import seed_components, select_device
from ..models.setup_components import hardware_inventory
from ..training.data import read_jsonl
from ..training.timing import append_jsonl, atomic_json, utc_now
from .cascade import FrozenCascade
from .metrics import aggregate, score_scene

ROOT = Path(__file__).resolve().parents[3]


def report_directory(run_id):
    if not run_id or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for char in run_id):
        raise ValueError("Invalid evaluation run ID")
    return ROOT / "reports/evaluation" / run_id


def freeze(run_id):
    report = report_directory(run_id)
    report.mkdir(parents=True, exist_ok=False)
    cfg_path = ROOT / "config/evaluation-e2.yaml"
    protocol = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    dataset = ROOT / protocol["dataset_root"]
    candidate_path = ROOT / "artifacts" / protocol["candidate_id"] / "candidate.json"
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    if candidate["training_status"] != "completed":
        raise ValueError("E2 requires a completed frozen training candidate")
    train_report = ROOT / "reports/training" / protocol["candidate_id"]
    if sha256(dataset / "checksums.jsonl") != candidate["dataset_checksum_index"]:
        raise ValueError("Dataset differs from the candidate training version")
    for name in ("yolo26n-die.yaml", "preprocessing.yaml"):
        if sha256(ROOT / "config" / name) != sha256(train_report / "provenance/config" / name):
            raise ValueError(f"Training architecture/preprocessing changed: {name}")
    for name in ("classifier.py", "detector.py", "labels.py", "preprocessing.py", "runtime.py"):
        if sha256(ROOT / "src/dice_viewer/models" / name) != sha256(train_report / "provenance/models" / name):
            raise ValueError(f"Model implementation changed since training: {name}")
    atomic_json(report / "dataset-integrity-before.json", verify(dataset))
    rows = read_jsonl(dataset / "splits/e2.jsonl")
    if any(not 2 <= len(row["instances"]) <= 8 or row["split"] != "e2" for row in rows):
        raise ValueError("Non-E2 scene found in the held-out manifest")
    for component in ("detector", "classifier"):
        path = candidate_path.parent / candidate["checkpoints"][component]
        if sha256(path) != candidate["results"][component]["best_checkpoint_sha256"]:
            raise ValueError("Candidate weights changed")
    frozen = {
        "run_id": run_id, "frozen_at_utc": utc_now(), "configuration": protocol,
        "config_sha256": sha256(cfg_path), "candidate_manifest_sha256": sha256(candidate_path),
        "candidate": candidate, "dataset_checksum_index": candidate["dataset_checksum_index"],
        "e2_manifest_sha256": sha256(dataset / "splits/e2.jsonl"),
        "images": len(rows), "references": sum(len(row["instances"]) for row in rows),
        "joint_references": sum(i["type_known"] and i["value_known"] for row in rows for i in row["instances"]),
        "fully_labeled_scenes": sum(all(i["type_known"] and i["value_known"] for i in row["instances"]) for row in rows),
        "approval_reference": "ok, pode seguir para a E2; continued by user after context/token interruption",
        "execution_scope": "Offline cascade and E2 metrics on CUDA/CPU; no training, HTTP/MinIO deployment or model promotion",
        "metric_definitions": {
            "spatial": "maximum-cardinality, then maximum-total-IoU matching; threshold >= 0.50; class independent",
            "recognition": "spatial pair plus exact type and textual token; wrong labels count as a recognition false prediction and missed correct reference",
            "recognition_precision": "correct complete pairs / (returned predictions minus pairs with unknown joint reference); unmatched predictions are evaluable errors",
            "scene_exact": "all reference type/token labels known, all instances correct, no extra returned objects; unknown-label scenes excluded",
            "accepted": "separate matching of accepted predictions against ALL references; unknown-label pairs excluded from accepted precision; unmatched accepted predictions count as errors",
            "automatic_coverage": "correct accepted reference objects / all jointly labeled ground-truth objects",
            "manual_intervention": "images whose status is not ok / all processed images; includes no-candidate and over-eight errors",
            "macro_f1": "conditional on spatially paired, labeled returned items; average over reference-supported classes, with all class supports shown",
            "over_eight": "no partial successful return; internal detector candidates remain available for spatial diagnostics",
            "invalid_geometry": "no successful return; invalid boxes remain in detector diagnostics as unmatched candidates",
            "ap": "Separate pinned Ultralytics evaluator pass on CUDA with conf=0.001/max_det=300; operational cascade uses conf=0.25/max_det=64; not claimed as a full COCO evaluator",
            "timing": "single-image offline synchronized wall clock, decoded image through final policy; excludes model loading and metric matching; not HTTP/client latency",
        },
        "context7": "Resolution attempted; MCP initialization failed. Official Ultralytics docs and installed source inspected.",
        "documentation": ["https://docs.ultralytics.com/modes/val/", "https://docs.ultralytics.com/reference/engine/results/"],
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "torchvision", "ultralytics", "pillow", "numpy")},
        "supporting_hashes": {str(path.relative_to(ROOT)): sha256(path) for path in
                              [*(ROOT / "src/dice_viewer/models").glob("*.py"),
                               *(ROOT / "src/dice_viewer/data").glob("*.py"),
                               ROOT / "config/yolo26n-die.yaml", ROOT / "config/preprocessing.yaml",
                               dataset / "manifests/label_maps.json"]},
    }
    if frozen["versions"]["ultralytics"] != "8.4.173":
        raise ValueError("The evaluation adapter requires pinned Ultralytics 8.4.173")
    atomic_json(report / "protocol.json", frozen)
    provenance = report / "provenance"
    provenance.mkdir()
    for source in (ROOT / "src/dice_viewer/evaluation").glob("*.py"):
        shutil.copyfile(source, provenance / source.name)
    shutil.copyfile(cfg_path, provenance / cfg_path.name)
    shutil.copyfile(dataset / "splits/e2.jsonl", provenance / "e2.jsonl")
    atomic_json(report / "protocol-checksums.json", {
        str(p.relative_to(report)): sha256(p) for p in report.rglob("*") if p.is_file()
    })
    print(json.dumps({"status": "frozen", "run_id": run_id, "images": len(rows), "references": frozen["references"]}), flush=True)


def ap_dataset(dataset, rows, directory):
    for kind, extension in (("images", ".png"), ("labels", ".txt")):
        target = directory / kind / "val"
        target.mkdir(parents=True)
        for row in rows:
            source = dataset / "yolo" / kind / "e2" / (row["image_id"] + extension)
            path = target / source.name
            if kind == "images":
                try:
                    os.link(source, path)
                    continue
                except OSError:
                    pass
            shutil.copyfile(source, path)
    path = directory / "dataset.yaml"
    path.write_text(yaml.safe_dump({"path": str(directory.resolve()), "train": [], "val": "images/val", "names": {0: "die"}}), encoding="utf-8")
    return path


def evaluate(run_id, device_request, name):
    report = report_directory(run_id)
    frozen = json.loads((report / "protocol.json").read_text(encoding="utf-8"))
    for relative, expected in json.loads((report / "protocol-checksums.json").read_text(encoding="utf-8")).items():
        if sha256(report / relative) != expected:
            raise ValueError("Frozen protocol/provenance changed before evaluation")
    cfg = frozen["configuration"]
    for package, version in frozen["versions"].items():
        if importlib.metadata.version(package) != version:
            raise ValueError(f"Library version changed since freeze: {package}")
    for relative, expected in frozen["supporting_hashes"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"Model/data implementation or input changed since freeze: {relative}")
    if sha256(ROOT / "config/evaluation-e2.yaml") != frozen["config_sha256"]:
        raise ValueError("Current evaluation settings differ from the frozen protocol")
    for source in (ROOT / "src/dice_viewer/evaluation").glob("*.py"):
        if sha256(source) != sha256(report / "provenance" / source.name):
            raise ValueError("Evaluation implementation differs from the frozen source snapshot")
    dataset = ROOT / cfg["dataset_root"]
    if sha256(dataset / "splits/e2.jsonl") != frozen["e2_manifest_sha256"]:
        raise ValueError("Held-out scene manifest changed")
    if sha256(ROOT / "artifacts" / cfg["candidate_id"] / "candidate.json") != frozen["candidate_manifest_sha256"]:
        raise ValueError("Candidate metadata changed after protocol freeze")
    target = report / name
    target.mkdir(exist_ok=False)
    torch.set_num_threads(cfg["torch_threads"])
    SETTINGS.update({"sync": False})
    seed_components(cfg["seed"])
    selection = select_device(device_request, allow_cpu_fallback=False)
    if name == "cuda" and not selection.resolved.startswith("cuda"):
        raise ValueError("CUDA measurement requires actual CUDA")
    if name == "cpu" and selection.resolved != "cpu":
        raise ValueError("CPU measurement requires actual CPU")
    start = time.perf_counter()
    started_at = utc_now()
    atomic_json(target / "status.json", {"status": "running", "started_at_utc": started_at, "device": selection.resolved})
    try:
        rows = read_jsonl(dataset / "splits/e2.jsonl")
        cascade = FrozenCascade(ROOT, cfg, selection.resolved)
        loaded = time.perf_counter()
        train_row = read_jsonl(dataset / "splits/train.jsonl")[0]
        warmups = []
        for _ in range(cfg["timing"]["warmup_e1_images"]):
            warmups.append(cascade.infer(dataset / train_row["prepared_image"])["timings_seconds"]["total"])
        scored = []
        for index, row in enumerate(rows):
            output = cascade.infer(dataset / row["prepared_image"])
            scene = score_scene(row, output, cfg["matching"]["iou_threshold"])
            append_jsonl(target / "predictions.jsonl", {"image_id": row["image_id"], **output})
            append_jsonl(target / "scenes.jsonl", {key: value for key, value in scene.items() if key not in ("objects", "confusion")})
            for obj in scene["objects"]:
                append_jsonl(target / "objects.jsonl", obj)
            scored.append(scene)
            if (index + 1) % 20 == 0 or index + 1 == len(rows):
                progress = {"status": "running", "device": selection.resolved, "completed_images": index + 1,
                            "total_images": len(rows), "elapsed_seconds": time.perf_counter() - start, "updated_at_utc": utc_now()}
                atomic_json(target / "progress.json", progress)
                print(json.dumps(progress), flush=True)
        labels = LabelMaps.load(dataset / "manifests/label_maps.json")
        metrics = aggregate(scored, labels)
        measurements = {}
        for key in scored[0]["timings_seconds"]:
            values = np.array([scene["timings_seconds"][key] for scene in scored], dtype=np.float64)
            measurements[key] = {"n": len(values), "mean_seconds": float(values.mean()),
                                 "p50_seconds": float(np.quantile(values, .5)), "p95_seconds": float(np.quantile(values, .95)),
                                 "p99_seconds": float(np.quantile(values, .99)), "min_seconds": float(values.min()),
                                 "max_seconds": float(values.max()), "sum_seconds": float(values.sum())}
        atomic_json(target / "metrics.json", metrics)
        atomic_json(target / "timings.json", {"stages": measurements, "cold_first_e1_seconds": warmups[0],
                                               "e1_warmup_seconds": warmups, "model_load_seconds": loaded - start,
                                               "scope": frozen["metric_definitions"]["timing"]})
        if name == cfg["ap"]["device"]:
            ap_start = time.perf_counter()
            path = ap_dataset(dataset, rows, target / "ap-data")
            ap_args = {key: value for key, value in cfg["ap"].items() if key not in ("device", "evaluator")}
            ap = cascade.detector.val(data=str(path), imgsz=cfg["detector"]["imgsz"], device=selection.resolved,
                                      project=str(target), name="detector-ap", exist_ok=False,
                                      plots=True, save_json=False, verbose=False, **ap_args)
            atomic_json(target / "detector-ap.json", {"metrics": {key: float(value) for key, value in ap.results_dict.items()},
                                                       "AP75": float(ap.box.map75), "parameters": cfg["ap"],
                                                       "wall_seconds": time.perf_counter() - ap_start})
        atomic_json(target / "hardware.json", hardware_inventory())
        candidate = ROOT / "artifacts" / cfg["candidate_id"]
        for component, detail in frozen["candidate"]["results"].items():
            if sha256(candidate / frozen["candidate"]["checkpoints"][component]) != detail["best_checkpoint_sha256"]:
                raise ValueError("Weights changed during evaluation")
        final = {"status": "completed", "started_at_utc": started_at, "finished_at_utc": utc_now(),
                 "wall_seconds": time.perf_counter() - start, "device": selection.resolved,
                 "device_selection_reason": selection.reason, "images": len(rows),
                 "training_executed": False, "parameters_tuned": False, "model_promoted": False}
        atomic_json(target / "status.json", final)
        print(json.dumps(final), flush=True)
    except BaseException as error:
        atomic_json(target / "status.json", {"status": "failed", "error": str(error),
                                             "elapsed_seconds": time.perf_counter() - start})
        (target / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "evaluate"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--name", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.action == "freeze":
        freeze(args.run_id)
    else:
        evaluate(args.run_id, args.device, args.name)


if __name__ == "__main__":
    main()
