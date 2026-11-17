"""Produce a manual component inventory from real E1 training examples.

No optimizer is instantiated, no backward pass is run and no model is trained.
E2 and validation images are not loaded for model execution in this command.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import save_file

from ..data.prepare import sha256, write_json
from .classifier import MultiheadResNet18
from .checkpoints import load_initial_state
from .detector import build_detector, predict_components
from .labels import LabelMaps
from .losses import MaskedMultitaskLoss
from .preprocessing import crop_body, tensor_from_prepared_crop
from .runtime import seed_components, select_device

ROOT = Path(__file__).resolve().parents[3]


def inspect(args):
    torch.set_num_threads(4)
    seed_components(42)
    selection = select_device(args.device)
    device = selection.resolved
    dataset = ROOT / "data/datasets/v1-20261117-7b23477513de"
    report_dir = ROOT / "reports/stage3-20261117"
    approval = json.loads((ROOT / "data/preparation/dataset-approval.json").read_text(encoding="utf-8"))
    if not approval["approved"] or sha256(dataset / "checksums.jsonl") != approval["checksum_index_sha256"]:
        raise ValueError("Prepared dataset approval/index mismatch")
    # Check file content, not merely existence, before loading an approved label map or crop.
    checksums = {item["path"]: item["sha256"] for item in
                 map(json.loads, (dataset / "checksums.jsonl").read_text(encoding="utf-8").splitlines())}

    def approved_file(relative):
        path = dataset / relative
        if sha256(path) != checksums[relative]:
            raise ValueError(f"Approved data file hash mismatch: {relative}")
        return path

    labels = LabelMaps.load(approved_file("manifests/label_maps.json"))
    crops = [json.loads(line) for line in approved_file("classifier/train.jsonl").read_text(encoding="utf-8").splitlines()]
    rows = {row["image_id"]: row for row in
            map(json.loads, approved_file("splits/train.jsonl").read_text(encoding="utf-8").splitlines())}
    weights = json.loads((report_dir / "pretrained-weights.json").read_text(encoding="utf-8"))
    classifier = MultiheadResNet18(labels, ROOT / weights["classifier"]["path"], weights["classifier"]["sha256"])
    loaded_states = {}
    if args.load_initial_states:
        loaded_states["classifier"] = load_initial_state(
            classifier, ROOT / "artifacts/initial-components-v1", "classifier",
            dataset / "manifests/label_maps.json", ROOT / "config/yolo26n-die.yaml")
    classifier.freeze_backbone()
    classifier.train()
    freeze_inventory = {"backbone_trainable_parameters": sum(p.numel() for p in classifier.backbone.parameters() if p.requires_grad),
                        "backbone_training_mode": classifier.backbone.training,
                        "batchnorm_training_modes": sorted({m.training for m in classifier.backbone.modules() if isinstance(m, torch.nn.BatchNorm2d)}),
                        "type_head_training_mode": classifier.type_head.training}
    classifier.eval().float().to(device)
    # Use real incomplete labels to inspect each empty-task branch, when present.
    selected = [crops[0]]
    for field in ("value_mask", "type_mask"):
        item = next((crop for crop in crops if not crop[field]), None)
        if item and item["crop_id"] not in {c["crop_id"] for c in selected}:
            selected.append(item)
    loss_function = MaskedMultitaskLoss()
    observations = []
    with torch.inference_mode():
        for crop in selected:
            with Image.open(approved_file(crop["path"])) as image:
                tensor = tensor_from_prepared_crop(image).unsqueeze(0).to(device)
            outputs = classifier(tensor)
            losses = loss_function(
                outputs,
                torch.tensor([crop["type_id"] if crop["type_mask"] else -100], device=device, dtype=torch.long),
                torch.tensor([crop["token_id"] if crop["value_mask"] else -100], device=device, dtype=torch.long),
                torch.tensor([crop["type_mask"]], device=device, dtype=torch.bool),
                torch.tensor([crop["value_mask"]], device=device, dtype=torch.bool))
            observations.append({"crop_id": crop["crop_id"], "image_id": crop["image_id"],
                                 "type_mask": crop["type_mask"], "value_mask": crop["value_mask"],
                                 "input_shape": list(tensor.shape),
                                 "output_shapes": {key: list(value.shape) for key, value in outputs.items()},
                                 "finite_logits": all(bool(torch.isfinite(x).all()) for x in outputs.values()),
                                 "losses": {key: float(value) if isinstance(value, torch.Tensor) else value for key, value in losses.items()}})
    crop = selected[0]
    parent = rows[crop["image_id"]]
    with Image.open(approved_file(parent["prepared_image"])) as image:
        original = image.copy()
        recrop, geometry = crop_body(image, crop["bbox_xyxy"])
    with Image.open(approved_file(crop["path"])) as image:
        prepared = tensor_from_prepared_crop(image)
    geometry_inventory = {"crop_id": crop["crop_id"], "metadata": geometry,
                          "same_geometry_as_stage2": torch.equal(recrop, prepared),
                          "max_normalized_pixel_difference": float((recrop - prepared).abs().max())}
    detector = build_detector(ROOT / "config/yolo26n-die.yaml", ROOT / weights["detector"]["path"], weights["detector"]["sha256"])
    if args.load_initial_states:
        loaded_states["detector"] = load_initial_state(
            detector.model, ROOT / "artifacts/initial-components-v1", "detector",
            dataset / "manifests/label_maps.json", ROOT / "config/yolo26n-die.yaml")
    predictions = predict_components(detector, original, device=device)
    detector_inventory = {"head_classes": detector.model.model[-1].nc, "names": detector.names,
                          "end2end": detector.model.model[-1].end2end,
                          "prediction_nms_argument": detector.predictor.args.nms,
                          "backend_end2end": detector.predictor.model.end2end,
                          "backend_fp16": detector.predictor.model.fp16,
                          "output_box_tensor_shapes": [list(result.boxes.data.shape) for result in predictions],
                          "purpose": "shape/loading inspection only; class head has not been trained for dice"}
    if not (all(item["finite_logits"] for item in observations)
            and all(torch.isfinite(torch.tensor(item["losses"]["loss"])) for item in observations)
            and geometry_inventory["same_geometry_as_stage2"]
            and detector_inventory["backend_end2end"] and not detector_inventory["backend_fp16"]):
        raise RuntimeError("Component inspection contains an invalid observation; inspect the console and component state")
    if args.save_initial_states:
        target = ROOT / "artifacts/initial-components-v1"
        target.mkdir(parents=True, exist_ok=True)
        for filename, model in (("resnet18-multihead.safetensors", classifier), ("yolo26n-die.safetensors", detector.model)):
            path = target / filename
            if path.exists():
                raise FileExistsError("Initial model states already exist; use a new artifact version")
            save_file({k: value.detach().cpu().contiguous() for k, value in model.state_dict().items()}, str(path))
        write_json(target / "manifest.json", {
            "status": "initialization_only_not_trained_not_approved_for_serving", "seed": 42,
            "label_maps_sha256": sha256(dataset / "manifests/label_maps.json"),
            "architecture_sha256": sha256(ROOT / "config/yolo26n-die.yaml"),
            "states": {p.name: sha256(p) for p in target.glob("*.safetensors")},
            "pretrained_sources": weights})
    inventory = {"device": asdict(selection), "type_classes": len(labels.types), "token_classes": len(labels.tokens),
                 "classifier": observations, "frozen_backbone": freeze_inventory, "preprocessing": geometry_inventory,
                 "detector": detector_inventory, "training_started": False,
                 "no_backward_or_optimizer_step": True, "e2_or_validation_model_execution": False,
                 "cuda_visible_devices": __import__("os").environ.get("CUDA_VISIBLE_DEVICES"),
                 "loaded_initial_state_hashes": loaded_states,
                 "limitations": "No accuracy, calibration, training batch capacity or latency claim is made from this inspection"}
    destination = report_dir / f"components-{args.report_name}.json"
    write_json(destination, inventory)
    print(json.dumps(inventory, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--report-name", default="cpu")
    parser.add_argument("--save-initial-states", action="store_true")
    parser.add_argument("--load-initial-states", action="store_true")
    inspect(parser.parse_args())
