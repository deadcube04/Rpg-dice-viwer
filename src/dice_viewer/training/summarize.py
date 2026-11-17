"""Document measured phase durations and inspect actual trained candidate files."""

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open

from ..data.prepare import sha256
from .timing import atomic_json

ROOT = Path(__file__).resolve().parents[3]


def hms(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{secs:05.2f}"


def write_csv(path, values):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)


def summarize(run_id):
    report = ROOT / "reports/training" / run_id
    status = json.loads((report / "status.json").read_text(encoding="utf-8"))
    run = json.loads((report / "run.json").read_text(encoding="utf-8"))
    if status["status"] != "completed":
        raise ValueError("Only completed runs can become a training-time reference")
    rows = [json.loads(line) for line in (report / "epochs.jsonl").read_text(encoding="utf-8").splitlines()]
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["component"], row["phase"])].append(row)
    phases = []
    for (component, phase), entries in grouped.items():
        phases.append({"run_id": run_id, "component": component, "phase": phase, "epochs": len(entries),
                       "first_epoch": min(row["epoch"] for row in entries), "last_epoch": max(row["epoch"] for row in entries),
                       "training_seconds": sum(row["train_seconds"] for row in entries),
                       "validation_seconds": sum(row["validation_seconds"] for row in entries),
                       "checkpoint_seconds": sum(row["checkpoint_seconds"] for row in entries),
                       "epoch_total_seconds": sum(row["epoch_wall_seconds"] for row in entries),
                       "median_epoch_seconds": statistics.median(row["epoch_wall_seconds"] for row in entries),
                       "minimum_epoch_seconds": min(row["epoch_wall_seconds"] for row in entries),
                       "maximum_epoch_seconds": max(row["epoch_wall_seconds"] for row in entries),
                       "training_samples_per_second": sum(row["train_samples"] for row in entries) / sum(row["train_seconds"] for row in entries)})
    write_csv(report / "PHASE_TIMES.csv", phases)
    # Inspect candidate files without running a model or loading pickle checkpoints.
    inspection = {}
    artifact = ROOT / "artifacts" / run_id
    for component, result in status["components"].items():
        path = artifact / component / "best.safetensors"
        if sha256(path) != result["best_checkpoint_sha256"]:
            raise ValueError(f"Candidate hash mismatch: {component}")
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            tensors = {name: handle.get_tensor(name) for name in handle.keys()}
        nonfinite = [name for name, tensor in tensors.items() if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all())]
        dtypes = sorted({str(tensor.dtype) for tensor in tensors.values() if tensor.is_floating_point()})
        if nonfinite or dtypes != ["torch.float32"]:
            raise ValueError(f"Invalid candidate state: {component}, {nonfinite}, {dtypes}")
        inspection[component] = {"sha256": sha256(path), "tensor_count": len(tensors), "floating_dtypes": dtypes,
                                 "nonfinite_tensors": nonfinite, "bytes": path.stat().st_size,
                                 "selected_epoch": result["selected_epoch"], "approved": False}
        if component == "classifier":
            inspection[component]["type_head_shape"] = list(tensors["type_head.weight"].shape)
            inspection[component]["value_head_shape"] = list(tensors["value_head.weight"].shape)
        del tensors
    atomic_json(report / "candidate-inspection.json", inspection)
    components = status["components"]
    outside_components = status["run_wall_seconds"] - sum(result["component_wall_seconds"] for result in components.values())
    atomic_json(report / "TIMING_REFERENCE.json", {
        "run_id": run_id, "started_at_utc": status["started_at_utc"], "finished_at_utc": status["finished_at_utc"],
        "run_wall_seconds": status["run_wall_seconds"], "outside_component_seconds": outside_components,
        "phases": phases, "components": components, "hardware": run["hardware"], "versions": run["versions"],
        "recipe_hashes": run["recipe_hashes"], "dataset_checksum_index": run["dataset_checksum_index"],
        "time_semantics": run["time_semantics"], "postprocessing_not_included_in_run_time": True,
    })
    strata = json.loads((report / "classifier-validation/strata.json").read_text(encoding="utf-8"))
    by_type = defaultdict(lambda: defaultdict(int))
    for entry in strata:
        for key, value in entry.items():
            if isinstance(value, int):
                by_type[entry["die_type"]][key] += value
    metrics = components["classifier"]["final_validation_metrics"]
    counts = {name: round(metrics[name + "_accuracy"] * metrics[support + "_support"])
              for name, support in (("type", "type"), ("token", "value"), ("joint", "joint"))}
    pips = [entry for entry in strata if entry["die_type"] == "d6" and entry["modality"] == "pips"]
    pips_correct = sum(entry.get("value_correct", 0) for entry in pips)
    pips_support = sum(entry.get("value_support", 0) for entry in pips)
    hardware = run["hardware"]
    classifier_recipe = run["training_configurations"]["classifier"]["training"]
    detector_recipe = run["training_configurations"]["detector"]["training"]
    def accuracy(entry, numerator, denominator):
        if not entry.get(denominator, 0):
            return "— (sem suporte)"
        return f"{entry.get(numerator, 0)}/{entry.get(denominator, 0)} ({100 * entry.get(numerator, 0) / max(1, entry.get(denominator, 0)):.2f}%)"
    lines = ["# Referência de tempo de treinamento — V1 E1", "",
             f"**Execução concluída:** `{run_id}`. **Total medido: {hms(status['run_wall_seconds'])}**.", "",
             "Tempos formatados como `horas:minutos:segundos`; os valores numéricos de CSV e JSON estão em segundos.", "",
             f"Início UTC: `{status['started_at_utc']}`. Fim UTC: `{status['finished_at_utc']}`.", "",
             "## Tempos por componente", "",
             "| Componente | Épocas | Selecionada | Só treino | Validação durante as épocas | Checkpoints | Total do componente |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, result in components.items():
        lines.append(f"| {name} | {result['epochs_completed']} | {result['selected_epoch']} | {hms(result['training_seconds'])} | "
                     f"{hms(result['validation_seconds'])} | {hms(result['checkpoint_seconds'])} | {hms(result['component_wall_seconds'])} |")
    lines += ["", f"Preparação e finalização externas aos componentes: **{outside_components:.2f} s**. "
              "O total de cada componente inclui preparação, épocas, checkpoints e validação final; "
              "por isso ele é maior que a soma apenas de treino e validação durante as épocas.", "",
              "## Tempos por fase", "",
              "| Componente / fase | Épocas | Só treino | Validação | Total das épocas | Mediana por época |",
              "|---|---:|---:|---:|---:|---:|"]
    for phase in phases:
        lines.append(f"| {phase['component']} / {phase['phase']} | {phase['epochs']} | {hms(phase['training_seconds'])} | "
                     f"{hms(phase['validation_seconds'])} | {hms(phase['epoch_total_seconds'])} | {phase['median_epoch_seconds']:.2f} s |")
    lines += ["", f"`heads`: backbone e BatchNorm congelados. `full_finetune`: backbone descongelado na época {classifier_recipe['head_only_epochs'] + 1}, "
              "preservando o estado AdamW das cabeças. "
              f"Critérios: paciência {detector_recipe['patience']} no detector e {classifier_recipe['early_stopping_patience_full_finetune']} na fase completa do classificador. "
              f"Motivos de parada: detector `{components['detector']['stop_reason']}`; classificador `{components['classifier']['stop_reason']}`.", "",
              "## Condições para comparação futura", "",
              f"- GPU: {hardware.get('gpu', {}).get('name', 'sem GPU')}; CPU: {hardware['cpu']}; RAM: {hardware['ram_bytes'] / 1024**3:.2f} GiB; threads PyTorch: {run['torch_threads']}.",
              f"- Plataforma: `{hardware['platform']}`; dispositivo: `{run['device']['resolved']}`. Precisão, seed, resoluções e workers constam nas receitas preservadas.",
              f"- Lotes efetivos: detector {components['detector']['batch_size']}; classificador {components['classifier']['batch_size']}.",
              f"- PyTorch {run['versions']['torch']}, TorchVision {run['versions']['torchvision']} e Ultralytics {run['versions']['ultralytics']}.",
              "- Dataset, agrupamentos e receitas vinculados a hashes em [run.json](./run.json).",
              "- Relógio monotônico com sincronização CUDA nas fronteiras; treino inclui leitura, aumentos, transferências e otimização.",
              "- Não são tempos exclusivos de kernels GPU. Cache do sistema, temperatura e outros processos podem afetar a medição.",
              "- Importações anteriores ao relógio, implementação/revisão do código e documentação posterior não entram no total medido.",
              "- Tentativas anteriores que falharam são preservadas separadamente e não são somadas a esta referência; consulte `TENTATIVAS.md`.",
              "- Compare também tempo por época e por fase: a parada antecipada pode produzir quantidades diferentes de épocas.", "",
              "## Resultados de validação E1", "",
              f"Detector selecionado: AP50:95 **{components['detector']['final_validation_metrics']['metrics/mAP50-95(B)']:.6f}**, época {components['detector']['selected_epoch']}.", "",
              f"Classificador selecionado: época {components['classifier']['selected_epoch']}; "
              f"tipo **{counts['type']}/{metrics['type_support']} = {metrics['type_accuracy'] * 100:.2f}%**; "
              f"token **{counts['token']}/{metrics['value_support']} = {metrics['token_accuracy'] * 100:.2f}%**; "
              f"tipo+token **{counts['joint']}/{metrics['joint_support']} = {metrics['joint_accuracy'] * 100:.2f}%**. "
              "Os denominadores diferem pelas máscaras independentes.", "",
              "| Tipo | Tipo correto | Token correto | Tipo + token corretos |",
              "|---|---:|---:|---:|"]
    for kind, entry in sorted(by_type.items()):
        lines.append(f"| {kind} | {accuracy(entry, 'type_correct', 'type_support')} | {accuracy(entry, 'value_correct', 'value_support')} | {accuracy(entry, 'joint_correct', 'joint_support')} |")
    lines += ["", "Interprete as diferenças entre tipo e token com os suportes da tabela. "
              f"D20: {by_type['d20'].get('value_correct', 0)}/{by_type['d20'].get('value_support', 0)} tokens corretos. "
              f"D4: suporte de tipo {by_type['d4'].get('type_support', 0)}. "
              f"D6 por pontos: {pips_correct}/{pips_support} tokens corretos. "
              "Resultados com poucas amostras não demonstram generalização. As modalidades anotadas foram preservadas.", "",
              "Estes resultados usam validação para seleção; **E2 não foi executada**. Permanecem as lacunas de cobertura já aprovadas. "
              "Os pesos são candidatos, sem aprovação para uso e sem calibração. A integração da Etapa 5 depende de novo sinal do usuário.", "",
              "## Arquivos para reutilização", "",
              "- [CSV por época](./timings.csv) e [CSV por fase](./PHASE_TIMES.csv).",
              "- [Referência estruturada JSON](./TIMING_REFERENCE.json) e [curvas de aprendizado e tempo](./curves.png).",
              "- [Comparação por componente entre execuções](../TRAINING_COMPARISON.csv): `run_wall_seconds` se repete nas duas linhas; **não somar esse campo**.",
              "- [Uma linha por execução](../TRAINING_RUNS.csv), para comparar totais sem duplicação.",
              "- [Tentativas e falha inicial](../TENTATIVAS.md).",
              "- [Inspeção dos candidatos FP32 e hashes](./candidate-inspection.json).",
              "- [Relatório automático completo](./RELATORIO_TREINAMENTO.md).", ""]
    lines += ["Para regenerar as tabelas de referência de uma execução concluída, use `python -m dice_viewer.training.summarize <run_id>` no ambiente do projeto. "
              "Os dados originais por época são preservados; esse comando não executa treinamento ou inferência.", ""]
    (report / "TEMPOS_TREINAMENTO.md").write_text("\n".join(lines), encoding="utf-8")
    comparison = ROOT / "reports/training/TRAINING_COMPARISON.csv"
    unique = {}
    for row in csv.DictReader(comparison.open(encoding="utf-8", newline="")):
        unique[row["run_id"]] = {key: row[key] for key in (
            "run_id", "status", "started_at_utc", "device", "gpu", "dataset", "dataset_checksum_index", "run_wall_seconds", "torch", "ultralytics")}
    write_csv(comparison.with_name("TRAINING_RUNS.csv"), list(unique.values()))
    print(json.dumps({"status": "documented", "run_id": run_id, "total_seconds": status["run_wall_seconds"],
                      "components_inspected": list(inspection), "phases": len(phases)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    summarize(parser.parse_args().run_id)
