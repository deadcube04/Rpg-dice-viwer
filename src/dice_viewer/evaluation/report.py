"""Summarize frozen E2 outputs without changing any model or inference parameter."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..data.prepare import sha256
from ..data.verify import verify
from ..training.data import read_jsonl
from ..training.timing import atomic_json
from .run import ROOT, report_directory


def percentage(value):
    return "N/A" if value is None else f"{100 * value:.2f}%"


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def examples(dataset, report, rows, predictions, scenes):
    reasons = {
        "no_candidates": lambda scene: scene["status"] == "no_candidates",
        "too_many_candidates": lambda scene: scene["status"] == "too_many_candidates",
        "missed_objects": lambda scene: scene["counts"]["spatial_fn"] > 0,
        "false_positives": lambda scene: scene["counts"]["spatial_fp"] > 0,
        "type_error": lambda scene: scene["counts"].get("paired_type_support", 0) > scene["counts"].get("paired_type_correct", 0),
        "token_error": lambda scene: scene["counts"].get("paired_value_support", 0) > scene["counts"].get("paired_token_correct", 0),
        "accepted_error": lambda scene: scene["counts"].get("accepted_evaluable", 0) > scene["counts"].get("accepted_correct", 0),
    }
    picked = []
    used = set()
    for name, predicate in reasons.items():
        scene = next((s for s in scenes if predicate(s) and s["image_id"] not in used), None)
        if scene:
            used.add(scene["image_id"])
            picked.append({"category": name, "image_id": scene["image_id"], "counts": scene["counts"]})
    atomic_json(report / "error-examples.json", picked)
    if not picked:
        return
    lookup = {row["image_id"]: row for row in rows}
    outputs = {prediction["image_id"]: prediction for prediction in predictions}
    width, height = 640, 430
    canvas = Image.new("RGB", (width * 2, height * ((len(picked) + 1) // 2)), "white")
    font_path = Path("C:/Windows/Fonts/consola.ttf")
    font = ImageFont.truetype(str(font_path), 12) if font_path.exists() else ImageFont.load_default()
    for index, example in enumerate(picked):
        row = lookup[example["image_id"]]
        with Image.open(dataset / row["prepared_image"]) as source:
            source = source.convert("RGB")
            scale = min(width / source.width, (height - 45) / source.height)
            image = source.resize((round(source.width * scale), round(source.height * scale)))
        panel = Image.new("RGB", (width, height), "white")
        offset = ((width - image.width) // 2, 45)
        panel.paste(image, offset)
        draw = ImageDraw.Draw(panel)
        draw.text((5, 4), example["category"] + " | " + row["image_id"], fill="black", font=font)
        draw.text((5, 21), "GT green; predictions red; unknown labels '?'", fill="black", font=font)
        for obj in row["instances"]:
            coords = [obj["bbox_xyxy"][i] * scale + offset[i % 2] for i in range(4)]
            draw.rectangle(coords, outline="lime", width=2)
            draw.text((coords[0], coords[1]), "GT " + str(obj["die_type"] or "?") + ":" + str(obj["face_token"] or "?"), fill="lime", font=font)
        output = outputs[row["image_id"]]
        for obj in output["items"] or output["candidates"]:
            coords = [obj["bbox_xyxy"][i] * scale + offset[i % 2] for i in range(4)]
            draw.rectangle(coords, outline="red", width=2)
            text = "P " + str(obj.get("die_type", "det")) + ":" + str(obj.get("face_token", "?"))
            draw.text((coords[0], max(45, coords[3] - 15)), text, fill="red", font=font)
        canvas.paste(panel, ((index % 2) * width, (index // 2) * height))
    canvas.save(report / "error-examples.png")


def render(run_id):
    report = report_directory(run_id)
    frozen = json.loads((report / "protocol.json").read_text(encoding="utf-8"))
    cfg = frozen["configuration"]
    dataset = ROOT / cfg["dataset_root"]
    summaries = {}
    latencies = []
    latencies_by_status = []
    for name in cfg["devices"]:
        directory = report / name
        status = json.loads((directory / "status.json").read_text(encoding="utf-8"))
        if status["status"] != "completed" or status["images"] != frozen["images"]:
            raise ValueError(f"Device evaluation incomplete: {name}")
        metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
        timings = json.loads((directory / "timings.json").read_text(encoding="utf-8"))
        summaries[name] = {"status": status, "metrics": metrics, "timings": timings}
        scenes = read_jsonl(directory / "scenes.jsonl")
        for scene_status in sorted({scene["status"] for scene in scenes}):
            times = [scene["timings_seconds"]["total"] for scene in scenes if scene["status"] == scene_status]
            latencies_by_status.append({"device": name, "status": scene_status, "images": len(times),
                                        "mean_seconds": float(np.mean(times)), "p50_seconds": float(np.quantile(times, .5)),
                                        "p95_seconds": float(np.quantile(times, .95))})
        write_csv(directory / "by-count.csv", [{"dice_count": key, **{k: v for k, v in value.items() if k != "counts"}}
                                                for key, value in metrics["by_count"].items()])
        write_csv(directory / "scene-timings.csv", [{"image_id": scene["image_id"], "dice_count": scene["dice_count"],
                                                     "status": scene["status"], **scene["timings_seconds"]} for scene in scenes])
        for count, value in metrics["by_count"].items():
            times = [scene["timings_seconds"]["total"] for scene in scenes if str(scene["dice_count"]) == count]
            latencies.append({"device": name, "dice_count": count, "images": len(times),
                              "mean_seconds": float(np.mean(times)), "p50_seconds": float(np.quantile(times, .5)),
                              "p95_seconds": float(np.quantile(times, .95)), "p99_seconds": float(np.quantile(times, .99))})
        for task, matrix in metrics["confusion_matrices"].items():
            with (directory / f"confusion-{task}.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["truth/prediction", *matrix["labels"]])
                for label, values in zip(matrix["labels"], matrix["matrix"]):
                    writer.writerow([label, *values])
    write_csv(report / "latency-by-count.csv", latencies)
    write_csv(report / "latency-by-status.csv", latencies_by_status)
    primary = summaries["cuda"]["metrics"]
    overall = primary["overall"]
    counts = overall["counts"]
    ap_path = report / "cuda/detector-ap.json"
    ap = json.loads(ap_path.read_text(encoding="utf-8"))
    rows = read_jsonl(dataset / "splits/e2.jsonl")
    examples(dataset, report, rows, read_jsonl(report / "cuda/predictions.jsonl"), read_jsonl(report / "cuda/scenes.jsonl"))
    atomic_json(report / "dataset-integrity-after.json", verify(dataset))
    for component, detail in frozen["candidate"]["results"].items():
        candidate_path = ROOT / "artifacts" / cfg["candidate_id"] / frozen["candidate"]["checkpoints"][component]
        if sha256(candidate_path) != detail["best_checkpoint_sha256"]:
            raise ValueError("Candidate weights changed")
    lines = ["# E2 — avaliação final offline de cenas com 2–8 dados", "",
             f"**Execução:** `{run_id}`. **Candidato congelado:** `{cfg['candidate_id']}`. Nenhum treinamento ou ajuste foi executado.", "",
             f"Conjunto: **{frozen['images']} cenas**, **{frozen['references']} instâncias**, **{frozen['joint_references']} referências com tipo e token conhecidos**. "
             f"Acerto completo de cena é avaliável em **{frozen['fully_labeled_scenes']} cenas**; as demais são excluídas somente dessa métrica.", "",
             "## Resultados principais — CUDA", "",
             "| Métrica | Resultado |", "|---|---:|",
             f"| AP50:95 do detector, avaliador Ultralytics | {ap['metrics']['metrics/mAP50-95(B)']:.6f} |",
             f"| AP50 / AP75 | {ap['metrics']['metrics/mAP50(B)']:.6f} / {ap['AP75']:.6f} |"]
    for label, key in (
        ("Precisão espacial, filtro 0,25", "detector_precision_at_025"),
        ("Recall espacial, filtro 0,25", "detector_recall_at_025"), ("F1 espacial, filtro 0,25", "detector_f1_at_025"),
        ("Recall espacial no limiar 0,95", "detector_recall_at_095"),
        ("Tipo correto entre pares retornados avaliáveis", "type_accuracy_on_matched_items"),
        ("Token correto entre pares retornados avaliáveis", "token_accuracy_on_matched_items"),
        ("Valor normalizado correto entre pares retornados avaliáveis", "normalized_value_accuracy_on_matched_items"),
        ("Precisão de reconhecimento completo", "complete_recognition_precision"),
        ("Recall de reconhecimento completo", "complete_recognition_recall"),
        ("Cena completamente correta", "scene_exact_match"), ("Contagem correta do detector", "scene_count_accuracy"),
        ("Precisão entre aceitos avaliáveis", "accepted_precision"),
        ("Cobertura correta automática", "correct_automatic_coverage"),
        ("Cenas que precisam de intervenção manual (inclui erros)", "manual_intervention_image_rate"),
    ):
        lines.append(f"| {label} | {percentage(overall[key])} |")
    lines += ["", "### Contagens e denominadores", "", "```json", json.dumps(counts, indent=2), "```", "",
              "O pareamento operacional maximiza o número de pares com IoU ≥ 0,50 e, entre eles, a soma de IoU; "
              "o desempate é determinístico e independe de tipos e valores preditos. "
              "Os aceites são pareados separadamente contra todas as referências. Aceitos pareados a rótulos incompletos "
              "não entram na precisão entre aceitos; falsos positivos aceitos entram como erro.", "",
              "Os limiares fixos são detecção ≥ 0,95, tipo ≥ 0,80 e valor ≥ 0,80, com comparação sem arredondamento. "
              "Pares tipo/token incompatíveis exigem revisão. A confiança não foi calibrada.", "",
              "O estado `ok` só indica aceite dos itens retornados: ele pode coexistir com dados omitidos. "
              "Por isso a taxa de intervenção não substitui o recall ou a cobertura automática correta.", "",
              "A AP foi medida em passagem separada pelo avaliador Ultralytics 8.4.173, com filtro 0,001 e máximo técnico 300. "
              "A cascata operacional usa filtro 0,25 e máximo técnico 64. Não se chama o pareamento operacional de avaliador COCO. "
              "Quando há mais de oito candidatos, não há retorno parcial de itens; candidatos internos permanecem no diagnóstico espacial.", "",
              "## Resultados por quantidade de dados — CUDA", "",
              "| Dados na cena | Cenas | Recall espacial | Recall completo | Cena exata | Cobertura automática correta |",
              "|---|---:|---:|---:|---:|---:|"]
    for count, value in primary["by_count"].items():
        lines.append(f"| {count} | {value['counts']['images']} | {percentage(value['detector_recall_at_025'])} | "
                     f"{percentage(value['complete_recognition_recall'])} | {percentage(value['scene_exact_match'])} | "
                     f"{percentage(value['correct_automatic_coverage'])} |")
    lines += ["", "## Resultados por tipo de referência — CUDA", "",
              "| Tipo | Referências | Detectadas | Tipo correto / pares avaliáveis | Token correto / pares avaliáveis | Reconhecimento completo / referências conjuntas |",
              "|---|---:|---:|---:|---:|---:|"]
    for kind, group in primary["object_strata"]["type"].items():
        lines.append(f"| {kind} | {group.get('references', 0)} | {group.get('localized', 0)} | "
                     f"{group.get('paired_type_correct', 0)}/{group.get('paired_type_support', 0)} | "
                     f"{group.get('paired_token_correct', 0)}/{group.get('paired_value_support', 0)} | "
                     f"{group.get('joint_correct', 0)}/{group.get('joint_references', 0)} |")
    lines += ["", "Resultados por modalidade, origem, contagem e tipo×modalidade estão nos JSONs de métricas, com suportes. "
              "Macro-F1 é condicional aos pares classificados e só inclui classes com referência avaliável, listadas junto dos suportes. "
              "Modalidade desconhecida permanece desconhecida. Negativas não pertencem ao conjunto E2 desta execução.", "",
              "## Tempos de inferência offline", "",
              "| Dispositivo | Cenas | Média | p50 | p95 | p99 | Tempo total da execução do dispositivo |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for name, summary in summaries.items():
        timing = summary["timings"]["stages"]["total"]
        lines.append(f"| {name} | {timing['n']} | {timing['mean_seconds']*1000:.2f} ms | {timing['p50_seconds']*1000:.2f} ms | "
                     f"{timing['p95_seconds']*1000:.2f} ms | {timing['p99_seconds']*1000:.2f} ms | {summary['status']['wall_seconds']:.2f} s |")
    lines += ["", "Precisão FP32, concorrência 1, uma imagem por chamada, lotes de até oito recortes. "
              "Três aquecimentos usam uma imagem E1; cold start e carregamento são registrados à parte. "
              "Os percentis usam interpolação linear sobre as cenas diferentes, não repetições da mesma cena. "
              "O tempo por imagem inclui decodificação e cascata, mas exclui pareamento/métricas; é sincronizado em CUDA. "
              "A duração total do dispositivo inclui carregamento, aquecimento, métricas e, em CUDA, AP adicional. "
              "Não são latências HTTP, MinIO, cliente ou medidas de carga concorrente.", "",
              "Os percentis gerais incluem as respostas sem candidatos e os erros de excesso; esses caminhos podem ser mais rápidos. "
              "Os tempos também estão separados por estado em `latency-by-status.csv`.", "",
              "## Limites e estado do projeto", "",
              "Esta execução atende à avaliação offline E2 solicitada diretamente pelo usuário. A composição local necessária foi implementada; "
              "o serviço BentoML/MinIO da Etapa 5 não foi concluído ou aprovado por esta avaliação. "
              "Nenhum resultado de E2 foi usado para alterar pesos, limiares, pré-processamento ou hiperparâmetros.", "",
              "Os grupos e as limitações de cobertura aprovadas permanecem; não se demonstra generalização para coleções ou dados físicos inéditos. "
              "Os 17.450 arquivos do dataset foram conferidos antes e depois; hashes dos pesos também foram preservados. "
              "Nenhum teste automatizado foi criado ou executado. A CPU foi exercitada em processo com CUDA ocultada, usando a wheel CUDA; "
              "não se declara validação de wheel CPU-only ou de serviço em produção. AP adicional em CPU não foi executada.", "",
              "## Artefatos", "",
              "- [Protocolo congelado e definições](./protocol.json).",
              "- [Métricas CUDA](./cuda/metrics.json) e [métricas CPU](./cpu/metrics.json).",
              "- [AP do detector](./cuda/detector-ap.json).",
              "- [Tempos por contagem e dispositivo](./latency-by-count.csv).",
              "- [Predições completas CUDA](./cuda/predictions.jsonl) e [CPU](./cpu/predictions.jsonl).",
              "- [Exemplos de erros e critérios de seleção](./error-examples.json).", ""]
    if (report / "error-examples.png").exists():
        lines += ["![Exemplos reais: referências em verde, predições em vermelho](./error-examples.png)", ""]
    lines += ["## Próximo marco", "", "Apresentar estes resultados ao usuário para aprovação ou rejeição explícita do candidato. "
              "Nenhum modelo foi promovido; a entrega do serviço continua dependendo das etapas de integração e operação previstas no plano.", ""]
    (report / "RELATORIO_E2.md").write_text("\n".join(lines), encoding="utf-8")
    atomic_json(report / "summary.json", {"run_id": run_id, "status": "completed", "candidate_id": cfg["candidate_id"],
                                          "cuda": primary["overall"], "cpu": summaries["cpu"]["metrics"]["overall"],
                                          "latencies": {name: value["timings"] for name, value in summaries.items()},
                                          "model_approved": False, "service_stage5_complete": False,
                                          "weights_or_thresholds_tuned": False})
    print(json.dumps({"status": "documented", "images_per_device": frozen["images"], "cuda_metrics": overall}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    render(parser.parse_args().run_id)
