"""Comparable wall-clock durations and candidate reports, updated per component."""

import json
import statistics
from pathlib import Path

from .timing import atomic_json, append_csv, utc_now


def duration(seconds):
    return f"{seconds / 3600:.3f} h ({seconds:.2f} s)"


def write_report(report: Path, metadata, results, status, error=None):
    epoch_path = report / "epochs.jsonl"
    epochs = [json.loads(line) for line in epoch_path.read_text(encoding="utf-8").splitlines()] if epoch_path.exists() else []
    summaries = {}
    for component in ("detector", "classifier"):
        rows = [row for row in epochs if row["component"] == component]
        if rows:
            summaries[component] = {
                "epochs_completed": len(rows), "training_seconds": sum(row["train_seconds"] for row in rows),
                "validation_seconds": sum(row["validation_seconds"] for row in rows),
                "epoch_wall_seconds": sum(row["epoch_wall_seconds"] for row in rows),
                "mean_epoch_seconds": statistics.mean(row["epoch_wall_seconds"] for row in rows),
                "median_epoch_seconds": statistics.median(row["epoch_wall_seconds"] for row in rows),
                "first_epoch_seconds": rows[0]["epoch_wall_seconds"],
                "median_after_first_seconds": statistics.median(row["epoch_wall_seconds"] for row in rows[1:]) if len(rows) > 1 else None,
                "minimum_epoch_seconds": min(row["epoch_wall_seconds"] for row in rows),
                "maximum_epoch_seconds": max(row["epoch_wall_seconds"] for row in rows),
            }
    atomic_json(report / "timing-summary.json", {"status": status, "run_id": metadata["run_id"],
                                                "updated_at_utc": utc_now(), "components": summaries,
                                                "completed_component_results": results})
    lines = ["# Etapa 4 — treinamento E1 e tempos de referência", "",
             f"**Execução:** `{metadata['run_id']}`. **Estado:** `{status}`.", "",
             f"Início em UTC: `{metadata['started_at_utc']}`. Dispositivo: `{metadata['device']['resolved']}`.", "",
             "## Tempos medidos", "",
             "| Componente | Épocas concluídas | Treino | Validação por época | Total das épocas | Mediana por época |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, summary in summaries.items():
        lines.append(f"| {name} | {summary['epochs_completed']} | {duration(summary['training_seconds'])} | "
                     f"{duration(summary['validation_seconds'])} | {duration(summary['epoch_wall_seconds'])} | "
                     f"{summary['median_epoch_seconds']:.2f} s |")
    if not summaries:
        lines.append("| Preparação / primeira época em andamento | 0 | — | — | — | — |")
    lines += ["", "Os valores são medições de relógio de parede, em segundos, por `time.perf_counter`, com sincronização CUDA nas fronteiras das fases. "
              "O treino inclui leitura de dados, aumentos, transferências, forward, backward e passos de otimizador. "
              "Validação e serialização são registradas separadamente. Não são tempos exclusivos de kernels GPU.", "",
              "`timings.csv` contém uma linha por época; `epochs.jsonl` inclui métricas, fase e timestamps. "
              "`progress.json` registra o andamento entre épocas. `timing-summary.json` oferece médias, medianas, extremos e primeira época.", "",
              "Os totais de componente também incluem montagem/carregamento, validação final e serialização. "
              "O total da execução inclui ainda a conferência de integridade, preparação das entradas e relatórios. "
              "A marca inicial ocorre depois da importação das bibliotecas e seleção do dispositivo; "
              "a gravação final de status e resumo ocorre após a última marca. "
              "Compare execuções com o mesmo escopo e considere cache do sistema, temperatura, outros processos e parada antecipada.", "",
              "## Configuração e proveniência", "",
              f"- Dataset: `{metadata['dataset_version']}`; índice SHA-256 `{metadata['dataset_checksum_index']}`.",
              "- Hardware, versões, dispositivo, número de threads, configurações e estados iniciais estão em `run.json` e no snapshot `provenance/`.",
              "- Detector: FP32, até 100 épocas, lote CUDA 8 / CPU 2, SGD, paciência 15; seleção por AP50:95 em validação E1.",
              "- Classificador: FP32, até 40 épocas (5 somente cabeças); lote CUDA 16 / CPU 4; AdamW; paciência 8 na fase completa.",
              "- OOM e valores não finitos interrompem; não há redução silenciosa de lote, troca de precisão ou recuperação automática.",
              "- E2 não é carregada pelos treinadores. Pesos resultantes são candidatos, sem promoção automática.", "",
              "## Resultados de validação E1", ""]
    for component, result in results.items():
        lines += [f"### {component}", "", f"Época selecionada: **{result['selected_epoch']}**. Parada: `{result['stop_reason']}`.", "",
                  f"Duração total do componente: **{duration(result['component_wall_seconds'])}**; preparação: {result['setup_seconds']:.2f} s; "
                  f"validação final: {result['final_validation_seconds']:.2f} s.", "",
                  "```json", json.dumps(result["final_validation_metrics"], indent=2), "```", ""]
    lines += ["## Limitações", "",
              "As métricas acima são de validação usada para seleção e não substituem a avaliação final E2. "
              "As sete combinações sem supervisão conjunta no treino, 22 sem suporte conjunto na validação e o baixo suporte de d4 "
              "permanecem limitações aprovadas. Não foi feita calibração de confiança.", "",
              "A aprovação desta execução permite treinar e apresentar resultados. A integração da Etapa 5 depende de novo sinal explícito do usuário.", ""]
    if error:
        lines += ["## Interrupção registrada", "", error, ""]
    (report / "RELATORIO_TREINAMENTO.md").write_text("\n".join(lines), encoding="utf-8")
    return summaries


def plot_curves(report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [json.loads(line) for line in (report / "epochs.jsonl").read_text(encoding="utf-8").splitlines()]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for component in ("detector", "classifier"):
        selected = [row for row in rows if row["component"] == component]
        axes[0].plot([r["epoch"] for r in selected], [r["epoch_wall_seconds"] for r in selected], label=component)
    detector = [r for r in rows if r["component"] == "detector"]
    classifier = [r for r in rows if r["component"] == "classifier"]
    axes[1].plot([r["epoch"] for r in detector], [r["metrics"]["metrics/mAP50-95(B)"] for r in detector])
    axes[2].plot([r["epoch"] for r in classifier], [r["metrics"]["joint_accuracy"] for r in classifier])
    for ax, title, ylabel in zip(axes, ("Tempo por época", "Detector — validação E1", "Classificador — validação E1"),
                                ("Segundos", "AP50:95", "Acerto conjunto")):
        ax.set(title=title, xlabel="Época", ylabel=ylabel)
        ax.grid(alpha=.2)
    axes[0].legend()
    fig.savefig(report / "curves.png", dpi=160)
    plt.close(fig)


def append_comparison(directory, metadata, results, elapsed, status):
    for component, result in results.items():
        append_csv(directory / "TRAINING_COMPARISON.csv", {
            "run_id": metadata["run_id"], "component": component, "status": status,
            "started_at_utc": metadata["started_at_utc"], "device": metadata["device"]["resolved"],
            "gpu": metadata["hardware"].get("gpu", {}).get("name"), "dataset": metadata["dataset_version"],
            "dataset_checksum_index": metadata["dataset_checksum_index"], "recipe_sha256": metadata["recipe_hashes"][component],
            "batch_size": result["batch_size"], "epochs_completed": result["epochs_completed"],
            "selected_epoch": result["selected_epoch"], "stop_reason": result["stop_reason"],
            "training_seconds": result["training_seconds"], "validation_seconds": result["validation_seconds"],
            "component_wall_seconds": result["component_wall_seconds"], "run_wall_seconds": elapsed,
            "torch": metadata["versions"]["torch"], "ultralytics": metadata["versions"]["ultralytics"],
        })
