# Etapa 4 — treinamento E1 e tempos de referência

**Execução:** `v1-e1-20261117T175200Z`. **Estado:** `completed`.

Início em UTC: `2026-11-17T17:49:41.292090+00:00`. Dispositivo: `cuda:0`.

## Tempos medidos

| Componente | Épocas concluídas | Treino | Validação por época | Total das épocas | Mediana por época |
|---|---:|---:|---:|---:|---:|
| detector | 27 | 1.003 h (3609.48 s) | 0.128 h (460.68 s) | 1.138 h (4095.44 s) | 150.67 s |
| classifier | 28 | 0.218 h (785.51 s) | 0.031 h (109.95 s) | 0.250 h (901.24 s) | 32.52 s |

Os valores são medições de relógio de parede, em segundos, por `time.perf_counter`, com sincronização CUDA nas fronteiras das fases. O treino inclui leitura de dados, aumentos, transferências, forward, backward e passos de otimizador. Validação e serialização são registradas separadamente. Não são tempos exclusivos de kernels GPU.

`timings.csv` contém uma linha por época; `epochs.jsonl` inclui métricas, fase e timestamps. `progress.json` registra o andamento entre épocas. `timing-summary.json` oferece médias, medianas, extremos e primeira época.

Os totais de componente também incluem montagem/carregamento, validação final e serialização. O total da execução inclui ainda a conferência de integridade, preparação das entradas e relatórios. A marca inicial ocorre depois da importação das bibliotecas e seleção do dispositivo; a gravação final de status e resumo ocorre após a última marca. Compare execuções com o mesmo escopo e considere cache do sistema, temperatura, outros processos e parada antecipada.

## Configuração e proveniência

- Dataset: `v1-20261117-7b23477513de`; índice SHA-256 `5ecbe03114abb68790e2319e6e7b26f745c422735d48895a86d2761b6c0d1a73`.
- Hardware, versões, dispositivo, número de threads, configurações e estados iniciais estão em `run.json` e no snapshot `provenance/`.
- Detector: FP32, até 100 épocas, lote CUDA 8 / CPU 2, SGD, paciência 15; seleção por AP50:95 em validação E1.
- Classificador: FP32, até 40 épocas (5 somente cabeças); lote CUDA 16 / CPU 4; AdamW; paciência 8 na fase completa.
- OOM e valores não finitos interrompem; não há redução silenciosa de lote, troca de precisão ou recuperação automática.
- E2 não é carregada pelos treinadores. Pesos resultantes são candidatos, sem promoção automática.

## Resultados de validação E1

### detector

Época selecionada: **12**. Parada: `early_stopping`.

Duração total do componente: **1.149 h (4135.80 s)**; preparação: 18.75 s; validação final: 16.18 s.

```json
{
  "metrics/precision(B)": 0.9677800523694635,
  "metrics/recall(B)": 0.960925703098813,
  "metrics/mAP50(B)": 0.9759838441670202,
  "metrics/mAP50-95(B)": 0.7908866373844156
}
```

### classifier

Época selecionada: **20**. Parada: `early_stopping`.

Duração total do componente: **0.252 h (906.21 s)**; preparação: 0.38 s; validação final: 4.46 s.

```json
{
  "type_loss": 0.23529557344276172,
  "value_loss": 3.6432600921055056,
  "type_accuracy": 0.9565217391304348,
  "token_accuracy": 0.4412997903563941,
  "normalized_value_accuracy": 0.44234800838574423,
  "joint_accuracy": 0.42348008385744235,
  "joint_normalized_accuracy": 0.42452830188679247,
  "type_support": 966,
  "value_support": 954,
  "joint_support": 954,
  "loss": 3.878555665548267
}
```

## Limitações

As métricas acima são de validação usada para seleção e não substituem a avaliação final E2. As sete combinações sem supervisão conjunta no treino, 22 sem suporte conjunto na validação e o baixo suporte de d4 permanecem limitações aprovadas. Não foi feita calibração de confiança.

A aprovação desta execução permite treinar e apresentar resultados. A integração da Etapa 5 depende de novo sinal explícito do usuário.
