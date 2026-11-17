# Etapa 4 — treinamento E1 e tempos de referência

**Execução:** `v1-e1-20261117T174737Z`. **Estado:** `failed`.

Início em UTC: `2026-11-17T17:47:54.157341+00:00`. Dispositivo: `cuda:0`.

## Tempos medidos

| Componente | Épocas concluídas | Treino | Validação por época | Total das épocas | Mediana por época |
|---|---:|---:|---:|---:|---:|
| Preparação / primeira época em andamento | 0 | — | — | — | — |

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

## Limitações

As métricas acima são de validação usada para seleção e não substituem a avaliação final E2. As sete combinações sem supervisão conjunta no treino, 22 sem suporte conjunto na validação e o baixo suporte de d4 permanecem limitações aprovadas. Não foi feita calibração de confiança.

A aprovação desta execução permite treinar e apresentar resultados. A integração da Etapa 5 depende de novo sinal explícito do usuário.

## Interrupção registrada

PermissionError: [WinError 5] Acesso negado
