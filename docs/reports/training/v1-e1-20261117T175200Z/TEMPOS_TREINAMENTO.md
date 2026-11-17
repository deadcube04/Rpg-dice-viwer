# Referência de tempo de treinamento — V1 E1

**Execução concluída:** `v1-e1-20261117T175200Z`. **Total medido: 01:24:54.24**.

Tempos formatados como `horas:minutos:segundos`; os valores numéricos de CSV e JSON estão em segundos.

Início UTC: `2026-11-17T17:49:41.292090+00:00`. Fim UTC: `2026-11-17T19:14:35.534543+00:00`.

## Tempos por componente

| Componente | Épocas | Selecionada | Só treino | Validação durante as épocas | Checkpoints | Total do componente |
|---|---:|---:|---:|---:|---:|---:|
| detector | 27 | 12 | 01:00:09.48 | 00:07:40.68 | 00:00:25.24 | 01:08:55.80 |
| classifier | 28 | 20 | 00:13:05.51 | 00:01:49.95 | 00:00:05.77 | 00:15:06.21 |

Preparação e finalização externas aos componentes: **52.23 s**. O total de cada componente inclui preparação, épocas, checkpoints e validação final; por isso ele é maior que a soma apenas de treino e validação durante as épocas.

## Tempos por fase

| Componente / fase | Épocas | Só treino | Validação | Total das épocas | Mediana por época |
|---|---:|---:|---:|---:|---:|
| detector / finetune | 27 | 01:00:09.48 | 00:07:40.68 | 01:08:15.44 | 150.67 s |
| classifier / heads | 5 | 00:02:05.61 | 00:00:22.88 | 00:02:29.66 | 29.48 s |
| classifier / full_finetune | 23 | 00:10:59.90 | 00:01:27.07 | 00:12:31.57 | 32.53 s |

`heads`: backbone e BatchNorm congelados. `full_finetune`: backbone descongelado na época 6, preservando o estado AdamW das cabeças. Critérios: paciência 15 no detector e 8 na fase completa do classificador. Motivos de parada: detector `early_stopping`; classificador `early_stopping`.

## Condições para comparação futura

- GPU: NVIDIA GeForce RTX 2060; CPU: Intel(R) Core(TM) i5-9400F CPU @ 2.90GHz; RAM: 15.95 GiB; threads PyTorch: 4.
- Plataforma: `Windows-10-10.0.26300-SP0`; dispositivo: `cuda:0`. Precisão, seed, resoluções e workers constam nas receitas preservadas.
- Lotes efetivos: detector 8; classificador 16.
- PyTorch 2.14.1+cu126, TorchVision 0.29.1+cu126 e Ultralytics 8.4.173.
- Dataset, agrupamentos e receitas vinculados a hashes em [run.json](../../../evidence/training/v1-e1-20261117T175200Z/run.json).
- Relógio monotônico com sincronização CUDA nas fronteiras; treino inclui leitura, aumentos, transferências e otimização.
- Não são tempos exclusivos de kernels GPU. Cache do sistema, temperatura e outros processos podem afetar a medição.
- Importações anteriores ao relógio, implementação/revisão do código e documentação posterior não entram no total medido.
- Tentativas anteriores que falharam são preservadas separadamente e não são somadas a esta referência; consulte `TENTATIVAS.md`.
- Compare também tempo por época e por fase: a parada antecipada pode produzir quantidades diferentes de épocas.

## Resultados de validação E1

Detector selecionado: AP50:95 **0.790887**, época 12.

Classificador selecionado: época 20; tipo **924/966 = 95.65%**; token **421/954 = 44.13%**; tipo+token **404/954 = 42.35%**. Os denominadores diferem pelas máscaras independentes.

| Tipo | Tipo correto | Token correto | Tipo + token corretos |
|---|---:|---:|---:|
| d10 | 53/62 (85.48%) | 43/61 (70.49%) | 41/61 (67.21%) |
| d10_tens | 472/478 (98.74%) | 290/472 (61.44%) | 290/472 (61.44%) |
| d12 | 20/23 (86.96%) | 10/23 (43.48%) | 9/23 (39.13%) |
| d20 | 353/364 (96.98%) | 53/359 (14.76%) | 49/359 (13.65%) |
| d4 | 2/3 (66.67%) | 3/3 (100.00%) | 2/3 (66.67%) |
| d6 | 19/26 (73.08%) | 15/26 (57.69%) | 10/26 (38.46%) |
| d8 | 5/10 (50.00%) | 7/10 (70.00%) | 3/10 (30.00%) |

Interprete as diferenças entre tipo e token com os suportes da tabela. D20: 53/359 tokens corretos. D4: suporte de tipo 3. D6 por pontos: 0/1 tokens corretos. Resultados com poucas amostras não demonstram generalização. As modalidades anotadas foram preservadas.

Estes resultados usam validação para seleção; **E2 não foi executada**. Permanecem as lacunas de cobertura já aprovadas. Os pesos são candidatos, sem aprovação para uso e sem calibração. A integração da Etapa 5 depende de novo sinal do usuário.

## Arquivos para reutilização

- [CSV por época](../../../evidence/training/v1-e1-20261117T175200Z/timings.csv) e [CSV por fase](../../../evidence/training/v1-e1-20261117T175200Z/PHASE_TIMES.csv).
- [Referência estruturada JSON](../../../evidence/training/v1-e1-20261117T175200Z/TIMING_REFERENCE.json) e [curvas de aprendizado e tempo](../../../evidence/training/v1-e1-20261117T175200Z/curves.png).
- [Comparação por componente entre execuções](../../../evidence/training/TRAINING_COMPARISON.csv): `run_wall_seconds` se repete nas duas linhas; **não somar esse campo**.
- [Uma linha por execução](../../../evidence/training/TRAINING_RUNS.csv), para comparar totais sem duplicação.
- [Tentativas e falha inicial](../TENTATIVAS.md).
- [Inspeção dos candidatos FP32 e hashes](../../../evidence/training/v1-e1-20261117T175200Z/candidate-inspection.json).
- [Relatório automático completo](RELATORIO_TREINAMENTO.md).

Para regenerar as tabelas de referência de uma execução concluída, use `python -m dice_viewer.training.summarize <run_id>` no ambiente do projeto. Os dados originais por época são preservados; esse comando não executa treinamento ou inferência.
