# Etapa 3 — ambiente, componentes e receita

> **Aprovação e execução posteriores em 17/11/2026:** o usuário aprovou esta receita e solicitou registrar os tempos para comparação futura. A Etapa 4 foi concluída em **1h24min54,24s**: [tempos e resultados](../training/v1-e1-20261117T175200Z/TEMPOS_TREINAMENTO.md). O texto abaixo preserva a entrega apresentada para aprovação; a Etapa 5 ainda não está autorizada.

**Data:** 17/11/2026. **Estado:** preparação concluída; receita aguardando aprovação para iniciar a Etapa 4. Nenhum treinamento foi iniciado.

## Ambiente reproduzível

| Item | Valor conferido |
|---|---|
| CPU | Intel Core i5-9400F, seis processadores lógicos |
| RAM | 15,95 GiB |
| GPU | NVIDIA GeForce RTX 2060, 6 GiB, compute capability 7.5 |
| Driver NVIDIA | 610.88 |
| Python / uv | 3.11.9 / 0.12.23 |
| PyTorch | 2.14.1+cu126 |
| TorchVision | 0.29.1+cu126 |
| Ultralytics | 8.4.173 |
| Runtime CUDA do PyTorch | 12.6 |

As dependências diretas estão fixadas no [pyproject.toml](../../../pyproject.toml) e as transitivas no [uv.lock](../../../uv.lock). O ambiente foi sincronizado; a conferência do lock, das dependências e a compilação Python terminaram sem erros.

Evidências: [hardware](../../evidence/stage3-20261117/hardware.json) e [ambiente](../../evidence/stage3-20261117/environment.json).

## Componentes e pesos

O detector usa YOLO26n com arquitetura de classe única `die`, inicializada com os tensores compatíveis dos pesos oficiais. O classificador usa ResNet18 compartilhada e duas cabeças lineares independentes: **7 tipos** e **29 tokens**, dimensionados pelos mapas do dataset aprovado. A cabeça de valor é global, sem condicionamento pelo tipo.

| Inicialização | SHA-256 |
|---|---|
| YOLO26n | `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef` |
| ResNet18 `IMAGENET1K_V1` | `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec` |

As origens oficiais, tamanhos e caminhos estão em [pretrained-weights.json](../../evidence/stage3-20261117/pretrained-weights.json). Os estados iniciais foram salvos em `artifacts/initial-components-v1`, em safetensors, e recarregados com verificação de hashes, arquitetura e mapas. São **inicializações**, sem aprovação para uso como modelos treinados.

Na API instalada, a inferência sem NMS é selecionada explicitamente por `head.end2end=True` e `predict(nms=False, quantize=32)`. A arquitetura usa `reg_max=1`, conforme YOLO26. As escolhas estão registradas na [configuração do detector](../../../config/detector.yaml).

## Máscaras e pré-processamento

A cross-entropy recebe logits e máscaras independentes por tarefa. Exemplos reais sem rótulo de tipo ou valor produziram perda zero apenas na tarefa ausente, sem NaN. O congelamento do backbone também mantém as estatísticas de BatchNorm congeladas na fase inicial.

O pré-processamento usa orientação EXIF antes das caixas e canais RGB. O detector usa resolução 640 e o letterbox da versão fixada. Para o classificador:

- Margem de 10% por lado, limitada à imagem.
- Proporção preservada, padding preto e interpolação bilinear para 224 × 224.
- Escala 0–1, média `[0.485, 0.456, 0.406]` e desvio `[0.229, 0.224, 0.225]`.
- Os recortes já preparados recebem somente normalização, sem nova aplicação de geometria.

A geometria recomposta de uma imagem E1 real apresentou diferença de pixel normalizado igual a zero em relação ao recorte da Etapa 2. Consulte [preprocessing.yaml](../../../config/preprocessing.yaml).

## Receita proposta para aprovação

Detector e classificador serão treinados separadamente, usando apenas E1 aprovado.

| Parâmetro | YOLO26n | ResNet18 multihead |
|---|---|---|
| Orçamento | Até 100 épocas | Até 40 épocas: 5 só das cabeças e até 35 completas |
| Parada antecipada | Paciência 15 | Paciência 8 na fase completa |
| Otimizador | SGD | AdamW |
| Learning rate | Inicial 0,01; fator final 0,01 | Cabeças 0,001; backbone 0,0001 |
| Outros parâmetros | Momentum 0,937; weight decay 0,0005; warmup de 3 épocas | Betas 0,9/0,999; epsilon 1e-8; weight decay 0,0001; scheduler constante |
| Lote CUDA / CPU | 8 / 2 | 16 / 4 |
| Seleção | AP50:95 de validação | Maior acerto conjunto tipo+token; desempate por menor perda e época mais antiga |
| Precisão, seed e workers | FP32, seed 42, workers 0 | FP32, seed 42, workers 0 |

Na época 6 do classificador, preservar o estado do otimizador das cabeças e adicionar o grupo do backbone. Os pesos das perdas são 1 para tipo e 1 para valor. O acerto conjunto considera somente instâncias com ambos os rótulos conhecidos.

O baseline não usa mosaic, mixup, cutmix, copy-paste, flips, rotações ou aumentos geométricos. Pequenas variações fotométricas serão aplicadas somente ao treino, com os limites registrados nas configurações. Falta de memória interrompe a execução; reduzir o lote exige uma nova configuração registrada.

Configurações completas: [detector.yaml](../../../config/detector.yaml), [classifier.yaml](../../../config/classifier.yaml), [dataset.yaml](../../../config/dataset.yaml) e [defaults da biblioteca fixada](../../evidence/stage3-20261117/ultralytics-defaults.json).

## Dataset e limitações aprovadas

O dataset preparado e suas limitações foram aprovados pelo usuário. A aprovação está vinculada ao índice SHA-256 `5ecbe03114abb68790e2319e6e7b26f745c422735d48895a86d2761b6c0d1a73` e registrada em [dataset-approval.json](../../evidence/stage3-20261117/dataset-approval.json).

| Conjunto | Imagens | Recortes do classificador |
|---|---:|---:|
| Treino E1 | 3.846 | 3.846 |
| Validação E1 | 969 | 966 |
| Avaliação final E2 | 518 | 1.927 |

E2 permanece reservada à avaliação final, sem uso para ajuste de pesos ou seleção de parâmetros. As lacunas aprovadas incluem sete combinações sem supervisão conjunta no treino, 22 na validação e baixo suporte de d4. A fonte `roboflow/d4-turbo-rad-v4` continua excluída; a quarentena está preservada.

Consulte o [relatório da Etapa 2](../preparation/RELATORIO_ETAPA2.md) para a distribuição completa, os motivos de quarentena e as limitações de independência.

## Verificações e seus limites

Foram executadas inspeções com amostras reais do treino E1 em CPU, CUDA e em processo separado com `CUDA_VISIBLE_DEVICES=-1`, que selecionou CPU por fallback. Não houve backward ou passo de otimizador. Nenhuma imagem E2 ou de validação foi usada na execução de modelos nesta etapa.

- [CPU](../../evidence/stage3-20261117/components-cpu.json).
- [CUDA](../../evidence/stage3-20261117/components-cuda.json).
- [Fallback com CUDA indisponível](../../evidence/stage3-20261117/components-cuda-unavailable.json).
- [Recarga dos estados iniciais salvos](../../evidence/stage3-20261117/components-reloaded-initial-states.json).
- [Revisão estática independente](../../evidence/stage3-20261117/code-review.json).
- [Hashes da entrega técnica anterior à documentação Markdown](../../evidence/stage3-20261117/delivery-checksums.json).

Não foram criados ou executados testes automatizados. A capacidade dos lotes de treinamento, a latência, a acurácia e a calibração ainda não foram medidas. A demonstração CPU usou Windows e a wheel CUDA com GPU ocultada; ela não valida wheel CPU-only, Linux ou o futuro serviço BentoML.

O Context7 foi tentado, mas falhou durante a inicialização MCP. Foram usadas a documentação oficial e a inspeção do código instalado, conforme [documentation.json](../../evidence/stage3-20261117/documentation.json). Referências: [YOLO26](https://docs.ultralytics.com/models/yolo26/), [cabeças de detecção](https://docs.ultralytics.com/reference/nn/modules/head/) e [ResNet18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html).

## Próximo marco

**Aguardar aprovação desta receita antes de implementar e executar o treinamento da Etapa 4**, conforme o [plano de implementação](../../plans/PLANO_IMPLEMENTACAO_V1.md). A autorização para gerar Markdown não constitui aprovação do treinamento.

O [relatório textual original](../../evidence/stage3-20261117/RELATORIO_ETAPA3.txt) foi preservado como registro da entrega anterior à autorização para gerar Markdown.
