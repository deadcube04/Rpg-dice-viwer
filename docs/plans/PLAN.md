# Implementação V1 — auditoria, treinamento e serviço local

> **Andamento em 17/11/2026:** `v1-e1-20261117T175200Z` aprovado como **modelo inicial**, com limitações E2 e **retreino posterior manual** previsto. [Aprovação vinculada aos pesos](../evidence/approvals/model-initial-v1-20261117.json). Pesos, resultados e tempos iniciais preservados para comparação. O usuário [aprovou a Etapa 5](../evidence/approvals/stage5-flow-20261117.json); a **Etapa 7 está concluída e confirmada pelo usuário**, com pacote Docker `v1-initial-20261117-r3` saudável em CPU na porta 23000, CUDA e rollback conferidos. [Relatório final](../reports/stage7-20261117/RELATORIO_ETAPA7.md) e [operação](../reports/stage7-20261117/OPERACAO.md). Plano V1 concluído, com [confirmação final](../evidence/approvals/v1-final-acceptance-20261117.json) registrada. Os registros anteriores abaixo são históricos.

## 1. Objetivo e regras de execução

Implementar **YOLO26n Detect → recortes → ResNet18 multihead → BentoML**, com MinIO e funcionamento em CPU ou NVIDIA/CUDA, seguindo o documento de arquitetura.

**Cada etapa termina com uma parada obrigatória.** A próxima só começa após seu sinal explícito. Concluir tarefas, salvar revisões ou exportar arquivos não significa aprovação.

A primeira etapa inclui adaptar a ferramenta de auditoria e permitir sua revisão manual de **todas as 6.915 imagens**. Nenhum treinamento, preparação de splits definitivos ou implementação das etapas seguintes começa enquanto você não terminar essa revisão e aprovar.

Durante a execução:

- Atualizar a arquitetura com decisões, justificativas, referências e aprovações.
- Criar `PLANO_IMPLEMENTACAO_V1.md` ao lado da arquitetura, contendo este plano e o acompanhamento das etapas.
- Antes de implementar cada etapa, conferir suas decisões contra as seções aplicáveis da arquitetura. Divergências exigem resolução documental antes do código.
- Registrar entregas e aprovações vinculadas à versão dos dados, configurações ou modelo. Alterações posteriores invalidam as aprovações afetadas.
- Preservar originais, revisões anteriores e artefatos legados; não reutilizar pesos antigos como candidatos da nova arquitetura.
- Não criar, alterar ou executar testes automatizados. Usar análise estática, compilação, build e conferência manual.

## 2. Estrutura e aproveitamento da base

Manter o projeto em `C:/MLops-dice-viwer/dice-viwer` e o pacote Python `dice_viewer`, separando responsabilidades:

```text
config/
  dataset.yaml
  detector.yaml
  classifier.yaml
  preprocessing.yaml
  acceptance.yaml
  serving.yaml

audit/
  audit-ui/src/
    app/
    components/
    features/review/
    features/queue/
    lib/
  audit-back/
    src/dice_audit/    # API, SQLite, importadores e inventário
    pyproject.toml
    uv.lock
    runtime/
  skills/review-dice-images/
  start-audit.cmd
  README.md

src/dice_viewer/
  data/           # Importadores, inventário, agrupamentos e exportação
  models/         # YOLO26n e ResNet18 multihead
  training/       # Execuções e seleção de checkpoints
  inference/      # Imagem, recortes, scores, política e pipeline
  evaluation/     # Pareamento, métricas e relatórios
  serving/        # BentoML, contratos e armazenamento

data/
  annotations/
  manifests/
  derived/

audit/audit-back/runtime/
artifacts/<candidate_id>/
reports/<execution_id>/
deployment/
service.py
```

A organização é uma proposta técnica a registrar antes da implementação. Cada módulo terá uma responsabilidade; configurações e mapas serão compartilhados entre treino, avaliação e inferência.

**Aproveitar após adequação:** React/Vite/TypeScript, TanStack Query, editor visual, conceitos de snapshots e hashes, persistência S3 e configuração local de MinIO.

**Substituir os comportamentos incompatíveis encontrados:**

- YOLO11n por YOLO26n.
- MobileNet/EfficientNet por ResNet18.
- Cabeça de 20 valores e condicionamento pelo tipo por duas cabeças independentes sobre o backbone compartilhado.
- Exclusão de d4 e ausência de `d10_tens`.
- Confiança de anotação por imagem por máscaras por instância/tarefa.
- Splits fixados por câmera/data por agrupamentos auditados.
- Seleção automática de limiares pelos três limiares confirmados.
- Reserva obrigatória de GPU por configuração CPU padrão e CUDA opcional.
- Contrato e promoção legados pelo contrato essencial e aprovação manual.

## 3. Etapas e critérios para avançar

### Etapa 1 — ferramenta e revisão completa do dataset

**Referência:** seções 4, 11, 12 e decisões C20/C29–C31/C38.

1. Inventariar o ZIP: caminhos, hashes, dimensões, origem, anotações disponíveis e arquivos ausentes ou ilegíveis.
2. Usar o ZIP como referência imutável. Aproveitar os diretórios extraídos somente após verificar correspondência com ele.
3. Criar importadores separados para capturas, sidecars e cada fonte Roboflow.
4. Aplicar precedência: nova correção manual → `labels` finais/manifesto revisado → demais dados como histórico ou proposta não confirmada.
5. Incorporar os 11 sidecars revisados adicionais. Manter a imagem sem revisão final identificada como pendente.
6. Diferenciar caixas de corpo, numeral e polígonos. Caixas de numeral não entram automaticamente como caixas do detector.
7. Carregar todas as imagens na fila, inclusive duplicatas, negativas, cenas acima de oito dados e imagens sem rótulo.

**Adaptações da interface:**

- Imagem ampla com zoom, deslocamento, tela cheia e recorte ampliado do objeto selecionado.
- Criar, mover, redimensionar e remover caixas; mostrar ou ocultar sobreposições.
- Editar tipo e **token textual**, preservando `0`, `00` e `10`.
- Exibir separadamente o valor normalizado: `0`/`00` → 10.
- Permitir tipo desconhecido, face ilegível, ambiguidade, oclusão e corte de borda por instância.
- Registrar d6 por pontos/numerais e convenções de d4/d10.
- Confirmar ausência de dados ou excluir uma imagem com motivo, sem apagar o original.
- Filtros por origem, estado, tipo, contagem e problemas; busca por identificador.
- “Salvar” e “Salvar e próxima”, atalhos documentados e proteção contra perda de alterações.
- Progresso global e por fonte; retomada da última imagem.
- Duplicatas sugeridas para comparação, sem confirmação automática em lote.

Usar SQLite local para revisões transacionais, histórico e controle de versão. A auditoria funcionará sem modelos e sem MinIO.

O frontend seguirá React/Vite, TypeScript estrito, alias `@`, componentes próprios sobre Ant Design, Tailwind e soluções TanStack aplicáveis. A geometria das caixas será desenhada com SVG/canvas, separada dos estilos da interface.

**Contrato interno da auditoria:** identificadores estáveis de imagem/instância, origem, caixas canônicas em pixels após EXIF, tokens, máscaras por tarefa, decisão da imagem e revisão. Respostas externas terão validação em runtime; gravações com versão desatualizada retornarão conflito.

**Conclusão:** todas as entradas devem estar explicitamente revisadas ou excluídas com motivo. Rótulos antigos confiáveis não contam como sua nova revisão.

**Parada obrigatória:** entregar a ferramenta utilizável e aguardar sua conferência completa e seu sinal. Nenhuma etapa posterior será iniciada.

### Etapa 2 — dataset aprovado, agrupamentos e cobertura

**Referência:** seções 11–14.

1. Congelar um snapshot das revisões aprovadas, com hashes e proveniência.
2. Identificar duplicatas exatas, possíveis quase duplicatas, sessões e famílias de aumentos.
3. Preservar todos os derivados no mesmo agrupamento e partição.
4. Separar:
   - E1: imagens individuais para treino/validação.
   - E2: cenas independentes com dois a oito dados para avaliação final.
   - Negativas, excluídas e fora do escopo em conjuntos identificados.
5. Propor E1 com divisão agrupada 80%/20%, seed 42. Priorizar integridade dos grupos e apresentar as proporções efetivamente obtidas.
6. No baseline, não usar recortes de cenas de grupo no treino.
7. Gerar formatos YOLO e recortes somente depois de definir as partições.
8. Relatar cobertura por tipo × token, modalidade, origem e contagem. Classes sem supervisão não serão apresentadas como aprendidas.

**Conclusão:** manifesto canônico, splits, mapas, exclusões e relatório de cobertura inspecionados. Sem sobreposição conhecida entre partições.

**Parada:** aguardar sua aprovação do dataset preparado e das limitações de cobertura.

### Etapa 3 — ambiente e componentes do modelo

**Referência:** seções 7, 9, 14, 15 e REQ-HW-01/02.

1. Verificar hardware real, driver, memória e disponibilidade CUDA.
2. Consultar Context7, verificar compatibilidade e fixar versões exatas no lockfile. As versões existentes são candidatas, não versões já validadas.
3. Fixar origem/hash de `yolo26n.pt` e dos pesos ResNet18 `IMAGENET1K_V1`.
4. Implementar detector de classe única `die`.
5. Implementar ResNet18 compartilhada com cabeça de tipo e cabeça global de tokens; dimensões derivadas dos mapas.
6. Aplicar cross-entropy com máscaras independentes. Lotes sem rótulo de uma tarefa não produzirão NaN.
7. Implementar recortes com proporção preservada, padding e normalização versionados.

**Defaults propostos:** detector 640; recortes 224 × 224; margem de 10%; FP32; seed 42; scores brutos sem calibração; inferência YOLO26 em modo sem NMS, explicitamente configurado conforme a API da versão fixada.

A documentação oficial confirma o checkpoint [YOLO26n](https://docs.ultralytics.com/models/yolo26/) e os pesos [ResNet18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html). A integração será conferida na versão instalada.

**Conclusão:** ambiente reproduzível e componentes carregáveis, com dimensões e máscaras conferidas manualmente.

**Parada:** apresentar ambiente e receita completa antes de executar treinamento.

### Etapa 4 — treinamento E1: dados individuais

**Referência:** seções 15–16.

Executar detector e classificador separadamente, usando somente as partições aprovadas.

**Receita inicial proposta, registrada antes da execução:**

| Parâmetro | YOLO26n | ResNet18 multihead |
|---|---|---|
| Orçamento | Até 100 épocas | Até 40 épocas |
| Parada | 15 épocas sem melhora | 8 épocas sem melhora na fase de ajuste completo |
| Otimizador | SGD, LR 0,01, momentum 0,937, weight decay 0,0005 | AdamW, weight decay 0,0001 |
| Ajuste | Fine-tuning do detector | 5 épocas de cabeças com backbone congelado; depois ajuste completo |
| LR do classificador | — | Cabeças 0,001; backbone 0,0001 |
| Batch inicial | CUDA 8; CPU 2 | CUDA 16; CPU 4 |
| Seleção | AP50–95 de validação | Acerto conjunto tipo+token; desempate por perda de validação e época mais antiga |

- Pesos das perdas: tipo 1 e valor 1.
- Desativar mosaic, mixup, cutmix e copy-paste.
- Sem espelhamentos ou rotações aleatórias no primeiro baseline; usar apenas pequenas variações fotométricas.
- No congelamento do backbone, manter suas estatísticas de BatchNorm congeladas.
- Falta de memória interrompe a execução; reduzir lote requer nova configuração registrada, sem mudança silenciosa.
- Registrar curvas, duração, dispositivo, configurações, checkpoints e hashes.

**Conclusão:** candidato e relatório real de validação, incluindo d4, dezenas e d6 por pontos; sem metas de acurácia inventadas.

**Parada:** apresentar resultados e aguardar seu sinal para integração.

### Etapa 5 — cascata e BentoML para validação local

**Referência:** seções 8, 10, 17–19 e 22.

1. Compor detector, recortes, multihead e política.
2. Preservar associação entre candidatos e resultados.
3. Aplicar `>= 0.95/0.80/0.80`, sem arredondamento prévio.
4. Marcar saídas incompatíveis para revisão, sem substituir predições silenciosamente.
5. Implementar exatamente `POST /v1/recognize`, com JPEG/PNG binário e campos da seção 17.
6. Validar imagem → persistir original no MinIO → inferir → responder.
7. Implementar limites, erros, timeout e inicialização CPU-only.
8. Manter candidato identificado para validação, sem promoção automática.

Defaults: 64 candidatos técnicos, lote de até oito recortes, filtro 0,25, um worker, concorrência 1, timeout de processamento 30 s. Excesso de oito candidatos retorna erro, sem truncamento silencioso.

Conferir entrada binária, resposta e códigos HTTP contra o OpenAPI e a documentação de [tipos do BentoML](https://docs.bentoml.org/en/latest/build-with-bentoml/iotypes.html).

**Conclusão:** requisições manuais demonstráveis, inclusive revisão, ausência de candidatos, excesso e falhas de armazenamento.

**Parada:** aguardar sua aprovação do fluxo integrado.

### Etapa 6 — avaliação E2: grupos de dois a oito

**Referência:** seções 20–21.

1. Congelar pesos, mapas, pré-processamento e política.
2. Avaliar exclusivamente cenas E2 reservadas, sem ajustar parâmetros pelos resultados.
3. Usar pareamento espacial um a um com IoU ≥ 0,50, maximizando quantidade de pares e depois soma de IoU.
4. Produzir métricas de detecção, tipo, token, valor normalizado, reconhecimento conjunto e cena completa.
5. Medir revisão, precisão entre aceitos, cobertura correta automática, omissões e falsos positivos.
6. Estratificar por contagem, tipo e modalidade; mostrar suporte, exclusões e exemplos de erro.
7. Medir CPU e CUDA quando disponível. Cenários não executados serão identificados.

**Conclusão:** relatório vinculado ao candidato, sem reclassificar E2 como validação nem promover pesos.

**Parada:** aguardar sua aprovação ou rejeição explícita do modelo.

### Etapa 7 — subida local da versão aprovada

**Referência:** seções 21–22 e C26/C27.

1. Empacotar bundle imutável com pesos, mapas, transformações, política e ambiente.
2. Configurar Docker Compose CPU padrão e override NVIDIA/CUDA.
3. Subir BentoML e MinIO em interfaces locais, com volumes persistentes e bucket privado sem expiração automática.
4. Carregar somente o bundle aprovado, verificar readiness e realizar conferência manual.
5. Registrar operação, tempos medidos e procedimento de retorno ao bundle anterior.

**Conclusão:** serviço local atendendo com a versão aprovada e imagens persistidas.

**Parada:** aguardar sua confirmação da entrega final.

## 4. Validação e limitações registradas no plano inicial

Em cada entrega, usar build/typecheck/lint quando aplicáveis, análise estática Python e conferência manual. Na auditoria, verificar especialmente persistência após reinício, conflitos de gravação, navegação sem perda, coordenadas com zoom/EXIF e distinção `0`/`00`/`10`.

As avaliações estatísticas de E1/E2 pertencem ao pipeline científico solicitado; não implicam criação de suites de testes de software.

O Context7 foi localizado inicialmente, mas sua ferramenta ficou indisponível na tentativa de consulta. A execução deverá tentar novamente antes de fixar dependências e registrar a consulta efetivamente realizada. As referências oficiais acima serviram de apoio provisório.

O hardware não pôde ser consultado porque a criação de processos está bloqueada nesta sessão. Nenhum arquivo foi criado ou alterado, nenhum treino foi iniciado e nenhuma qualidade dos rótulos foi presumida.

### Modularização do audit — 14/10/2026

A auditoria foi extraída para o subprojeto audit, contendo audit-ui e audit-back. Backend usa o pacote independente dice_audit, pyproject/uv.lock/.venv próprios e apenas Pillow; frontend conserva package.json/bun.lock próprios. Launcher, documentação, skill e runtime foram centralizados em audit. Não importar código da auditoria pelo pacote dice_viewer.

SQLite e caches migrados para audit/audit-back/runtime. Backup antes da movimentação e antes da atualização de cache_path; digest das anotações/histórico conferido e preservado. O estado existente de 6.915 imagens, uma revisão concluída e duas versões de histórico permaneceu igual. Snapshots históricos não foram alterados. Portas e contrato /api preservados. Guia atual: dice-viwer/audit/README.md; launcher: dice-viwer/audit/start-audit.cmd. Nenhuma revisão nova, split, treinamento ou etapa posterior foi iniciado.
