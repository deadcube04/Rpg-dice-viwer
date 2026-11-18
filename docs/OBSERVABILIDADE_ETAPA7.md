# Etapa 7 — Documentação, entrega e validação final

Escopo: consolidar README/guia operacional, cálculos/interpretação, evidências
versionáveis e execução da instrumentação com o pacote científico intacto.
Sem treinamento, retreino, pesos novos, dashboards, alertas ou commits.

## Entregáveis

- README, índice e guia de execução apontam para OBSERVABILIDADE.md.
- Guia cobre latências, histogramas, boxplots, taxas, Prometheus, dados para
  KS/PSI, privacidade, retenção, carga, testes e limitações.
- tools/observe.py e deploy/observability/compose.yaml oferecem uma execução
  repetível com a release aprovada, overlay somente leitura e CPU padrão.
- Receitas observe/observe-status/observe-logs/observe-stop no Just.
- Testes verificam a recusa de módulo científico modificado/ausente ou código
  não aprovado no inicializador raiz do pacote.
- docs/evidence/observability contém snapshots JSONL, cliente e Prometheus,
  imagens sintéticas, gráfico e hashes; esses arquivos são versionáveis.

## Preservação do modelo

A release original não foi editada ou reempacotada. O launcher verifica a
release e os módulos científicos do overlay; a inicialização continua
verificando aprovações, pacote, pesos, versões e dependências científicas.
A aplicação instrumentada tem fontes diferentes, sem reivindicar aprovação
científica nova. A V1 original permanece disponível pelo launcher anterior.

O projeto de conferência dice-observability foi criado nas portas 25000–25002.
Build da imagem aprovada passou com cache, API/MinIO iniciaram, o overlay foi
aplicado e a API ficou healthy em CPU. Foi testada parada seguida de retomada,
preservando o diretório de logs e os volumes. As implantações anteriores não
foram removidas. O histórico de desenvolvimento em runtime/observability-stage2
permanece no local anterior; a entrega contém uma captura separada.

## Validação manual final — PowerShell

Validação automática final realizada: 60 testes em Windows (59 passaram,
um symlink pulado) e os mesmos 60 testes em contêiner Linux, todos passando,
inclusive integração real sequencial/concorrente e symlink. A execução Linux
reutilizou a imagem pronta, sem dependências ML instaladas no host adicional.
A revisão independente do launcher/documentação não encontrou bloqueios;
a auditoria das capturas não encontrou campos privados, imagens ou exceções
brutas. O gráfico foi renderizado e inspecionado visualmente.

As evidências incluem 91 eventos request e 33 die no histórico de desenvolvimento,
mais o histórico do projeto final e três cargas. A carga final tem 20 medições
sintéticas, média 159,83 ms, mediana 146,32 ms e p95 nearest-rank 197,18 ms,
com 20 respostas 404/no_candidates e zero falhas de cliente. Esses números
são apenas evidência funcional do ambiente observado, não meta de desempenho.

As capturas foram conferidas quanto a esquema, privacidade, IDs, resumos e
versionabilidade. O manifesto SHA-256 cobre evidências e fontes selecionados;
finais LF foram definidos para JSONL/Prometheus para manter hashes em clones.
A atualização inclui README, docs/README.md, docs/EXECUCAO.md, o guia principal,
este relatório, evidências, .gitattributes, justfile, tools/observe.py,
deploy/observability/compose.yaml e tests/test_observe_launcher.py.

```powershell
Set-Location C:\MLops-dice-viwer\dice-viwer
just observe dice-observability 25000
just health 25000
just observe-status dice-observability
just load docs/evidence/observability/input-manifest.json 20 42 1 25000
curl.exe --include --max-time 10 http://127.0.0.1:25000/metrics
```

Esperado: inicialização saudável em CPU, carga com 20 tentativas medidas e um
warmup, HTTP 404/no_candidates para os exemplos sintéticos e métricas 200.
O primeiro comando pode fazer download/build e levar vários minutos no clone.
Não usar portas já ocupadas; preserve o mesmo projeto e porta nas chamadas.

```powershell
Get-Content '.\runtime\portable\dice-observability\observability\recognition.jsonl' -Tail 3
Get-Content '.\docs\evidence\observability\recognition.jsonl' -TotalCount 1
git ls-files --others --exclude-standard -- docs/evidence/observability
git diff --check
```

Esperado: novas linhas no runtime, amostra histórica legível na captura da
entrega e arquivos de evidência aparecendo como versionáveis enquanto não
adicionados pelo usuário. git diff --check deve ficar sem saída.
O JSONL ativo não foi retirado de runtime e o diretório de credenciais não
foi liberado pelo gitignore. Não houve git add/commit/push/merge.

O guia completo contém os testes locais/live e exemplos executáveis para
calcular estatísticas. A cópia entregue é fixa: executar carga não altera os
arquivos em docs/evidence. Prepare outra captura explicitamente se desejar
atualizar a evidência, preservando as definições e o manifesto de hashes.

## Limites de interpretação

Os exemplos sintéticos não avaliam acurácia ou drift, nem representam um
benchmark de produção. Resultados sob concorrência incluem rejeições 429/503
legítimas da capacidade atual. Histogramas de erros não medem inferência bem-
sucedida. A seed fixa entradas; tempos, IDs e escalonamento variam.
Retenção operacional pode remover histórico; a captura de entrega não expira.
As demais limitações técnicas do guia permanecem aplicáveis.

A aprovação manual final do usuário permanece pendente; a validação automática
não a substitui. Não executar novas etapas, commit ou publicação sem autorização.
