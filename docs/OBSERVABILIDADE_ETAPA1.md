# ETAPA 1 — PREPARAÇÃO, INSPEÇÃO E BASELINE
Data: 2026-11-17 (America/Sao_Paulo)
Repositório: C:/MLops-dice-viwer/dice-viwer
Commit inspecionado: 3cdb86a3db7a9c554f90361d66f218f33160ec5e
Estado inicial do Git: limpo.
Escopo: somente inspeção, verificações e estratégia. Sem instrumentação.

## 1. MAPA DA APLICAÇÃO

recognition_service.py importa DiceRecognitionService de recognition/service.py.
service.py também aponta para essa classe, com alias DiceService.
POST /v1/recognize recebe JPEG/PNG binário direto, sem query, JSON ou multipart.
inputs.py gera request_id e valida cabeçalhos, corpo e limites: 10 MiB,
4096 por dimensão, 16 milhões de pixels, imagem estática, EXIF aplicado uma vez.
openapi.py contém middleware que também rejeita MIME inválido antes do descriptor.
Esse caminho precisa ser incluído na futura observabilidade.

contract.py define saída estrita: request_id, image_id, status, results,
dices_count, request_requires_review, thresholds, versions, timings_ms.
Cada dado inclui bbox, shape, die_type, face_value, três confianças,
auto_accepted e review_reasons. Até oito dados. Limiares: 0.95/0.8/0.8.
Resultados normais: 200 ok/needs_review; 404 no_candidates.
Erros: 400 invalid_request; 413 image_too_large; 415 unsupported_media_type;
422 invalid_image/too_many_dice/scene_too_complex; 500 internal_error;
503 storage_unavailable/model_unavailable; 504 inference_timeout.
Os códigos e envelopes acima não devem mudar.

service.py usa um worker, uma thread de inferência e lock; CPU é o padrão.
O prazo de processamento é 30 s; leitura do corpo admite até 45 s;
o timeout BentoML é 90 s. Após 504, o worker pode continuar até liberar o lock.
timings_ms.total começa no worker: não é a latência HTTP completa e não inclui
leitura do upload ou espera anterior. Decode/persist/detect/classify são etapas.

store.py exige MinIO/S3 existente para originais; head_bucket na inicialização,
put_object antes de inferir, bytes exatos com metadata de integridade/versão.
Isso é requisito atual da inferência, não uma dependência nova de observabilidade.
operations.py rejeita lifecycle inesperado; não há expiração de originais.
Não aplicar a futura retenção de logs a imagens ou aos artefatos científicos.

engine.py verifica aprovação, manifesto, protocolo, fontes congeladas, versões
e pesos antes de carregar FrozenCascade. Não instrumentar a cascata congelada.
operations.py registra startup/warmup em startup.json, sobrescrito por startup;
isso não constitui histórico append-only por requisição.
Há logging textual com request_id/image_id/status em parte dos caminhos,
warnings e traceback. Não há JSONL completo por requisição/dado, características
de entrada, retenção própria de logs ou gerador de carga dedicado identificado
no escopo de serving/implantação.

## 2. MÉTRICAS E IMPLANTAÇÃO

Dependências existentes: BentoML 1.4.39, prometheus-client 0.26.0, Pillow,
NumPy, Starlette e biblioteca padrão. metrics.enabled=False no serviço.
O código instalado do BentoML contém endpoint/exportador e instrumentação
HTTP de contagem, duração e requisições em andamento. Preferir sua reutilização;
confirmar exposição e cobertura reais de /metrics na Etapa 4, sem presumir
o comportamento HTTP atual enquanto a API estiver indisponível.
Não criar contadores HTTP equivalentes em paralelo sem necessidade demonstrada.

deploy/v1/compose.yaml tem MinIO, storage-init e API. API com filesystem
read-only, usuário sem privilégios, /tmp efêmero e volume service-state em /state.
Futura persistência de logs pode usar /state/observability, sem serviço externo.
tools/model.py restaura release v1-initial-20261117-r3 e usa Docker/Compose;
justfile fornece restore/up/health/recognize/status/logs/stop.
Porta padrão 23000; MinIO/console nas duas portas seguintes. CPU padrão.
runtime/v1 e runtime/portable/<projeto> são estados distintos: não misturar.

ATENÇÃO À ENTREGA FUTURA: o launcher constrói a release congelada, não src/.
Alterar src/ isoladamente não atualiza a execução portátil. Não editar releases,
dump, pesos ou aprovações congelados. Planejar camada/release de aplicação
instrumentada separada, com mesmos artefatos e verificações científicas,
sem reapresentar aprovação histórica de fluxo como aprovação do novo código.
Validar essa entrega na etapa de implantação; nenhuma release nova nesta etapa.

## 3. VERIFICAÇÕES EXECUTADAS

Python host: 3.11.9; uv: 0.12.23; Docker CLI e Just encontrados.
Ambiente .venv: bentoml 1.4.39, prometheus-client 0.26.0, Pillow 12.3.0,
numpy 2.4.6, pydantic 2.13.5, torch 2.14.1+cu126, starlette 1.7.0.
Wheel CUDA instalada não altera o padrão CPU do serviço.
AST: 45 arquivos Python de src/, tools/, deploy/v1/ e entrypoints válidos.
python tools/model.py restore: passou, verificou release já existente (126 arquivos).
SHA-256 e tamanho das três partes do dump: passaram contra manifest.json.
python tools/model.py --help e just --list: passaram.
git diff --check: passou antes do registro deste documento; repetir após escrita.
Nenhum arquivo de teste/configuração pytest versionado identificado.
.venv/Scripts/python.exe -m unittest discover -s src -p test*.py:
Ran 0 tests, OK. Isso não representa cobertura ou baseline funcional aprovado.
pytest não está instalado; nenhuma instalação ou suite nova nesta etapa.
O pedido atual autoriza testes de instrumentação nas etapas futuras, apesar
da orientação anterior no-tests; essa autorização não antecipa sua implementação.

LIMITAÇÃO DE AMBIENTE: docker version/docker ps falharam porque o pipe
dockerDesktopLinuxEngine não existe. Daemon Linux indisponível.
GET /readyz nas portas 23000, 24000 e 13000: conexão indisponível.
Não foi possível medir latência HTTP, executar inferência real, verificar S3
ao vivo ou confirmar resposta atual de /metrics. Não há baseline numérico
de desempenho desta sessão. Build/downloads não foram iniciados.

## 4. ESTRATÉGIA PROPOSTA PARA AS ETAPAS FUTURAS — NÃO IMPLEMENTADA

Etapa 2: módulo de observabilidade separado; schema_version, UTC, IDs gerados
pelo servidor, resultado HTTP/código seguro, versões/dispositivo, metadados
permitidos de entrada, tempos e decisões. Eventos request e die relacionados
por request_id e ID do dado. Uma conclusão HTTP por requisição, inclusive
rejeições de middleware, entrada inválida, lock ocupado e timeout.
Conclusão tardia do worker deve ser evento distinto, nunca outra requisição.
Não logar corpo, imagem/base64, EXIF, cabeçalhos arbitrários, URL S3,
credenciais, tokens, nomes de arquivos do cliente ou exceções não sanitizadas.
Falha de telemetria não deve trocar HTTP/resultado nem interromper inferência;
expor a perda por diagnóstico seguro e, quando disponível, contador próprio.

Etapa 3: aproveitar RGB já decodificado sem mudar pixels usados pelo modelo.
Registrar bytes, largura, altura, pixels, proporção, estatísticas numéricas de
brilho/contraste/canais e nitidez com definição e versão explícitas.
Usar cópia/amostra limitada para conter custo; registrar ausência em rejeições.
Não registrar embeddings ou imagem; não interpretar confiança como calibração.
Guardar valores individuais contínuos para histogramas, boxplots, KS e PSI;
definir amostras de referência/janelas, faltantes e bins fixos posteriormente.
Sem cálculo de drift, dashboards ou alertas nesta entrega.

Etapa 4: habilitar e conferir Prometheus nativo em GET /metrics; adicionar
somente métricas do domínio/telemetria ausentes. Labels de cardinalidade
limitada; IDs, bbox, valores contínuos e mensagens nunca serão labels.
Separar duração HTTP, tempo no worker e tempo de cada estágio.

Etapa 5: arquivo JSONL local em append, escrita sincronizada entre threads,
rotação com arquivos fechados imutáveis e limpeza somente de segmentos antigos.
Proposta inicial revisável: rotação diária UTC ou 10 MiB, retenção de 30 dias
e teto total de 1 GiB, configuráveis. Teto pode reduzir a janela efetiva.
Append-only refere-se à escrita no segmento: expiração exclui segmentos
inteiros, sem reescrever registros. Documentar perda em crash/disco cheio,
flush/fsync e suporte inicial a um processo; não prometer auditoria inviolável.

Etapa 6: cliente de carga em biblioteca padrão, manifesto ordenado e seed,
CPU e carga sequencial como padrão, concorrência explícita, timeout e
resultados por tentativa; não retentar silenciosamente. Não usar imagens
de teste para ajustar modelo. Separar cold start/warmup e medir pelo cliente.
Testes de instrumentação autorizados pelo pedido atual: caminhos de entrada,
respostas/erros, timeout/conclusão tardia, sanitização, falha de escrita,
rotação/retenção, contagem Prometheus e compatibilidade do contrato.

Etapa 7: documentar média/mediana/p95/p99, histogramas com unidades/bins,
boxplots com quartis/IQR, taxa 5xx e rejeições 4xx separadas. 404 no_candidates
é resultado de domínio; needs_review não é erro técnico. Definir denominadores.
Logs guardam amostras individuais; buckets Prometheus não substituem essas
amostras para boxplots/KS/PSI. Sem tráfego rotulado, não inferir acurácia/drift.
README e guia operacional serão atualizados na etapa correspondente.

## 5. VALIDAÇÃO MANUAL — WINDOWS POWERSHELL

Na raiz:

```powershell
Set-Location C:\MLops-dice-viwer\dice-viwer
Get-Content .\docs\OBSERVABILIDADE_ETAPA1.md
```

Esperado: este baseline e estratégia, sem código de instrumentação.


```powershell
python tools/model.py restore
```

Esperado: Approved model restored and verified; não inicia containers.
Com release já presente, valida seus arquivos, mas não relê as partes do dump.


```powershell
.\.venv\Scripts\python.exe -m unittest discover -s src -p 'test*.py'
```

Esperado no baseline: Ran 0 tests. Não significa que inferência foi testada.

Inicie Docker Desktop com engine Linux e aguarde disponibilidade. Depois:

```powershell
docker version --format '{{.Server.Version}}'
```

Esperado: versão do servidor; erro de pipe significa pré-requisito pendente.

Use projeto novo/porta livre para preservar implantações anteriores:

```powershell
python tools/model.py run --project dice-observabilidade-baseline --port 24000
```

Esperado: release verificada, build quando necessário, serviços iniciados e
healthcheck saudável em CPU. Primeiro build exige internet e espaço em disco.
Portas 24000, 24001 e 24002 devem estar livres. O comando gera estado/volumes.


```powershell
curl.exe --include --max-time 10 http://127.0.0.1:24000/readyz
```

Esperado: HTTP 200 quando pronto.


```powershell
curl.exe --include --max-time 10 -X POST http://127.0.0.1:24000/v1/recognize -H 'Content-Type: application/json' --data '{}'
```

Esperado: HTTP 415, error.code unsupported_media_type, IDs/envelope da V1.


```powershell
curl.exe --include --max-time 10 -X POST http://127.0.0.1:24000/v1/recognize -H 'Content-Type: image/png' --data-binary 'invalid'
```

Esperado: HTTP 422, error.code invalid_image.

Para uma imagem própria, substitua apenas o caminho:

```powershell
curl.exe --include --max-time 45 -X POST http://127.0.0.1:24000/v1/recognize -H 'Content-Type: image/png' --data-binary '@C:\imagens\dado.png'
```

Esperado: 200 ok/needs_review ou 404 no_candidates conforme a imagem;
image_id, results, versions e timings_ms no contrato atual. Cenas acima dos
limites podem retornar 422. Um resultado específico não é garantido.
A requisição válida armazena o original no MinIO existente, como já previsto.


```powershell
curl.exe --include --max-time 10 http://127.0.0.1:24000/metrics
```

Somente registrar resposta de baseline: métricas estão desativadas no serviço.
Exposição Prometheus ainda não é critério de aprovação da Etapa 1.


```powershell
just status dice-observabilidade-baseline
```

Esperado: API/MinIO ativos, API saudável; storage-init concluído com código 0.

```powershell
just stop dice-observabilidade-baseline
```

Esperado: serviços parados; volumes, originais e credenciais preservados.


```powershell
git diff --check
git status --short
```

Esperado: diff --check sem saída e somente este documento novo nesta entrega.

## 6. ENCERRAMENTO

Arquivo criado: docs/OBSERVABILIDADE_ETAPA1.md.
Nenhum código, configuração, modelo, peso ou artefato aprovado alterado.
Nenhum commit, push, merge ou mudança de índice.
Etapa 1 entregue com validação operacional complementada abaixo após o usuário
iniciar o Docker. A aprovação manual do usuário continua pendente.
Não iniciar Etapa 2 sem resposta explícita: aprovado, pode seguir.

## 7. Complemento operacional após inicialização do Docker

Conferência em 2026-11-17, aproximadamente 21:57 (America/Sao_Paulo).
Esta seção atualiza a limitação de disponibilidade registrada na seção 3;
os resultados anteriores foram preservados como histórico da inspeção.

Foi reutilizado o projeto existente `dice-model`, sem build, nova implantação,
reinício ou parada dos serviços. O acesso ao pipe Docker exigiu execução fora
do sandbox; consultas HTTP funcionaram diretamente.

| Verificação | Resultado observado |
| --- | --- |
| `just status dice-model` | API healthy; MinIO ativo; storage-init Exited (0) |
| `just health 23000` | HTTP 200 |
| POST com Content-Type application/json | HTTP 415, unsupported_media_type |
| POST com Content-Type image/png e corpo invalid | HTTP 422, invalid_image |
| `GET /docs.json` | Respostas documentadas: 200, 400, 404, 413, 415, 422, 500, 503, 504 |
| `GET /metrics` | HTTP 404, Not Found; baseline com métricas desativadas |
| `just recognize` com PNG local existente | HTTP 200, needs_review, dois resultados, request_requires_review=true |
| Identificação em /state/startup.json | Release v1-initial-20261117-r3; requested_device=cpu; resolved_device=cpu |
| HeadObject do original dessa requisição | 44.701 bytes; SHA-256 igual ao arquivo enviado |

Entrada utilizada: recorte já existente no dataset local, sem alterar seus bytes.
SHA-256: `8d9c1c714e9b419458ee2a8a7240b29724403bbc52cb0db27f1e73ad635705cb`.
request_id: `23887559c3c743f5a6dd647ea7f0dc5e`.
image_id: `c047f3d290d848ad972f193ca1d6a402`.
Tempos informados pela resposta: decode 2,209 ms; persist 13,224 ms;
detect 130,233 ms; classify 75,073 ms; total 237,869 ms.
São uma única observação do worker, não latência HTTP completa, benchmark,
histograma ou percentis. O recorte produziu dois candidatos com revisão;
esta conferência não avalia acurácia nem modifica regras do modelo.

O original foi persistido pelo comportamento existente da API. Não foram
impressas credenciais nem conteúdo da imagem. A cópia do OpenAPI consultado
ficou em `runtime/observability-stage1-openapi.json`, ignorada pelo Git.
Não foram implementados testes novos, logging, métricas ou retenção.

### Repetir a validação no Windows PowerShell

```powershell
Set-Location C:\MLops-dice-viwer\dice-viwer
just status dice-model
just health 23000
just recognize "data/datasets/v1-20261117-7b23477513de/classifier/crops/e2/00305b126c3d649706836060.png" 23000 image/png
curl.exe --include --max-time 10 -X POST http://127.0.0.1:23000/v1/recognize -H 'Content-Type: application/json' --data '{}'
curl.exe --include --max-time 10 -X POST http://127.0.0.1:23000/v1/recognize -H 'Content-Type: image/png' --data-binary 'invalid'
curl.exe --include --max-time 10 http://127.0.0.1:23000/metrics
git diff --check
git status --short
```

Esperado, em ordem: serviços no estado da tabela; readiness 200; inferência
200/needs_review para o mesmo recorte/modelo, com novos IDs e tempos variáveis;
415; 422; 404; diff --check sem saída; somente este documento não versionado.
O recorte depende do dataset local e não acompanha necessariamente um clone
portátil. Nesse caso, use uma imagem PNG própria e avalie o contrato, sem
exigir a mesma classificação. Cada envio válido cria outro original no MinIO.

Não foi feita carga, indução de indisponibilidade, timeout ou validação
exaustiva dos caminhos HTTP. O comportamento real observado é suficiente
para registrar o baseline operacional, sujeito à aprovação manual do usuário.
