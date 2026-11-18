# Etapa 2 — Logging JSONL por requisição e por dado

## Plano de implementação

Objetivo: registrar entradas permitidas, saídas, desempenho e erros das
requisições que alcançam a aplicação, preservando contrato, HTTP e modelo.
Referência: [baseline aprovado](OBSERVABILIDADE_ETAPA1.md).

Implementação na sessão atual, sem commits, mudança de pesos, dependências
novas, métricas, características de imagem, rotação ou retenção nesta etapa.
Testes autorizados explicitamente pelo pedido de observabilidade do usuário.

- [x] Criar testes `unittest` em `tests/test_recognition_observability.py`
  para append, concorrência de escrita, sanitização, falha de arquivo,
  rejeições, sucesso com dados, timeout e término tardio sem duplicação.
- [x] Criar `recognition/observability.py`: writer JSONL local, contexto por
  requisição, projeções permitidas e middleware ASGI que não consome nem
  modifica os corpos. Evento `request` único e eventos `die` da saída pública.
- [x] Integrar contexto em `inputs.py`, rejeição de MIME em `openapi.py` e
  resposta/worker em `service.py`. Não modificar `contract.py` ou cascata.
- [x] Executar testes, conferir sintaxe/diff e comparar hashes protegidos.
- [x] Entregar comandos de validação manual que executem o código de trabalho,
  sem substituir a release aprovada. Parar para aprovação da Etapa 2.

Decisões: `DICE_OBSERVABILITY_DIR` seleciona diretório; padrão
`DICE_STATE_DIR/observability` quando configurado, senão
`runtime/observability` na raiz do projeto. Arquivo `recognition.jsonl` UTF-8,
um objeto por linha, `schema_version=1`, timestamps UTC e durações em ms.
Um processo suportado, com lock entre threads. Não configurar handlers globais.

Falhas cobertas: disco/diretório indisponível não muda resposta; campos
arbitrários e mensagens de exceção não vazam; worker após 504 não gera outra
requisição; entrada rejeitada pelo middleware ainda gera evento; múltiplas
threads não intercalam linhas. Diagnóstico de falha contém apenas código fixo.

Limite conhecido: o middleware customizado é interno aos controles BentoML.
HTTP 429 do framework e falhas anteriores à aplicação não entram neste arquivo.
Não mudar admissão, timeout ou ordem interna do framework para ampliar cobertura.
Os controles externos permanecem candidatos às métricas nativas da Etapa 4.

## Comportamento entregue

`RecognitionLogging` envolve o middleware do contrato para observar os POSTs
que chegam a `/v1/recognize`. O descriptor associa o UUID público ao contexto;
não é aceito ID enviado pelo cliente. A resposta continua sendo a mesma
`JSONResponse`. O middleware encaminha os corpos sem copiá-los ou decodificá-los.
O contexto usa `ContextVar`, com referência explícita passada ao worker.

| Evento | Conteúdo |
| --- | --- |
| `request` | Uma conclusão por requisição observada; HTTP, erro seguro, IDs, versões, dispositivo, tipo de conteúdo permitido e bytes efetivamente lidos, resumo da saída e tempos |
| `die` | Um evento para cada resultado devolvido, com bbox, shape, die_type, face_value, três confianças, auto_accepted e review_reasons |
| `worker_completion` | Término posterior à conclusão/cancelamento HTTP, sem repetir evento request ou registrar dados como se tivessem sido devolvidos ao cliente |

Os eventos compartilham `schema_version`, `timestamp_utc`, `request_id`,
`image_id`, `versions` e `device`. Nas rejeições anteriores ao serviço,
versões/dispositivo podem ser nulos. `received_bytes` mede o corpo consumido
pela aplicação, não o Content-Length declarado, nem tráfego total na rede.
Para rejeição de MIME antes da leitura, esse número é zero.

`http_duration_ms` vai da entrada no middleware até o último corpo enviado
ao servidor ASGI. Exclui rede do cliente, controles externos do BentoML e a
escrita posterior do próprio log. Não é medição da latência ponta a ponta.
`worker_duration_ms` mede o processamento do worker; `worker_timings_ms`
contém apenas estágios concluídos. `output.timings_ms` mantém os tempos da
resposta pública. Ausência de um tempo não significa duração zero.

Após timeout/cancelamento, `worker_pending=true` indica que o worker ainda
estava ativo no encerramento HTTP. Se a persistência terminar depois do 504,
o evento tardio guarda o image_id efetivo sem alterar o registro anterior.
`worker_http_status` é o código do resultado interno tardio, não uma nova
resposta enviada ao cliente. Não contar `die`/`worker_completion` como requests.

A escrita usa UTF-8, `O_APPEND`, uma linha por objeto, lock no writer e
flush/close por lote. Um lote contém a requisição e seus dados. Falhas de
abertura, serialização ou escrita emitem apenas `observability_write_failed`,
sem payload, caminho, mensagem de exceção ou traceback. O traceback do erro
interno da aplicação foi substituído por diagnóstico com código fixo para
evitar vazamento de informações de dependências.

Não são gravados bytes/base64 de imagens, EXIF, nomes do cliente, cabeçalhos,
query, credenciais, tokens ou mensagens arbitrárias de erro. Resultados e
confianças são projeções de campos permitidos; não se serializa o corpo bruto.

## Arquivos desta etapa

- Novo: `src/dice_viewer/recognition/observability.py`.
- Alterados: `src/dice_viewer/recognition/service.py`, `inputs.py`, `openapi.py`.
- Novos: `tests/test_recognition_observability.py` e
  `tests/test_recognition_logging_service.py`.
- Novo: este documento.

O relatório da Etapa 1 já existia antes desta etapa e continua não versionado.
`contract.py`, cascata, limiares, pesos, aprovações, dump e release não foram
alterados. Nenhum commit, push, merge ou mudança de índice foi executado.

## Evidências de validação

- 19 testes `unittest` passaram. Há testes do fluxo real descriptor/serviço,
  com doubles somente para inferência e S3 externos; não carregam pesos.
- Casos: append, 80 escritas em oito threads, campos sigilosos, falha de
  arquivo, MIME inválido, erros, 404 sem candidatos, serviço ocupado, erro
  interno, timeout com persistência tardia, cancelamento, submissão ao executor,
  contextos ASGI simultâneos e falha de escrita do evento tardio.
- Testes reproduziram a correlação tardia incorreta e worker_pending indevido
  antes das respectivas correções; depois passaram.
- Revisão independente somente de leitura: sem alteração de contrato ou
  vazamento identificado; os problemas concretos de correlação/estado foram
  corrigidos e as lacunas principais de concorrência/cancelamento cobertas.
- Sintaxe AST válida em 46 arquivos de src/, tools/, deploy/v1/ e tests/.
- `just restore` verificou os 126 arquivos da release original.
- `git diff --check` sem erros; contrato público e cascata sem diff.

Foi iniciado o contêiner temporário `dice-observability-stage2` na porta 24000,
com imagem já existente, rede/MinIO de `dice-model`, fontes de trabalho em
mount somente leitura e estado separado em `runtime/observability-stage2/state`.
A API aprovada em 23000 permaneceu ativa. Não houve build/download de imagem.

Verificações HTTP: readiness 200; imagem real 200/needs_review com dois dados;
imagem sintética vazia 404/no_candidates; JSON rejeitado com 415; PNG inválido
com 422. As primeiras quatro requisições geraram quatro eventos request e
dois eventos die, com IDs de request únicos e sem token sentinela do cabeçalho.
Uma chamada posterior comparou respostas nas portas 23000 e 24000: todos os
campos, exceto IDs e tempos, coincidiram; ambas validaram RecognitionResponse.
O JSONL permanece cumulativo, portanto a contagem cresce nas repetições.

## Validação manual — Windows PowerShell

O contêiner temporário foi deixado ativo para sua conferência. Os comandos
abaixo usam as receitas existentes do Just sempre que disponíveis.

```powershell
Set-Location C:\MLops-dice-viwer\dice-viwer
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
just health 23000
just health 24000
```

Esperado: 19 testes passando, API aprovada e API instrumentada com HTTP 200.
Os testes usam diretórios temporários e não acessam o MinIO/modelo real.

```powershell
$log = '.\runtime\observability-stage2\state\observability\recognition.jsonl'
$antes = @(Get-Content $log).Count
just recognize "data/datasets/v1-20261117-7b23477513de/classifier/crops/e2/00305b126c3d649706836060.png" 24000 image/png
$depois = @(Get-Content $log)
$depois.Count - $antes
$depois | Select-Object -Last 3 | ForEach-Object { $_ | ConvertFrom-Json } | ConvertTo-Json -Depth 8
```

Esperado para o recorte local: HTTP 200/needs_review, dois dados; incremento
de três linhas (um request e dois die) e mesmo request_id/image_id nas três.
Os valores/decisões do modelo coincidem com a porta 23000; IDs e tempos variam.
O recorte existe neste ambiente, mas pode não acompanhar outro clone.
Uma PNG própria pode gerar contagem/status diferentes. Cada envio válido
armazena mais um original pelo comportamento existente do serviço.

```powershell
curl.exe --include --max-time 10 -X POST http://127.0.0.1:24000/v1/recognize -H 'Content-Type: application/json' -H 'Authorization: Bearer STAGE2_SECRET_SENTINEL' --data '{}'
curl.exe --include --max-time 10 -X POST http://127.0.0.1:24000/v1/recognize -H 'Content-Type: image/png' --data-binary 'invalid'
Get-Content $log | Select-Object -Last 2 | ForEach-Object { $_ | ConvertFrom-Json } | Select-Object event,request_id,http_status,error_code
Select-String -LiteralPath $log -SimpleMatch 'STAGE2_SECRET_SENTINEL'
```

Esperado: HTTP 415 e 422, uma linha request para cada erro com o mesmo
request_id da resposta, e nenhuma saída de Select-String. Use apenas o token
fictício mostrado; não coloque credenciais reais nos comandos de conferência.

```powershell
.\.venv\Scripts\python.exe -c "from PIL import Image; Image.new('RGB',(640,640)).save('runtime/observability-stage2/blank.png')"
just recognize "runtime/observability-stage2/blank.png" 24000 image/png
Get-Content $log | Select-Object -Last 1 | ConvertFrom-Json | ConvertTo-Json -Depth 8
```

Esperado: 404/no_candidates e somente evento request para essa imagem,
output.dices_count=0, error_code nulo e tempos/dispositivo CPU registrados.
404 sem candidatos é resultado do domínio, não falha técnica de inferência.

```powershell
just restore
git diff --check
git diff --exit-code -- src/dice_viewer/recognition/contract.py src/dice_viewer/evaluation/cascade.py
git status --short
```

Esperado: release verificada, duas verificações Git sem saída/código 0,
três fontes modificados e os novos arquivos listados acima, além do relatório
anterior. Arquivos de runtime são ignorados pelo Git.

### Reiniciar o contêiner de conferência

Se ele estiver parado, execute `docker start dice-observability-stage2`.
Se alterar fontes Python após o startup, reinicie somente esse contêiner.
Para pará-lo ao concluir, use `docker stop dice-observability-stage2`.
Não usar `just stop dice-model` para esta finalidade, pois isso para a API
aprovada e seu MinIO. O diretório de logs no host permanece após a parada.

Se o contêiner ainda não existir, esta é a configuração usada (portas livres,
Docker Linux e implantação dice-model ativa são pré-requisitos):

```powershell
$stageRoot = (Get-Location).Path
$stageState = Join-Path $stageRoot 'runtime\observability-stage2\state'
New-Item -ItemType Directory -Path $stageState -Force | Out-Null
$stageSource = (Resolve-Path 'src').Path
$stageSecrets = (Resolve-Path 'runtime\portable\dice-model\secrets').Path
$stageImage = docker inspect --format '{{.Image}}' dice-model-api-1
$stageManifest = Get-Content -Raw 'models\v1-initial-20261117-r3\manifest.json' | ConvertFrom-Json
$stagePin = $stageManifest.release_sha256
docker run --detach --name dice-observability-stage2 --network dice-model_default --publish 127.0.0.1:24000:3000 --read-only --tmpfs /tmp:size=256m --cap-drop ALL --security-opt no-new-privileges --mount "type=bind,source=$stageSource,target=/instrumented/src,readonly" --mount "type=bind,source=$stageState,target=/state" --mount "type=bind,source=$stageSecrets,target=/run/secrets,readonly" --env PYTHONPATH=/instrumented/src --env DICE_DEVICE=cpu --env "DICE_BUNDLE_SHA256=$stagePin" --env DICE_STATE_DIR=/state --env DICE_OBSERVABILITY_DIR=/state/observability --env S3_ENDPOINT=http://minio:9000 --env S3_ACCESS_KEY_FILE=/run/secrets/minio-user --env S3_SECRET_KEY_FILE=/run/secrets/minio-password --env DICE_IMAGE_BUCKET=dice-originals-v1 $stageImage python -m bentoml serve dice_viewer.recognition.service:DiceRecognitionService --host 0.0.0.0 --port 3000
```

Esperado: ID do novo contêiner; após carregamento/warmup, `just health 24000`
retorna 200. Os arquivos de credenciais são montados somente para leitura;
seus valores não são impressos. Essa montagem é um procedimento local de
conferência, não uma nova release aprovada ou entrega portátil final.

## Limitações e riscos

- O checkout local tem `config/yolo26n-die.yaml` diferente do hash científico
  aprovado. A execução direta com `recognition.serve` na raiz abortou na
  verificação de integridade. Essa diferença já existia; não foi corrigida
  nem contornada desabilitando verificações. A conferência usa os artefatos
  intactos da release, mantendo os fontes da aplicação em montagem separada.
- O logger não cobre rejeições/timeout externos do BentoML, falhas de startup
  ou pedidos que não alcançam seu middleware. Erro sem resposta enviada tem
  http_status nulo/response_completed=false; não se inventa código 500.
- Append-only é local, sem proteção contra edição externa. Não há fsync,
  garantia transacional, retenção, rotação ou limite de disco nesta etapa.
  Crash/disco cheio pode perder eventos ou deixar a última linha incompleta.
- Escrita síncrona e lock podem atrasar novas requisições em disco lento.
  Exceções de I/O são contidas; travamento do filesystem não é isolado.
  Não há suporte a vários processos compartilhando o mesmo arquivo.
- Dados numéricos de imagens e Prometheus continuam para as Etapas 3 e 4.
  O contêiner de conferência não altera o launcher portátil original.

Etapa 2 encerrada para validação manual. Não iniciar a Etapa 3 sem resposta
explícita do usuário: **aprovado, pode seguir**.
