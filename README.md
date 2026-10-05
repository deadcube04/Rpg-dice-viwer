# Reconhecimento de dados de RPG

Serviço Python local para uma foto de **um** D6, D8, D10, D12 ou D20. A resposta inclui tipo, face, confiança, decisão e versão. Zero ou vários dados exigem revisão. O símbolo `0` no D10 é apresentado como `10`.

## Dados e atribuição

Fonte: [G-G-Games/diecamera-frames](https://huggingface.co/datasets/G-G-Games/diecamera-frames), derivado do [dieCamera](https://github.com/eschatus/diecamera), licença declarada **CC BY-SA 4.0**. Esta cópia local foi usada para fins acadêmicos. A cópia do manifesto tem 460 fotos; o README da fonte informa números mais antigos. O snapshot é identificado pelo SHA-256 do `metadata.jsonl` e dos arquivos em `data/raw/snapshot.json`.

O split ocorre por foto, antes dos recortes. Treino: outras câmeras e datas, 342 fotos e 655 recortes com face confiável; validação: HD USB a partir de 13/08/2026, 52 fotos e 84 recortes; teste reservado: iPhone e Nintendo Switch, 66 fotos e 134 recortes. Os IDs, câmeras e hashes ficam em `data/processed/frame_manifest.jsonl` e `reader_manifest.jsonl`. A única anotação inválida `D20=0` permanece na detecção, mas não é usada para ler a face. Fotos com `values_trusted=false` contribuem apenas para detecção.

O teste usa câmeras distintas **dentro do mesmo dieCamera**. Ainda falta a avaliação em outro dataset exigida pelo documento original. Os resultados e metas devem ser reportados separadamente.

## Ambiente

Requer Python 3.11, Docker Desktop com WSL2 e GPU NVIDIA para treino e serviço. A referência é RTX 2060 com 6 GB. Na raiz do projeto, instalar `uv` e executar `uv sync --group train --locked`. Um `uv.exe` local pode ficar em `.tools/Scripts`; nesse caso, inclua essa pasta no `PATH` da sessão. O lock fixa as versões usadas; no Windows, Torch e Torchvision vêm do índice CUDA 12.6.

Copie `.env.example` para `.env`, crie credenciais locais fortes e mantenha o arquivo fora do Git. Na sessão PowerShell que executa os comandos Python, importe as variáveis do arquivo:

```powershell
Get-Content .env | ForEach-Object { if ($_ -match '^([^#=]+)=(.*)$') { [Environment]::SetEnvironmentVariable($matches[1],$matches[2],'Process') } }
$env:DVC_SITE_CACHE_DIR = (Join-Path (Get-Location) '.cache/dvc-site')
$env:DVC_SYSTEM_CONFIG_DIR = (Join-Path (Get-Location) '.cache/dvc-system')
$env:DVC_NO_ANALYTICS = 'true'
```

Inicie armazenamento e rastreamento:

```powershell
docker compose up -d minio
uv run --group train python -m dice_viewer.bootstrap
docker compose up -d mlflow
```

Os endpoints do Compose são publicados somente em `127.0.0.1`. O MinIO usa volumes locais, portanto **não é backup externo**. O bucket de dados é `dice-datasets`; MLflow usa SQLite persistente e `dice-mlflow-artifacts`.

## Dados, treino e escolha

O snapshot original é criado uma vez, antes de reproduzir os estágios. `dvc add` e `dvc push` o publicam no MinIO local. Credenciais são lidas de variáveis de ambiente.

```powershell
uv run --group train dice-data snapshot --source C:\diecamera-frames\data --target data\raw
uv run --group train dvc add data\raw
uv run --group train dvc push data\raw.dvc
uv run --group train dvc repro
```

`dvc repro` prepara os recortes, treina YOLO11n e os dois leitores (MobileNetV3 Small e EfficientNet-B0), e seleciona na validação. Há semente 42, parada antecipada e hiperparâmetros no MLflow. O detector usa imagens com caixas válidas. Só recortes com faces confiáveis alimentam os leitores. A confiança de inferência é o produto das probabilidades de detector, tipo e valor; ela define uma ordenação, não uma probabilidade calibrada. O limiar é congelado antes do teste. A calibração exige limite superior de Wilson de 95% inferior a 5% para falso aceite e pelo menos 20 aceitações. Se não houver evidência, o limiar `1.01` envia todas as predições para revisão.

O teste reservado é executado **uma única vez** depois de `selection.json`:

```powershell
uv run --group train dice-evaluate test
uv run --group train dice-promote
```

`artifacts/evaluation.json` registra métricas por dado e por imagem com um dado, intervalo de Wilson, detecção, leitor com recorte correto, aceitação e p50/p95. A meta de negócio é acerto conjunto ≥95%, falso aceite <5%, p95 de inferência ≤1 s e redução ≥50% do tempo mediano. Uma meta não atingida deve ser declarada como tal.

`dice-report` escreve `artifacts/target-report.json` com os quatro critérios e marca evidência ausente como meta ainda não comprovada. Execute novamente após o benchmark e o estudo humano.

## Serviço e operação

Depois da promoção, fixe `DICE_MODEL_TAG` em `.env` à tag retornada. Gere a imagem BentoML com `bentoml build` e `bentoml containerize <bento:tag>`; registre `DICE_IMAGE` em `.env` com a tag imutável da imagem criada. Execute `dice-release --image <imagem:tag> --bento <bento:tag>` para guardar o ID de imagem e o predecessor em `artifacts/releases.jsonl`. Então:

```powershell
docker compose up -d dice-service
```

API local em `http://127.0.0.1:3000`:

- `POST /v1/predictions`: multipart com campo `file`, JPEG ou PNG até 10 MB/25 MP. Retorna `prediction_id`, `die_type` (por exemplo, `D20`), `sides`, `value`, `confidence`, `model_version`, `status` (`accepted`/`review_required`) e `reason`.
- `POST /v1/predictions/{id}/feedback`: JSON `{"confirmed":true,"sides":20,"value":14}`. Uma confirmação deve coincidir com a predição; uma correção usa `confirmed:false`.
- `GET /health`: prontidão de GPU, modelo, SQLite e MinIO. `GET /v1/metrics`: métricas Prometheus.

As imagens de entrada são guardadas em `dice-feedback`, e predições e feedbacks no SQLite persistente. Feedback não promove nem treina modelos automaticamente. Faça revisão periódica dos exemplos confirmados; correções espontâneas não medem a taxa real de erro.

Após aquecimento do contêiner, `dice-benchmark` registra p50/p95 da inferência e da chamada HTTP. `dice-study manual` e `dice-study api` conduzem o estudo interativo nas mesmas 30 fotos de teste, registrando medianas e erros em `artifacts/study-*.json`. `dice-study compare` calcula a redução observada. A observação humana é necessária. `dice-feedback-report --database <caminho-do-sqlite>` compara a distribuição de feedback com a de treino. Para verificar a operação, envie fotos com um, zero e vários dados; corrija uma predição, reinicie o serviço e confirme a persistência; em seguida aponte `DICE_IMAGE`/`DICE_MODEL_TAG` para a versão anterior, execute `docker compose up -d --force-recreate dice-service` e confira `/health`.

## Versionamento e limites

[GitHub do projeto](https://github.com/deadcube04/Rpg-dice-viwer) guarda código, configuração, `uv.lock`, descritores DVC e documentação. Imagens, pesos, caches, logs, bancos e segredos ficam fora do Git. Cada leitor registra a revisão de código, hash do código, hash de manifesto e split e ID de run MLflow. A promoção vincula esses identificadores à tag BentoML. O modelo promovido também é exportado ao bucket local `dice-models`.

Esta fase mede dados e modelo e faz verificação manual da API; não inclui suíte automatizada de testes. Não há integração com o RPG Manager nem leitura automática de vários dados.
