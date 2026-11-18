# RPG Dice — inferência com observabilidade MLOps

O serviço reconhece dados físicos de RPG em imagens. Esta entrega registra
entradas numéricas, resultados, tempos e erros em JSONL e expõe GET /metrics.
O contrato de POST /v1/recognize e os artefatos científicos foram preservados.

## O log é automático — onde fica e o que deve ser versionado

**O uso normal da API já grava o log. Não é necessário executar o gerador de
carga nem ativar um modo especial.** Em um diretório novo, apenas subir o
serviço não cria o JSONL: ele é criado automaticamente quando a primeira
requisição POST /v1/recognize termina, inclusive se ela retornar erro.
Se o arquivo já existir, novos eventos são acrescentados ao histórico.
Consultar readiness ou /metrics não gera eventos de inferência nesse arquivo.

| Arquivo | Finalidade | Git |
| --- | --- | --- |
| runtime/portable/dice-observability/observability/recognition.jsonl | Log ativo do serviço; cresce durante o uso normal | Permanece no runtime ignorado |
| [docs/evidence/observability/recognition.jsonl](docs/evidence/observability/recognition.jsonl) | Cópia para o professor avaliar | **Deve ser adicionada e commitada na entrega; está fora do gitignore** |

A cópia de entrega é uma captura fixa, não é atualizada automaticamente pelo
serviço. Estar fora do gitignore não significa já estar versionada: ela só
acompanha um clone depois de incluída em um commit. Não versione credenciais,
volumes ou todo o runtime para entregar esse arquivo.

## 1. Antes de começar

Abra o PowerShell na pasta do repositório, onde está o arquivo justfile.
Precisa de Python 3.11.8+, Docker com containers Linux, Compose com suporte
a --wait e Just. Não precisa de dataset, PyTorch ou credenciais no host
para executar a aplicação e gerar carga.

```powershell
python --version
docker version
just --version
```

Esperado: versões dos programas e seção Server do Docker. Se não conectar
ao engine, abra o Docker Desktop e aguarde. Portas 25000–25002 devem estar
livres. O primeiro build baixa dependências e usa cerca de 4 GB para a imagem;
cada subida consulta o registry.

## 2. Terminal 1 — iniciar e acompanhar o arquivo

```powershell
just observe dice-observability 25000
just health 25000
just recognize docs/evidence/observability/inputs/black.png 25000 image/png
```

Espere o primeiro comando terminar. Esperado: API saudável em CPU, readiness
HTTP 200 e inferência HTTP 404/no_candidates. A imagem preta é sintética e
não contém dados para reconhecer. **Esse 404 é esperado**, não uma falha.
O envio inicial cria o arquivo de log.

Agora, ainda no Terminal 1:

```powershell
Get-Content '.\runtime\portable\dice-observability\observability\recognition.jsonl' -Tail 5 -Wait
```

Deixe esse comando aberto. Ele mostra as últimas cinco linhas e acompanha
novas linhas. Cada linha é um objeto JSON, não uma imagem. O arquivo
permanece no Windows após parar ou reiniciar o contêiner.

## 3. Terminal 2 — gerar carga e ver o rastro evoluir

Abra outro PowerShell na mesma pasta:

```powershell
just load docs/evidence/observability/input-manifest.json 20 42 1 25000
```

Argumentos: manifesto, quantidade medida, seed, concorrência e porta.
Esperado: **20 tentativas medidas e um warmup**, sem retry. As imagens
sintéticas normalmente retornam 404/no_candidates. No Terminal 1 devem
aparecer 21 novos eventos request, além do envio inicial. A seed repete
a sequência de entradas, não os IDs ou tempos.

O comando informa uma pasta nova em runtime/load contendo:

- run.json: parâmetros e hashes das entradas, sem nomes de imagens;
- attempts.jsonl: uma linha por tentativa, com fase warmup/measured;
- summary.json: contagens e percentis somente da fase medida.

Para imagem própria: `just recognize "C:/imagens/dado.png" 25000 image/png`.
JPEG exige image/jpeg. Pode retornar 200/ok, needs_review ou 404 sem candidatos.
Quando há resultados, cada dado gera também um evento die. Não conte esses
eventos como outras requisições. O ID da resposta permite localizar seu registro.

## 4. Terminal 2 — conferir /metrics

```powershell
curl.exe --include --max-time 10 http://127.0.0.1:25000/metrics
```

Esperado: HTTP 200 e texto Prometheus, com HELP, TYPE e séries:

- bentoml_service_request_total: contador por endpoint/código HTTP;
- bentoml_service_request_duration_seconds_bucket/count/sum: duração HTTP em segundos;
- bentoml_service_request_in_progress: requisições em andamento.

As durações do JSONL são em milissegundos; as métricas acima usam segundos.
Contadores reiniciam com o processo; o JSONL persistido continua. Scrapes
não contam como inferência nem acrescentam eventos ao JSONL de inferência.
O [guia de análise](docs/OBSERVABILIDADE.md) explica média, percentis,
histogramas, boxplots e taxas, com comandos e um gráfico pronto.

## 5. Testar e encerrar

Os testes do gerador usam somente Python e um servidor local temporário:

```powershell
python -m unittest discover -s tests -p test_load_generator.py -v
```

Esperado: 11 testes passando. A suíte completa pode usar a imagem Docker pronta,
sem instalar ML no host; o comando está no [guia completo](docs/OBSERVABILIDADE.md#testes-e-encerramento).

Ctrl+C no Terminal 1 encerra apenas o acompanhamento do arquivo. Para parar
os serviços, execute no Terminal 2:

```powershell
just observe-status dice-observability
just observe-stop dice-observability
```

Esperado: status e depois parada, preservando logs, originais e credenciais.
Para retomar, repita just observe. Não remova volumes se quiser preservar
os originais. Código 1 na carga indica erro de cliente/HTTP 5xx; concorrência
maior pode produzir 429/503, registrados sem retry.

## Material para avaliação

- [Evidências versionáveis, incluindo JSONL](docs/evidence/observability/README.md).
- [Histórico JSONL de exemplo](docs/evidence/observability/recognition.jsonl).
- [Entendimento do negócio e justificativa das variáveis](docs/ENTENDIMENTO_DO_NEGOCIO.md).
- [Execução, análise, retenção e limitações](docs/OBSERVABILIDADE.md).

Ativo: runtime/portable/dice-observability/observability/recognition.jsonl.
As cópias em docs/evidence são fixas e não expiram pela retenção operacional.
Runtime continua ignorado para preservar estado e credenciais.

Sem Just: `python tools/observe.py run --project dice-observability --port 25000`
e `python tools/load.py --manifest docs/evidence/observability/input-manifest.json --requests 20 --seed 42 --base-url http://127.0.0.1:25000`.
`just up` mantém a V1 original; use **just observe** para a aplicação instrumentada.
Outros documentos: [índice](docs/README.md) e [execução original](docs/EXECUCAO.md).
