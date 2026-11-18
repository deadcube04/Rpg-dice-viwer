# Observabilidade MLOps — RPG Dice Viewer

Esta entrega instrumenta a V1 mantendo os pesos, política, limiares, CPU padrão
e contrato público de POST /v1/recognize. Não implementa treinamento, retreino,
modelos novos, dashboards, alertas de drift ou serviço externo de logs.

A [justificativa de negócio/CRISP-DM](ENTENDIMENTO_DO_NEGOCIO.md) explica por
que cada variável foi escolhida. Para acompanhar o rastro ao vivo com dois
terminais, siga o [README](../README.md).

## Executar em Windows PowerShell

Pré-requisitos: Python 3.11.8+, Docker Desktop com containers Linux amd64,
Compose com suporte a --wait e, opcionalmente, Just. O primeiro build requer
internet e aproximadamente 4 GB para a imagem, além de caches e originais.
Não é necessário dataset local, credencial prévia ou PyTorch no host.
O launcher consulta o registry do MinIO em cada run, mesmo com cache;
portanto a retomada por esse comando não promete operação offline.

Na raiz do clone, com portas 25000–25002 livres:

```powershell
just observe dice-observability 25000
just health 25000
just observe-status dice-observability
curl.exe --include --max-time 10 http://127.0.0.1:25000/metrics
```

Esperado: pacote verificado, API healthy em CPU, MinIO ativo, storage-init
concluído com código 0 e HTTP 200 para readiness/métricas. Antes de inferências,
podem existir poucas séries exportadas. API/docs ficam em 25000; MinIO e seu
console ficam em 25001 e 25002, restritos a 127.0.0.1.

Sem Just, o equivalente é:

```powershell
python tools/observe.py run --project dice-observability --port 25000
python tools/observe.py status --project dice-observability
```

O launcher reutiliza a imagem da release intacta; o modelo é verificado pelos
checks existentes. Os fontes instrumentados são montados em /instrumented/src,
somente leitura. Antes da subida, módulos científicos atuais devem coincidir
com os hashes da release; a raiz do pacote admite o inicializador atual que
contém apenas docstring. Alterações em modelos/cascata são recusadas. O comando
primeiro inicia a release original e depois recria a API com o overlay; espere
o término bem-sucedido antes de assumir que a instrumentação está ativa.

Essa camada de aplicação não constitui aprovação científica nova. Os arquivos
de release/dump/aprovação não são editados e o contrato é o mesmo. A configuração
do checkout que já diferia do protocolo não é usada: as configurações científicas
continuam no pacote aprovado. Não use just up para atualizar a instrumentação;
ele executa a V1 original. Após editar fontes, execute just observe novamente.

## Enviar imagens e gerar carga

```powershell
just recognize docs/evidence/observability/inputs/black.png 25000 image/png
just load docs/evidence/observability/input-manifest.json 20 42 1 25000
```

A primeira chamada retorna normalmente 404/no_candidates. A carga usa duas
imagens sintéticas, um warmup separado e 20 tentativas medidas, seed 42,
concorrência 1. O resultado esperado é HTTP 404, sem erros de cliente. Isso
exercita decode/persist/detect/contrato/observabilidade, não acurácia de dados.

Cada chamada válida persiste um original no MinIO pelo comportamento já
existente. Para imagem própria: `just recognize "C:/imagens/dado.png" 25000 image/png`.
JPEG exige image/jpeg. A resposta poderá ser 200/ok ou needs_review, 404 sem
candidatos ou erro conforme o contrato. Mais detalhes: [Etapa 6](OBSERVABILIDADE_ETAPA6.md).

O gerador usa somente biblioteca padrão Python. Sem Just:

```powershell
python tools/load.py --manifest docs/evidence/observability/input-manifest.json --requests 20 --seed 42 --concurrency 1 --warmup 1 --timeout 45 --base-url http://127.0.0.1:25000
```

Ele informa o diretório novo em runtime/load, com run.json, attempts.jsonl e
summary.json. Não há retry, redirect ou uso de proxies do ambiente. Código 1
significa que os artefatos foram produzidos, mas houve erro do cliente ou 5xx;
2 indica configuração/entrada/saída inválida. O mesmo plano se repete por seed,
mas tempos e resultados sob concorrência variam. Concorrência até 16; aumentar
para 3 pode produzir 429/503. Isso mede limites existentes, sem mudar admissão.

## Onde ficam os registros

Para projeto dice-observability:

```powershell
$log = '.\runtime\portable\dice-observability\observability\recognition.jsonl'
Get-Content $log -Tail 5
Get-ChildItem (Split-Path $log) -Filter '*.jsonl'
```

Dentro da API, o caminho é /state/observability/recognition.jsonl. O diretório
está montado no host e permanece após stop/restart. Outro nome de projeto tem
seu diretório próprio; não compartilhe arquivo entre processos. O histórico
de desenvolvimento anterior continua em runtime/observability-stage2/state/
observability; ele não foi movido. A cópia de entrega em docs/evidence é uma
captura versionável, independente do runtime ignorado pelo Git.

| Evento | Como analisar |
| --- | --- |
| request | Uma conclusão por POST /v1/recognize, incluindo respostas das camadas de timeout/admissão do BentoML |
| die | Resultado individual devolvido; relacionar por request_id/image_id |
| worker_completion | Processamento que terminou após a conclusão/cancelamento HTTP; não é outra requisição |

Campos comuns: schema_version, timestamp_utc, request_id, image_id, versions e
device. request inclui HTTP, erro seguro, bytes consumidos, tipo permitido,
resumo da saída, tempos e características numéricas. die inclui bbox, tipo,
valor, três confianças, revisão e aceite automático. Não são gravadas imagens,
base64, EXIF, nomes do cliente, query, cabeçalhos ou mensagens brutas de exceção.

O observador envolve a aplicação ASGI completa pelo service_class do BentoML,
fora das camadas de admissão/timeout e tratamento de erros. Respostas externas
429/504/500 também geram request, com códigos seguros framework_rejected,
framework_timeout e framework_error. Se o corpo nativo não tiver request_id,
o registro usa um ID interno; ele não é adicionado ao corpo ou header público.
Nesses casos a reconciliação usa janela e contagens por HTTP, sem fingir um ID
de resposta inexistente. Bytes não consumidos podem aparecer como zero.
Não há segundo observador interno, evitando duplicação de eventos normais.

Os novos requests têm http_observation_version=2. Capturas anteriores sem esse
campo não distinguem as revisões do observador; trate-as como histórico e não
misture suas latências com a nova medida. Falhas de startup ou de protocolo antes de existir um escopo
ASGI continuam fora do log de inferência.

input.image_features possui versão própria, status e extraction_ms. A versão
rgb256-bilinear-bt601-lap4-v1 usa o RGB já orientado, com amostra de maior lado
até 256, sem alterar pixels da inferência. Inclui dimensões, brilho e contraste
de luma normalizada, momentos RGB e variância do Laplaciano. Ausências não
viram zeros; amostras menores que 3 em algum lado têm Laplaciano null.
Definições exatas: [Etapa 3](OBSERVABILIDADE_ETAPA3.md).

## Latência: escolher a medida correta

| Campo | Unidade e escopo |
| --- | --- |
| attempts.jsonl: latency_ms | Milissegundos, rede cliente até leitura da resposta/erro; inclui respostas HTTP de erro e timeouts; exclui leitura/hash do arquivo, fila e escrita do cliente |
| request: http_duration_ms | Milissegundos na fronteira ASGI completa até último corpo enviado, incluindo controles BentoML; exclui rede do cliente e escrita posterior do log |
| request: worker_duration_ms | Milissegundos no worker, inclusive extração e trabalho até o resultado interno |
| output.timings_ms | Tempos já definidos no contrato: decode, persist, detect, classify e total do worker |
| input.image_features.extraction_ms | Custo da extração opcional, somente no log |
| Prometheus request_duration_seconds | Segundos medidos pela instrumentação HTTP nativa; escopo inclui middlewares internos |

Não subtrair timestamps de hosts diferentes para obter latência: os relógios
podem não estar sincronizados. As durações são medidas com relógio monotônico.
Não somar todos os tempos como componentes disjuntos: total inclui componentes
e existem overheads. Falta de medição não significa zero. Separar CPU/CUDA,
versão, resultado HTTP, warmup e workload antes de comparar desempenho.

Para uma amostra válida de N durações x_i, média = soma(x_i)/N. Mediana é o
centro dos dados ordenados (média dos dois centrais quando N é par). O gerador
usa percentis nearest-rank: p95 é o elemento ceil(0.95*N), numerado a partir
de 1; seu p50 pode diferir da mediana aritmética para N par. Declarar o método.

## Calcular média, mediana, percentis e taxas

O exemplo abaixo usa a captura final entregue. Troque o caminho por uma
execução nova para analisar o seu ambiente. Ele não altera nenhum registro.

```powershell
@'
import json, math, statistics
from pathlib import Path
rows = [json.loads(line) for line in Path('docs/evidence/observability/load-final/attempts.jsonl').read_text(encoding='utf-8').splitlines()]
rows = [r for r in rows if r['phase'] == 'measured']
x = sorted(r['latency_ms'] for r in rows if r['latency_ms'] is not None)
responses = [r for r in rows if r['http_status'] is not None]
technical = sum(500 <= r['http_status'] < 600 for r in responses)
client_failures = sum(r['client_error'] is not None for r in rows)
print('Tentativas medidas:', len(rows), 'respostas HTTP:', len(responses), 'amostras:', len(x))
if x:
    print('Media ms:', statistics.mean(x), 'mediana ms:', statistics.median(x))
    print('p95 nearest-rank ms:', x[math.ceil(.95*len(x))-1])
print('Taxa tecnica HTTP 5xx:', technical/len(responses) if responses else None)
print('Taxa de falha do cliente:', client_failures/len(rows) if rows else None)
'@ | python -
```

404/no_candidates é resultado do domínio; 200/needs_review não é erro técnico.
Relatar 4xx/rejeições separadamente, incluindo 429, além de 5xx e falhas de
cliente. Os denominadores acima são diferentes de propósito: resposta HTTP
ausente não vira status 500 fictício. Uma tentativa pode ter HTTP e falha de
leitura ao mesmo tempo; as categorias não são necessariamente disjuntas.

## Histogramas e boxplots

Um histograma conta observações em intervalos de duração; comparar somente
com as mesmas unidades, limites de bins e população. Declarar N e separar
warmup. Histograma concentrado à esquerda com cauda longa indica que a média
pode ocultar respostas lentas. Respostas rápidas de erro não indicam melhoria
da inferência; estratificar por HTTP/status e declarar erros excluídos.

No boxplot, a caixa cobre Q1–Q3 (25%–75%), a linha marca a mediana e IQR=Q3-Q1.
Bigodes de Tukey vão até os valores observados dentro de Q1-1.5*IQR e
Q3+1.5*IQR. Pontos externos são observações para investigar, não erros a apagar.
O cálculo de quartis depende do método; o exemplo usa Matplotlib padrão.

Exemplo opcional com o ambiente local já preparado (Matplotlib é dependência
transitiva existente, não requisito do launcher portátil):

```powershell
@'
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
rows = [json.loads(line) for line in Path('docs/evidence/observability/load-final/attempts.jsonl').read_text(encoding='utf-8').splitlines()]
x = [r['latency_ms'] for r in rows if r['phase']=='measured' and r['latency_ms'] is not None]
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
bins = [0, 50, 100, 150, 200, 250, 300, 400, 600, max(1000, max(x)+1)]
ax[0].hist(x, bins=bins)
ax[0].set(xlabel='Latencia do cliente (ms)', ylabel='Tentativas', title=f'Histograma, N={len(x)}')
ax[1].boxplot(x, whis=1.5)
ax[1].set(ylabel='Latencia do cliente (ms)', title='Boxplot, sem warmup')
fig.tight_layout()
Path('runtime').mkdir(exist_ok=True)
fig.savefig('runtime/latencia-avaliacao.png', dpi=160)
'@ | .\.venv\Scripts\python.exe -
```

Esperado: runtime/latencia-avaliacao.png. A entrega inclui um
[gráfico já renderizado](evidence/observability/latency-final.png), sem exigir
Matplotlib do professor. As duas imagens sintéticas produzem 404; o gráfico
caracteriza apenas esse fluxo no ambiente observado, não desempenho E2 de dados.

## Prometheus

GET /metrics reutiliza instrumentos nativos do BentoML. Nenhum coletor externo
foi instalado. A exposição mantém séries de /v1/recognize e métricas sem label
endpoint; descarta caminhos arbitrários e não conta os próprios scrapes.
O framework ainda aloca séries internas por caminho: o filtro de exposição
não limita esse consumo interno. O serviço está restrito a localhost.

Principais séries: bentoml_service_request_total, request_in_progress e
request_duration_seconds_bucket/count/sum, com o mesmo prefixo. A label
http_response_code permite taxas por HTTP; service_version pode ser "not
available" ao servir fontes e não substitui a versão do modelo no JSONL.
Os arquivos mmap internos ficam em /tmp Linux; não usá-los em bind Windows.

Exemplo didático de leitura, com todas as séries da mesma população/janela:

```text
request_duration_seconds_bucket{le="0.1"} 8
request_duration_seconds_bucket{le="0.25"} 10
request_duration_seconds_bucket{le="+Inf"} 10
request_duration_seconds_sum 1.1
request_duration_seconds_count 10
```

São oito respostas de até 100 ms e duas entre 100 e 250 ms: buckets são
acumulativos, então subtraia 10-8 para obter o segundo intervalo. A média é
1.1/10=0.11 segundo, ou 110 ms. Se dois dos dez HTTP forem 5xx, a taxa técnica
é 2/10=20%. Em uma janela real, use diferenças entre scrapes ou rate, tratando
resets. Não é possível reconstruir um boxplot exato desses buckets: use as
amostras individuais do JSONL/cliente. Warmup não é separado no contador nativo;
para excluí-lo, use phase=measured dos registros do cliente.

Exemplo PromQL de razão de erro técnico em 5 minutos, caso se conecte um
Prometheus futuramente:

```promql
sum(rate(bentoml_service_request_total{endpoint="/v1/recognize",http_response_code=~"5.."}[5m]))
/
sum(rate(bentoml_service_request_total{endpoint="/v1/recognize"}[5m]))
```

Para p95 HTTP aproximado por buckets:

```promql
histogram_quantile(0.95, sum by (le) (rate(bentoml_service_request_duration_seconds_bucket{endpoint="/v1/recognize"}[5m])))
```

Sem tráfego, razões podem ser indefinidas. rate/increase tratam resets de
contador, mas o histórico depende de um coletor. Buckets são cumulativos,
aproximam quantis e não substituem amostras individuais para boxplots/KS/PSI.

## Dados para análises futuras: KS e PSI

O JSONL guarda valores individuais de brilho, contraste, RGB, nitidez,
dimensões, confiança e tempos, associados a versão, dispositivo e resultado.
Isso permite montar referência/janela futura sem alterar o modelo. Não há
cálculo de drift, alerta ou mudança automática de aceite nesta entrega.

Antes de comparar: selecionar mesma feature version e população, status=ok
da extração, um registro por request; não duplicar eventos die/worker_completion.
Valores null/ausentes exigem uma taxa de ausência separada, nunca substituição
silenciosa por zero. Para confiança por dado, estratificar por tipo/modelo e
considerar dependência entre dados da mesma imagem. Scores não são probabilidades
calibradas. Repetições da mesma imagem na carga não são amostras independentes.

KS compara distribuições empíricas: D=max_x |F_referencia(x)-F_atual(x)|.
Relatar tamanhos das amostras, D, procedimento e, se aplicável, p-valor com suas
hipóteses. Diferença estatística não prova queda de acurácia; imagens repetidas,
seleção de casos e comparações múltiplas podem invalidar interpretações simples.

Para PSI, fixar bins usando apenas referência e tratar extremos/ausências.
Com proporções p_i da referência e q_i atuais:
PSI = soma_i (q_i-p_i)*ln(q_i/p_i). Para zeros, declarar suavização, por exemplo
adicionar epsilon às contagens e renormalizar antes das proporções. Não ajustar
bins após observar o resultado. Não existe limiar universal de drift/retreino;
a política precisaria de validação e rótulos independentes, fora deste escopo.

## Persistência e retenção

Default: rotação diária UTC ou 10 MiB, retenção de 30 dias e teto de 1 GiB de
bytes lógicos. DICE_LOG_MAX_BYTES, DICE_LOG_RETENTION_DAYS e
DICE_LOG_MAX_TOTAL_BYTES permitem configuração na criação da API; retenção
aceita 1..36500 dias e teto total deve cobrir um segmento. Valores inválidos
emitem diagnóstico fixo e usam os padrões. Configure variáveis antes de
just observe, que as encaminha pelo overlay Compose.

Cada lote usa append, flush, fsync e close. A rotação renomeia o ativo e
preserva seus bytes; a limpeza exclui apenas segmentos próprios fechados,
mais antigos primeiro, após write bem-sucedido. Nunca exclui ativo, originais
MinIO, imagens, modelos ou arquivos alheios. Uma cauda incompleta é isolada
num segmento fechado sem ser reescrita. Symlinks não são candidatos à limpeza.

O teto pode reduzir o histórico abaixo de 30 dias. Sem tráfego não há timer de
limpeza. Disco já cheio pode exigir liberação externa; falhas de I/O podem
impedir o teto. Um lote não é dividido e, acima do teto total, é rejeitado com
diagnóstico. Não há coordenação multiprocesso, transação, fsync de diretório
ou garantia contra adulteração/pane. Rotação e cópia de entrega têm propósitos
diferentes: a captura em docs/evidence não é expirada pelo serviço.

## Testes e encerramento

Testes exigem o ambiente de desenvolvimento com dependências do projeto;
o launcher e o gerador não exigem esse ambiente no host. Com uv 0.12.23:

```powershell
uv sync --locked
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Sem API configurada, os dois testes live são pulados. No Windows sem privilégio
de symlink, mais um teste é pulado; ele foi executado em Linux. Para integração
real, mantenha a API sem outro tráfego:

```powershell
$env:DICE_INTEGRATION_URL = 'http://127.0.0.1:25000'
$env:DICE_INTEGRATION_MANIFEST = (Resolve-Path 'docs/evidence/observability/input-manifest.json').Path
$env:DICE_INTEGRATION_LOG = (Resolve-Path 'runtime/portable/dice-observability/observability/recognition.jsonl').Path
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_load_integration.py -v
$env:DICE_INTEGRATION_URL = $null
$env:DICE_INTEGRATION_MANIFEST = $null
$env:DICE_INTEGRATION_LOG = $null
```

Esperado: dois testes passando e 12 requisições reais geradas. Conferem
tentativas/IDs/JSONL/características/dados e deltas Prometheus; tráfego externo
ou reinício durante a conferência pode invalidar os deltas.

Alternativamente, a suíte inteira pode usar a imagem Docker já pronta, sem
instalar dependências ML no host. Este comando usa mounts somente leitura,
um contêiner descartável e a API ociosa do projeto padrão:

```powershell
$sourceDir = (Resolve-Path 'src').Path
$testsDir = (Resolve-Path 'tests').Path
$toolsDir = (Resolve-Path 'tools').Path
$evidenceDir = (Resolve-Path 'docs/evidence/observability').Path
$logsDir = (Resolve-Path 'runtime/portable/dice-observability/observability').Path
$testImage = docker inspect --format '{{.Image}}' dice-observability-api-1
docker run --rm --read-only --tmpfs /tmp:size=512m --network dice-observability_default --workdir /workspace --mount "type=bind,source=$sourceDir,target=/workspace/src,readonly" --mount "type=bind,source=$testsDir,target=/workspace/tests,readonly" --mount "type=bind,source=$toolsDir,target=/workspace/tools,readonly" --mount "type=bind,source=$evidenceDir,target=/workspace/evidence,readonly" --mount "type=bind,source=$logsDir,target=/logs,readonly" --env PYTHONPATH=/workspace/src --env BENTOML_HOME=/tmp/bentoml-tests --env YOLO_CONFIG_DIR=/tmp/ultralytics-tests --env MPLCONFIGDIR=/tmp/matplotlib-tests --env DICE_INTEGRATION_URL=http://api:3000 --env DICE_INTEGRATION_MANIFEST=/workspace/evidence/input-manifest.json --env DICE_INTEGRATION_LOG=/logs/recognition.jsonl $testImage python -m unittest discover -s tests -q
```

Esperado na entrega revisada: 63 testes passando no Linux, inclusive symlink e os dois
testes live. Podem aparecer avisos informativos de inicialização asyncio e
configuração temporária do Ultralytics. A suíte gera 12 chamadas reais à API;
o contêiner de teste é removido, mas o JSONL da API continua persistido.

```powershell
just observe-logs dice-observability
just observe-stop dice-observability
```

Esperado: logs de operação e serviços desse projeto parados, com volumes,
credenciais e JSONL preservados. Para retomar, use just observe novamente.
Não executar docker compose down -v se quiser preservar originais/estado.
Não usar as receitas status/logs/stop da V1 original para gerir o overlay;
use observe-status/observe-logs/observe-stop.

## Critérios e limites da entrega

As [evidências](evidence/observability/README.md) incluem histórico JSONL,
tentativas individuais, resumo, exposição Prometheus, imagens sintéticas,
gráfico e manifesto de hashes. Nenhuma dessas evidências altera a aprovação
do modelo ou demonstra ausência de drift/adequação a produção. As etapas
incrementais e suas limitações estão em OBSERVABILIDADE_ETAPA1.md até ETAPA7.md.
Não houve commit/push/merge. A validação manual final do usuário continua
necessária; os testes automatizados não a substituem.
