# Etapa 4 — Prometheus nativo

Escopo: habilitar o exportador BentoML 1.4.39 em GET /metrics sem duplicar
contadores/histogramas HTTP. JSONL permanece no caminho atual, conforme pedido.
Sem instalar Prometheus, dashboards, coleta externa, retenção ou gerador de carga.

Plano: teste do app ASGI configurado, habilitação no decorator do serviço,
validação HTTP real no contêiner temporário, regressão e procedimentos manuais.
O teste primeiro retornou 404 em /metrics com a configuração anterior.

Arquivos desta etapa: service.py (metrics.enabled=True e filtro ASGI),
recognition/metrics.py (novo), tests/test_recognition_metrics.py (novo) e este
documento. O filtro reaproveita as amostras nativas, sem registrar contadores
ou histogramas adicionais.

## Métricas expostas

GET /metrics retorna HTTP 200, Content-Type text/plain com versão Prometheus.
As séries com label endpoint são limitadas a /v1/recognize na exposição;
as demais métricas nativas sem label endpoint são preservadas. Isso impede
publicar caminhos arbitrários recebidos pelo framework, inclusive possíveis
IDs ou nomes na URL. Scrapes de /metrics não incrementam a contagem HTTP.
Não são usados request_id, image_id, características, confianças ou valores
contínuos como labels. Não há contadores personalizados duplicados.

| Série | Significado |
| --- | --- |
| bentoml_service_request_total | Requisições HTTP por endpoint e http_response_code |
| bentoml_service_request_duration_seconds_bucket | Buckets cumulativos de duração em segundos |
| bentoml_service_request_duration_seconds_count | Número de durações observadas |
| bentoml_service_request_duration_seconds_sum | Soma das durações em segundos |
| bentoml_service_request_in_progress | Requisições em andamento |

O nome service_version pode ser "not available" ao servir fontes diretamente;
não é a versão do modelo. As versões científicas continuam no contrato/JSONL.
Readiness e rotas desconhecidas não são exportadas pelo filtro de caminhos.
A duração HTTP nativa inclui os middlewares internos ao instrumento; não é
tempo de inferência puro, nem latência de rede medida no cliente.

Exemplo de taxa de erro técnico para um Prometheus futuro:

```promql
sum(rate(bentoml_service_request_total{endpoint="/v1/recognize",http_response_code=~"5.."}[5m]))
/
sum(rate(bentoml_service_request_total{endpoint="/v1/recognize"}[5m]))
```

Sem tráfego, a razão pode ser indefinida. HTTP 404/no_candidates e 200/needs_review
não são erros técnicos; analisar rejeições 4xx separadamente. Nenhum servidor
Prometheus foi instalado nesta etapa.

## Validação realizada

30 testes passaram, incluindo teste do app ASGI real sem carregar pesos.
O cliente Prometheus nativo é configurado sem multiprocess apenas nesse teste
em memória, pois não existe supervisor BentoML nele; a configuração original
é restaurada ao terminar. Pode aparecer aviso asyncio de inicialização lenta.

O teste confirma 200 e formato parseável, incremento unitário do contador e
histogram_count para 415, scrapes excluídos, JSONL sem eventos de scrape,
e ausência de PRIVATE_MARKER após GET /unknown/PRIVATE_MARKER (404 preservado).
O caso de caminho arbitrário falhou antes de adicionar o filtro e passou depois.
Revisão independente identificou esse risco de label; ele foi tratado na
exposição. Não houve mudança no contrato ou nos modelos.

Na conferência Linux real, contador e histogram_count incrementaram exatamente
uma vez após 415; scrapes sucessivos não incrementaram os contadores. As
respostas do modelo em 23000 e 24000 validaram RecognitionResponse e coincidiram
em todos os campos, exceto IDs/tempos. A release original passou em just restore.

O contêiner antigo de conferência dice-observability-stage2 foi parado e
preservado. A porta 24000 agora pertence a dice-observability-stage4, com os
mesmos fontes e diretório de JSONL. A API aprovada em 23000 não foi alterada.
O contêiner novo usa BENTOML_HOME=/tmp/bentoml para arquivos internos mmap:
o bind mount Windows de /state produziu exposição vazia mesmo com DBs nativos
criados. Em /tmp Linux, as séries foram lidas corretamente. /state continua
recebendo JSONL/startup; o compose original já usa volume Linux nomeado.

## Validação manual no Windows PowerShell

```powershell
Set-Location C:\MLops-dice-viwer\dice-viwer
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
just health 24000
curl.exe --include --max-time 10 http://127.0.0.1:24000/metrics
```

Esperado: 30 testes passando, readiness 200, metrics 200 e texto com HELP,
TYPE e séries bentoml_service. Se não houver tráfego de inferência, envie uma
requisição antes de exigir séries dessa rota. O contêiner permanece ativo;
se estiver parado: docker start dice-observability-stage4 e aguarde readiness.
Não iniciar simultaneamente o contêiner antigo na mesma porta.

```powershell
curl.exe --max-time 10 -X POST http://127.0.0.1:24000/v1/recognize -H 'Content-Type: application/json' --data '{}'
$metrics = (Invoke-WebRequest http://127.0.0.1:24000/metrics).Content
$metrics -split "`n" | Select-String 'request_total|request_duration_seconds_count|request_in_progress'
```

Esperado: erro 415 no contrato atual e séries do endpoint /v1/recognize com
http_response_code="415". Repetir o POST aumenta contador e histogram_count
em um; apenas consultar /metrics não aumenta esses valores.

```powershell
curl.exe --include --max-time 10 http://127.0.0.1:24000/unknown/PRIVATE_MARKER
(Invoke-WebRequest http://127.0.0.1:24000/metrics).Content | Select-String 'PRIVATE_MARKER'
just recognize "data/datasets/v1-20261117-7b23477513de/classifier/crops/e2/00305b126c3d649706836060.png" 24000 image/png
```

Esperado: 404 na rota desconhecida; nenhuma saída contendo PRIVATE_MARKER;
inferência 200/needs_review para esse recorte local, com contagem/histograma
http_response_code="200" no próximo scrape. O JSONL continua no mesmo caminho
runtime/observability-stage2/state/observability/recognition.jsonl.

```powershell
just restore
git diff --check
git diff --exit-code -- src/dice_viewer/recognition/contract.py src/dice_viewer/evaluation/cascade.py
git status --short
```

Esperado: release verificada, verificações Git sem saída/código 0 e somente
as mudanças desta etapa e das anteriores ainda não versionadas. Nenhum commit,
push ou merge. Para parar só a conferência: docker stop dice-observability-stage4.

## Limitações

- O filtro limita a exposição, mas o BentoML ainda aloca séries internas
  usando caminhos arbitrários. Muitos caminhos distintos podem consumir
  memória/armazenamento interno; o filtro não é proteção contra esse tráfego.
- Os logs de acesso nativos do framework continuam independentes do JSONL.
  Não enviar credenciais em URLs. O filtro protege a representação /metrics.
- Métricas são efêmeras e reiniciam com o processo; usar rate/increase para
  tratar resets em um coletor futuro. JSONL é a evidência histórica separada.
- Arquivos mmap devem ficar em filesystem Linux, não em compartilhamento
  Windows. Aqui /tmp é efêmero e suficiente; não há nova dependência externa.
- Exposição é local (127.0.0.1 no Docker), sem autenticação adicionada.
  Não foram instalados dashboards, alertas, drift ou infraestrutura de coleta.
- Só a aplicação de conferência foi atualizada. A release aprovada original
  continua intacta; a entrega portátil final será tratada nas etapas seguintes.

Etapa 4 termina para validação manual. Não iniciar a Etapa 5 sem aprovação
explícita do usuário.
