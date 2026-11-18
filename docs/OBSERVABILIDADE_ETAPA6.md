# Etapa 6 — Carga reproduzível e testes integrados

Escopo: cliente de carga em biblioteca padrão Python, manifesto de PNG/JPEG,
seed, concorrência limitada, warmup separado e resultados individuais.
Não alterar inferência, artefatos aprovados ou política de registros.

Plano: testes com servidor HTTP local controlado para sequência reproduzível,
ausência de retry, sanitização e falhas; implementar cliente/CLI/receita Just;
testar na API real com reconciliação de JSONL/métricas; documentar e parar.

Padrão: 10 requisições medidas, seed 42, concorrência 1, um warmup sequencial,
timeout 45 s. Concorrência explícita até 16. Sem alvo de RPS: carga fechada,
nova tentativa entra quando há vaga. A ordem de entradas é reproduzível;
tempos, IDs, ordem de conclusão e rejeições sob concorrência podem variar.

Saída em diretório novo: run.json, attempts.jsonl e summary.json. O manifesto
de saída guarda hashes/tamanhos/tipos, sem caminhos das imagens. O manifesto
de entrada é configuração local e pode conter caminhos. Os registros do
cliente são artefatos de experimento, separados da retenção do JSONL da API.

## Implementação e arquivos

- Novo: `src/dice_viewer/recognition/load.py`, cliente, planejamento e resumo.
- Novo: `tools/load.py`, launcher sem instalar o pacote ou carregar o modelo.
- Alterado: `justfile`, receita `load manifest requests seed concurrency port`.
- Novos: `tests/test_load_generator.py` e `tests/test_load_integration.py`.
- Novo: este documento.

O servidor não foi alterado nesta etapa. Nenhum peso, política, aprovação,
release ou contrato foi modificado. Sem dependências novas, treinamento,
commit, push ou merge. O JSONL da API permanece no caminho anterior.

## Reprodutibilidade e registros

O manifesto de entrada tem formato `{"images":["imagem.png","outra.jpg"]}`.
Caminhos relativos são resolvidos em relação ao manifesto; caminhos absolutos
também são aceitos. Aceita 1 a 10000 caminhos únicos, com extensão PNG/JPEG,
arquivos não vazios de até 10 MiB. Os bytes não são decodificados pelo cliente:
imagens corrompidas podem ser usadas para observar rejeições da API.

As entradas são ordenadas por SHA-256 e tipo. Uma instância local Random(seed)
seleciona com reposição as entradas medidas; o warmup tem RNG separado com
a mesma seed. Alterar o número de warmups não muda o plano medido. O arquivo
é relido/verificado antes de cada tentativa: mudança de hash impede o envio
e gera input_changed. Os corpos não são mantidos todos em memória.

As tentativas medidas são numeradas a partir de zero. attempts.jsonl é
escrito na ordem de conclusão; ordenar por phase/sequence para comparar planos.
Só o coordenador escreve o arquivo e faz flush a cada linha. Não há retry
HTTP ou de transporte, redirecionamento ou uso de proxies do ambiente. O
destino deve ser uma origem HTTP(S), sem credenciais, query, fragmento ou
prefixo de caminho; o cliente acrescenta /v1/recognize. HTTPS verifica certificados.

run.json guarda entradas por hash/bytes/tipo, seed, parâmetros, alvo e versão
Python. attempts.jsonl registra fase, sequência, timestamp UTC, entrada por
hash, código HTTP, status do domínio, error_code permitido, IDs públicos,
sent, latency_ms e erro de cliente quando houver. Não salva imagens, cabeçalhos,
nomes de arquivo, texto bruto de resposta ou exceção. O manifesto local de
entrada contém caminhos e não deve ser confundido com esses resultados.

sent=true significa tentativa de operação de rede, não prova de recebimento
pelo servidor. Localmente, input_changed/input_unavailable não têm latência
HTTP nem resposta. timeout/transport_error também não são códigos HTTP.
Uma resposta HTTP 429/503 é mantida como tal. Um 429 externo BentoML pode não
ter request_id ou linha JSONL da aplicação; a métrica HTTP nativa ainda o conta.

O limite de resposta lida é 65536 bytes; excesso gera response_too_large.
JSON malformado ou excessivamente aninhado gera invalid_response, sem copiar
seu conteúdo ou abortar o restante da execução. O cliente projeta campos
permitidos, não valida sozinho todo o contrato de inferência.

summary.json contém somente a fase medida nas contagens HTTP/erros/latências;
warmup_attempts informa a fase excluída. failed_attempts_including_warmup
inclui erros do cliente e HTTP >=500 nas duas fases. HTTP 404/no_candidates,
needs_review e rejeições 4xx não viram erros técnicos por conveniência.
As contagens permitem analisar rejeições separadamente.

latency_ms mede a chamada de rede até leitura da resposta/erro, com relógio
monotônico; exclui fila no executor, leitura/hash local e gravação do resultado.
Os percentis usam nearest-rank: elemento ceil(p*N) da amostra ordenada.
Incluem todas as tentativas de rede medidas com duração, inclusive timeout e
HTTP de erro, e excluem erros locais anteriores à rede. Latências são amostras
individuais, adequadas a histogramas/boxplots futuros; um p95 com poucos casos
não é estimativa robusta. run_wall_seconds inclui warmup e custos locais da
execução, mas não a preparação inicial do manifesto.

O diretório de saída deve ser novo: não há overwrite, append a execução antiga
ou mistura silenciosa de resultados. O padrão é runtime/load/<UTC>-<sufixo>.
CLI imprime cada tentativa e um JSON final com output/summary.

| Código de saída | Significado |
| --- | --- |
| 0 | Execução completa, sem erro de cliente ou HTTP >=500, inclusive warmup |
| 1 | Execução completa com essas falhas observadas; artefatos foram produzidos |
| 2 | Erro de configuração/entrada/saída; diagnóstico fixo sem valores privados |
| 130 | Interrupção; arquivos podem conter somente parte da execução |

## Evidências e testes

11 testes com servidor HTTP real local controlado passaram no Python padrão,
sem carregar BentoML/PyTorch: seed, concorrência, ausência de retry, warmup,
sanitização, diretório existente, URL com credenciais/query, input alterado,
timeout, resposta inválida/profundamente aninhada, redirecionamento e excesso
de resposta. A revisão independente apontou RecursionError no parser de
resposta; teste reproduziu a falha, a correção converteu-a em invalid_response.

Dois testes integrados opt-in passaram contra a API instrumentada em CPU na
porta 24000. A fase sequencial reconcilia cinco tentativas (um warmup/quatro
medidas) com JSONL, IDs, características, número de dados e delta Prometheus.
A concorrente reconcilia sete tentativas (um warmup/seis medidas), incluindo
rejeições HTTP e eventuais admissões 429 sem linha da aplicação. Os testes
aguardam até 5 segundos pela escrita do JSONL após a resposta.

Também foram executadas as receitas Just com seis tentativas medidas:

- Concorrência 1: cinco HTTP 404/no_candidates e um 200/needs_review;
  seis respostas, zero erro de cliente; sete linhas de cliente com warmup.
- Concorrência 3: quatro HTTP 503/model_unavailable e dois 404; zero erro
  de cliente. O resumo foi gerado e o CLI/Just retornou 1 para sinalizar 5xx.
- Os planos medidos das duas execuções coincidiram ao ordenar por sequence.
  IDs, tempos e ordem de conclusão foram diferentes, como esperado.

A suíte completa final, com integração real habilitada, executou 56 testes:
55 passaram e um caso de symlink foi pulado no Windows. O caso de limite de
resposta usa uma fronteira pequena controlada no teste para não confundir
reset de conexão Windows com detecção do limite de leitura; a configuração
do cliente permanece 65536 bytes. Sintaxe AST válida em 56 arquivos Python;
just restore verificou a release original e os diffs protegidos ficaram vazios.

Artefatos dessas conferências estão em runtime/load; seguem locais e ignorados
pelo Git. Não se trata de benchmark, estimativa de acurácia ou aprovação do
modelo. Os originais válidos foram persistidos no MinIO pelo fluxo existente.

## Validação manual — Windows PowerShell

Primeiro, testes locais e prontidão:

```powershell
Set-Location C:\MLops-dice-viwer\dice-viwer
python -m unittest discover -s tests -p test_load_generator.py -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
just health 24000
```

Esperado: 11 testes do cliente passando; suíte completa com 56 testes,
normalmente três pulados (symlink Windows e dois integrados sem opt-in);
readiness HTTP 200. O contêiner dice-observability-stage4 permanece ativo.

Prepare um manifesto com o recorte local e uma imagem preta sintética:

```powershell
New-Item -ItemType Directory -Force '.\runtime\load-stage6' | Out-Null
.\.venv\Scripts\python.exe -c "from PIL import Image; Image.new('RGB',(640,640)).save('runtime/load-stage6/blank.png')"
$imagem = (Resolve-Path 'data/datasets/v1-20261117-7b23477513de/classifier/crops/e2/00305b126c3d649706836060.png').Path
$vazia = (Resolve-Path 'runtime/load-stage6/blank.png').Path
@{images=@($imagem,$vazia)} | ConvertTo-Json | Set-Content -Encoding utf8 '.\runtime\load-stage6\manifest.json'
```

Esperado: manifesto com duas entradas e PNG sintética de 640x640, sem tocar
em pesos ou imagens do dataset. Em outro clone, substitua o recorte por uma
PNG/JPEG própria; não exija a mesma classificação ou contagem de dados.

```powershell
$saida = just load runtime/load-stage6/manifest.json 6 42 1 24000
$saida
$execucao = $saida[-1] | ConvertFrom-Json
$pasta = $execucao.output
Get-Content (Join-Path $pasta 'summary.json')
@(Get-Content (Join-Path $pasta 'attempts.jsonl')).Count
```

Esperado: código 0 para esse conjunto em API ociosa; seis tentativas medidas,
um warmup, sete linhas, contagem 404=5 e 200=1 no resumo. O diretório é novo
a cada chamada. Repetir com mesma seed/imagens repete a sequência de hashes,
não os IDs ou latências.

Para a concorrência:

```powershell
just load runtime/load-stage6/manifest.json 6 42 3 24000
$LASTEXITCODE
```

Esperado: sete tentativas observáveis, com possíveis 429/503 devido à capacidade
atual da API. Código 1 é esperado se houver 5xx; significa falhas de capacidade
observadas, não ausência dos arquivos de saída. Não há retry para escondê-las.

Para executar a integração real, mantenha a API ociosa, sem outros clientes:

```powershell
$env:DICE_INTEGRATION_URL = 'http://127.0.0.1:24000'
$env:DICE_INTEGRATION_MANIFEST = (Resolve-Path 'runtime/load-stage6/manifest.json').Path
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_load_integration.py -v
$env:DICE_INTEGRATION_URL = $null
$env:DICE_INTEGRATION_MANIFEST = $null
```

Esperado: dois testes integrados passando. Eles enviam 12 requisições válidas
ao todo, persistem os originais correspondentes quando aceitos e conferem
JSONL/métricas. Seu resultado temporário de cliente é limpo ao terminar; o
JSONL da API permanece. Tráfego externo ou reinício durante o teste pode
alterar os deltas e fazê-lo falhar; não substitui sua aprovação manual.

Para saída explícita e controle de warmup/timeout, há também o launcher:

```powershell
python tools/load.py --manifest runtime/load-stage6/manifest.json --requests 6 --seed 42 --concurrency 1 --warmup 0 --timeout 45 --output runtime/load/validacao-manual-etapa6
```

Esperado: seis tentativas medidas sem warmup. Se o diretório já existir,
retorna 2 e o preserva; escolha outro nome para uma nova execução.

```powershell
just restore
git diff --check
git diff --exit-code -- src/dice_viewer/recognition/contract.py src/dice_viewer/evaluation/cascade.py
git status --short
```

Esperado: release verificada, diffs sem saída/código 0, somente arquivos
desta etapa e das anteriores ainda modificados/não versionados.

## Limitações e riscos

- A seed reproduz entradas, não escalonamento do SO, resposta sob saturação,
  estado do modelo, cold start, hardware ou condições de rede. Registrar
  ambiente e executar comparações separadas por CPU/dispositivo.
- Não há target RPS, sincronização de relógios, coordenação distribuída ou
  teste prolongado. Concorrência máxima 16; o resumo retém amostras em memória,
  portanto execuções muito grandes podem consumir memória significativa.
- O timeout urllib é de operações/inatividade de socket, não um prazo global
  rígido contra servidor que envie bytes indefinidamente. Ctrl+C pode aguardar
  threads ativas terminarem; não prometer interrupção imediata.
- O cliente não adiciona autenticação. Não há redirecionamento, retry ou proxy
  de ambiente. Não colocar credenciais em URLs/manifestos/nomes compartilhados.
- Saídas do cliente usam flush, sem garantia transacional/fsync por tentativa;
  interrupção pode deixar JSONL parcial e faltar summary.json. Ausência de
  complete=true exige tratar a execução como incompleta.
- Retenção do servidor pode remover segmentos antigos: reconciliar logo após
  a execução. Os arquivos do cliente não expiram por essa política e podem
  ocupar disco. A captura versionável da entrega será preparada na Etapa 7.

Etapa 6 encerra para validação manual. Não iniciar a Etapa 7 sem aprovação
explícita do usuário.
