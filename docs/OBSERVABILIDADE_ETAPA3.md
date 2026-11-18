# Etapa 3 — Características numéricas das imagens

## Plano e decisões

Escopo autorizado: somente características da imagem no JSONL da Etapa 2.
Não alterar o contrato HTTP, os pixels usados pelo modelo, os limiares,
os artefatos aprovados, Prometheus ou a política de retenção.

- [x] Criar testes numéricos com imagens conhecidas, preservação da fonte,
  limite de amostra e caso de imagem pequena; testar integração/falha isolada.
- [x] Criar `recognition/image_features.py`, usando Pillow/NumPy já presentes.
- [x] Integrar depois do decode existente, sem novo decode ou mudança no modelo.
- [x] Registrar características versionadas no request e no término tardio
  do worker, com estado de ausência/erro e duração da extração.
- [x] Executar testes, revisão, inferência comparativa e verificações protegidas;
  entregar procedimentos PowerShell e parar para aprovação manual.

Versão proposta: `rgb256-bilinear-bt601-lap4-v1`. Amostra sem ampliação,
maior lado até 256, proporção preservada por arredondamento ao inteiro mais
próximo. `resize` BILINEAR produz uma imagem separada da fonte RGB já
orientada pelo decoder. Arrays float64 existem apenas para a amostra.

Características: largura, altura, pixels e proporção originais; dimensões da
amostra; média/desvio populacional RGB normalizados em [0,1]; luma
Y=0.299R+0.587G+0.114B; brilho=média de Y; contraste=desvio populacional de Y;
nitidez=variância populacional de 4Y-cima-baixo-esquerda-direita apenas no
interior. Imagem de amostra menor que 3 em algum lado tem nitidez nula
(null) e zero pixels válidos para esse cálculo, sem inventar uma medida.

`input.image_features` terá `version`, `status`, `extraction_ms` e campos
numéricos quando disponíveis. O schema JSONL global continua 1, com adição
de campo opcional; filtrar pela versão de características em análises futuras.
Nenhum histograma, KS, PSI, drift ou julgamento de qualidade será calculado.

## Campos e interpretação

A extração ocorre uma vez por cena, depois de `decode_image`, antes da
persistência/inferência, no worker já existente. Dimensões correspondem ao
RGB orientado por EXIF. O original persistido continua sendo o upload exato.
Não são extraídas características dos recortes individuais dos dados.

| Campo em input.image_features | Definição/unidade |
| --- | --- |
| version | rgb256-bilinear-bt601-lap4-v1 |
| status | ok, not_available, pending ou error |
| width, height, pixels | Dimensões e total de pixels originais após orientação |
| aspect_ratio | width / height |
| sample_width, sample_height | Dimensões da amostra; maior lado até 256 |
| rgb_mean.red/green/blue | Média populacional de cada canal dividido por 255 |
| rgb_std.red/green/blue | Desvio padrão populacional de cada canal normalizado |
| brightness_mean | Média da luma da amostra, aproximadamente entre 0 e 1 |
| contrast_std | Desvio padrão populacional da luma, aproximadamente entre 0 e 0,5 |
| laplacian_variance | Variância populacional do Laplaciano interior, em unidades de luma normalizada ao quadrado |
| laplacian_valid_pixels | Número de pixels interiores que contribuíram para a medida |
| extraction_ms | Duração observada da extração, em milissegundos |

O Laplaciano usa `4*centro - cima - baixo - esquerda - direita`, sem padding.
Sua variância para valores normalizados está aproximadamente entre 0 e 16.
`laplacian_variance=null` com `laplacian_valid_pixels=0` significa amostra
pequena demais, não uma imagem perfeitamente lisa. Os outros campos ainda
podem ter status ok. Valores de ponto flutuante admitem resíduos numéricos.

Os coeficientes são aplicados aos canais RGB decodificados sem linearização
de gamma ou calibração de câmera; brilho aqui é um descritor de intensidade,
não uma medida física de iluminação. Nitidez não é probabilidade de acerto
nem limiar para aceite/revisão. Nenhuma dessas medidas altera a classificação.

Quando a imagem não é decodificada/processada, o bloco contém version,
status=not_available e reason=not_decoded_or_not_processed. Não há números
inventados para essas entradas. Se a resposta terminar durante a extração,
o request guarda status=pending; o evento worker_completion posterior guarda
o resultado final, sem modificar a linha anterior nem duplicar a requisição.
Se a extração lançar exceção, status=error e extraction_ms são registrados,
o diagnóstico usa somente `observability_feature_extraction_failed` e a
inferência continua. Não são serializados EXIF, imagem ou mensagem da exceção.

O JSONL anterior permanece legível e não é reescrito: eventos de etapas
anteriores podem não conter image_features. Em análises futuras, selecionar
uma única versão e status=ok, tratar ausências explicitamente e escolher uma
amostra por requisição. Não contar novamente o mesmo request a partir dos
eventos die ou worker_completion. Para casos pending, é possível associar o
resultado tardio pelo request_id. Não misturar amostras ausentes com zeros.

## Arquivos desta etapa

- Novo: `src/dice_viewer/recognition/image_features.py`.
- Alterados: `src/dice_viewer/recognition/observability.py` e `service.py`.
- Novo: `tests/test_image_features.py`.
- Ampliados: `tests/test_recognition_logging_service.py` e
  `tests/test_recognition_observability.py`.
- Novo: este documento.

Os demais arquivos modificados/não versionados no Git já pertenciam às
Etapas 1 e 2. Não houve commit, push, merge ou mudança de índice. Não foram
alterados pesos, política, cascata, contrato, release ou configuração aprovada.

## Evidências

- 29 testes passaram: os 19 anteriores e dez verificações novas/ampliadas,
  incluindo momentos numéricos conhecidos, degrau preto/branco após BILINEAR,
  amostra limitada, pixels/metadados originais preservados, orientação EXIF,
  falha simulada de extração e cancelamento enquanto a extração está pending.
- Os testes de integração primeiro falharam por ausência dos novos campos;
  passaram após integrar a extração ao serviço e ao contexto de observação.
- Revisão independente somente de leitura sem achados bloqueantes; as duas
  lacunas sugeridas de reamostragem/cancelamento foram cobertas por testes.
- Sintaxe AST válida em 48 arquivos de src/, tools/, deploy/v1/ e tests/.
- `just restore` verificou os 126 arquivos da release original.
- Contrato e cascata sem diff; `git diff --check` sem erros.

O contêiner temporário da Etapa 2 foi reiniciado para carregar os fontes da
Etapa 3. Ele mantém o nome `dice-observability-stage2`, porta 24000 e logs em
`runtime/observability-stage2/state/observability/recognition.jsonl`.
A API aprovada na porta 23000 permaneceu ativa; ambas retornaram readiness 200.

Na imagem real local de 224x224, a resposta manteve HTTP 200/needs_review e
os mesmos dois resultados da API original. A comparação de todos os campos,
exceto IDs e tempos, foi idêntica; ambas validaram RecognitionResponse e
nenhuma incluiu image_features no corpo HTTP. No JSONL dessa observação:

- brightness_mean: 0,5360463479141656.
- contrast_std: 0,16770964395177748.
- laplacian_variance: 0,008037906908434226.
- laplacian_valid_pixels: 49284.
- extraction_ms: 9,722001999762142.

Esse tempo é uma observação isolada, não um benchmark ou percentil.
A imagem preta de 640x640 retornou 404/no_candidates, amostra 256x256 e
brilho/contraste/Laplaciano iguais a zero. PNG inválido retornou 422 e JSON
retornou 415; ambos geraram características com status=not_available.

## Validação manual — Windows PowerShell

O contêiner de conferência permanece ativo com a Etapa 3 carregada.

```powershell
Set-Location C:\MLops-dice-viwer\dice-viwer
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
just health 23000
just health 24000
```

Esperado: 29 testes passando; readiness HTTP 200 nas duas portas.
Se o contêiner temporário estiver parado, use `docker start dice-observability-stage2`
e aguarde carregar. Para recarregar fontes já montados, use
`docker restart dice-observability-stage2`; não reinicie a API aprovada.
A criação completa do contêiner está documentada na Etapa 2.

```powershell
just recognize "data/datasets/v1-20261117-7b23477513de/classifier/crops/e2/00305b126c3d649706836060.png" 24000 image/png
$log = '.\runtime\observability-stage2\state\observability\recognition.jsonl'
$request = Get-Content $log | ForEach-Object { $_ | ConvertFrom-Json } | Where-Object event -eq 'request' | Select-Object -Last 1
$request.input.image_features | ConvertTo-Json -Depth 5
```

Esperado: HTTP 200/needs_review com dois dados; contrato HTTP sem campos novos.
No log, version=rgb256-bilinear-bt601-lap4-v1, status=ok, dimensões originais
e de amostra 224x224; números próximos aos registrados acima e tempo variável.
O recorte está disponível neste ambiente; em outro clone use uma PNG própria
e confira as dimensões correspondentes, sem exigir a mesma classificação.

```powershell
.\.venv\Scripts\python.exe -c "from PIL import Image; Image.new('RGB',(640,640)).save('runtime/observability-stage2/blank.png')"
just recognize "runtime/observability-stage2/blank.png" 24000 image/png
$request = Get-Content $log | ForEach-Object { $_ | ConvertFrom-Json } | Where-Object event -eq 'request' | Select-Object -Last 1
$request.input.image_features | ConvertTo-Json -Depth 5
```

Esperado: 404/no_candidates; width=height=640; pixels=409600;
sample_width=sample_height=256; brightness_mean=contrast_std=laplacian_variance=0;
laplacian_valid_pixels=64516. Um envio válido persiste o original no MinIO,
como já ocorria antes da instrumentação.

```powershell
curl.exe --include --max-time 10 -X POST http://127.0.0.1:24000/v1/recognize -H 'Content-Type: image/png' --data-binary 'invalid'
Get-Content $log | Select-Object -Last 1 | ConvertFrom-Json | Select-Object -ExpandProperty input | ConvertTo-Json -Depth 5
```

Esperado: 422/invalid_image; image_features.status=not_available, sem dimensões,
brilho, contraste ou nitidez fictícios. A versão aparece mesmo sem extração.

```powershell
just restore
git diff --check
git diff --exit-code -- src/dice_viewer/recognition/contract.py src/dice_viewer/evaluation/cascade.py
git status --short
```

Esperado: release verificada, verificações Git sem saída/código 0, somente
mudanças desta etapa e das anteriores ainda não versionadas. Os logs permanecem
cumulativos no runtime ignorado pelo Git. Para parar apenas a conferência:
`docker stop dice-observability-stage2`; os logs no host são preservados.

## Limitações e riscos

- A reamostragem suaviza detalhes. A nitidez é calculada na amostra e depende
  do tamanho/proporção da imagem; não comparar diretamente versões ou técnicas
  de amostragem diferentes. Os valores não diagnosticam drift por si só.
- Há custo adicional de CPU/tempo dentro do mesmo prazo de processamento de
  30 segundos; o prazo não foi aumentado. O total já existente na resposta
  inclui esse trabalho; o novo extraction_ms aparece somente no log.
- Só o array numérico tem tamanho limitado; o decode RGB original continua
  sujeito aos limites atuais de bytes/dimensões/pixels e ao custo do Pillow.
- Falhas de extração são isoladas, mas o custo de CPU e memória não é nulo.
  No caso de crash ou indisponibilidade do próprio logger, pode haver perda.
- Continuam as limitações da Etapa 2: controles externos BentoML fora do
  logger, escrita síncrona, um processo e ausência de rotação/retenção.
- A execução de conferência usa a release intacta e fontes em mount separado
  devido à divergência preexistente de configuração do checkout. Não é uma
  substituição da entrega portátil final ou da aprovação dos artefatos.

Etapa 3 encerrada para validação manual. Não iniciar a Etapa 4 sem resposta
explícita do usuário: **aprovado, pode seguir**.
