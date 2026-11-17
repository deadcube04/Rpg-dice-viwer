# Etapa 5 — serviço integrado para conferência local

Entrega de 17/11/2026. O serviço `DiceRecognitionService` recebe `POST /v1/recognize` com JPEG/PNG binário, valida a imagem, confirma a persistência do original no MinIO e executa a versão inicial aprovada `v1-e1-20261117T175200Z`. Não houve treinamento, ajuste de limiares ou alteração da avaliação E2. O retreino continua previsto para execução manual posterior.

## Execução e contrato

Serviço de conferência em `http://127.0.0.1:13000`; readiness em `/readyz`; documentação na raiz `/` e contrato em `/docs.json`. [OpenAPI obtido do serviço](../../evidence/stage5-20261117/openapi.json). A entrada explícita é `recognition_service:DiceRecognitionService`; a entrada histórica `service:DiceService` e os arquivos antigos de Compose continuam pertencendo à demonstração anterior.

No diretório `C:\MLops-dice-viwer\dice-viwer`, iniciar o serviço com o MinIO isolado já existente:

```cmd
docker start dice-stage5-minio
.venv\Scripts\python.exe -m dice_viewer.recognition.serve --local-minio-env runtime\stage5\minio.env --device cpu
```

O arquivo de credenciais está somente no runtime ignorado pelo Git e não faz parte do relatório. O launcher também aceita `S3_ENDPOINT`, `S3_ACCESS_KEY`, `S3_SECRET_KEY` e `DICE_IMAGE_BUCKET` do ambiente, dispensando `--local-minio-env`. `--bootstrap-bucket` cria um bucket privado ausente; não altera políticas nem apaga conteúdo. `--device cuda` usa CUDA quando disponível; `--device auto` permite fallback CPU. O padrão é CPU, um worker e uma inferência por vez.

```cmd
curl.exe -X POST -H "Content-Type: image/png" --data-binary @runtime/stage5/needs_review.png http://127.0.0.1:13000/v1/recognize
```

O MinIO de conferência usa `127.0.0.1:19000`, console `127.0.0.1:19001`, container `dice-stage5-minio`, volume `dice-stage5-minio-data` e bucket `dice-stage5-originals`. A imagem já instalada foi fixada em `rpg-media-minio@sha256:a3475aaa2694039238a5728913f45f58c45539012c74228796408fb8d9147ba6`. Os containers anteriores do projeto RPG não foram iniciados nem alterados. O bucket inspecionado não tem política pública nem expiração por lifecycle. Os dados persistem após reiniciar o container.

## Evidências observadas

Conferências HTTP reais, inspeção de objetos MinIO, leitura de código e compilação Python; não foram criados ou executados testes automatizados.

| Situação conferida | Resultado observado |
|---|---|
| Dado individual da validação E1 | `200 ok`, um item aceito |
| Cena E2 com candidato abaixo do limiar | `200 needs_review`, predições e motivos preservados |
| Cena E2 sem candidatos | `404 no_candidates`, lista vazia, revisão obrigatória e original salvo |
| Cena E2 com nove candidatos | `422 too_many_dice`, referência da imagem salva, sem subconjunto parcial |
| JPEG com orientação EXIF 6 | `200 needs_review`, caixas sobre a imagem orientada |
| Lado com 4.097 pixels | `413 image_too_large`, sem persistência |
| 4.001 × 4.000 pixels | `413 image_too_large`, sem persistência |
| Corpo maior que 10 MiB, incluindo envio chunked | `413 image_too_large` |
| Corpo inválido e PNG com IHDR/IEND mas sem IDAT | `422 invalid_image` |
| Tipo de conteúdo não suportado, inclusive pickle BentoML | `415 unsupported_media_type`, envelope público |
| Limiar enviado por query string | `400 invalid_request` |
| MinIO isolado parado durante uma requisição válida | `503 storage_unavailable`, `image_id = null`; armazenamento restabelecido depois |
| `CUDA_VISIBLE_DEVICES=-1`, dispositivo `auto` | Inicialização CPU com motivo `cuda_unavailable_cpu_fallback`; inferência HTTP concluída |
| Dispositivo `cuda` | Inicialização `cuda:0`; inferência HTTP concluída com os mesmos rótulos e decisão do exemplo CPU |

[Requisições e respostas](../../evidence/stage5-20261117/http-observations.json), [inspeção da persistência](../../evidence/stage5-20261117/persistence-inspection.json), [falha de armazenamento](../../evidence/stage5-20261117/storage-failure.json), [log CPU](../../evidence/stage5-20261117/service-cpu.log) e [log CUDA](../../evidence/stage5-20261117/service-cuda.log). A instância CUDA temporária foi encerrada; a CPU permanece disponível para conferência.

Seis objetos de respostas bem-sucedidas ou erros pós-inferência foram recuperados do MinIO: todos preservaram exatamente os bytes enviados, com SHA-256 igual ao metadado. Incluem JPEG com EXIF, ausência de candidatos e excesso de candidatos. Não houve reencodificação do original armazenado.

Limites: 10.485.760 bytes, 16 milhões de pixels e 4.096 pixels por lado, cumulativamente; imagem estática; oito candidatos operacionais, 64 técnicos; filtro 0,25; aceite 0,95/0,80/0,80 sem arredondamento. Os parâmetros são os congelados em E2. Atingir o limite técnico produz `422 scene_too_complex`; esse ramo e a indisponibilidade por memória foram inspecionados em código, sem fabricar excesso de detecções ou provocar OOM.

## Tempos e limites da medição

A resposta inclui `decode`, `persist`, `detect`, `classify` e `total` em milissegundos. Exemplos observados: cena de revisão CPU 389,07 ms; dado individual CPU 208,17 ms; primeira requisição CUDA 1.813,61 ms, incluindo inicialização preguiçosa do preditor. São observações funcionais de poucas requisições, sem benchmark de carga ou comparação estatística CPU/GPU. Não substituem as distribuições de latência offline de E2 nem os tempos de treinamento já registrados.

Para preservar integralmente a implementação científica de E2, a imagem RGB orientada é serializada temporariamente em PNG sem perda, em memória, antes de entrar na cascata congelada. Esse custo e a segunda leitura interna entram no `total`; `detect` e `classify` vêm dos relógios sincronizados da cascata. O original persistido continua intacto. O cliente também observa upload e custos HTTP externos ao `total`.

O processamento tem prazo de 30 segundos, após recebimento do corpo; o recebimento tem prazo de 45 segundos. O timeout externo BentoML de 90 segundos permite que a aplicação produza seu `504 inference_timeout`. Uma chamada Torch/S3 já iniciada não é interrompida à força: após 504, o worker mantém o lock até terminar e requisições concorrentes recebem `503 model_unavailable`. Nenhuma nova fase de inferência começa se o prazo já venceu. Uma escrita S3 concluída depois da resposta pode deixar um original retido sem referência confirmada na resposta; `image_id` só é informado após confirmação. Esse comportamento foi considerado na revisão. Não foi induzida uma inferência longa ou OOM após a correção final.

## Revisão e preservação

Uma revisão independente somente leitura identificou o tratamento de PNG estruturalmente inválido e indisponibilidade por memória; ambos foram corrigidos. O primeiro foi conferido novamente por HTTP (`422`); OOM foi validado por inspeção, sem provocar falta de memória. A exceção de MIME pickle do BentoML foi harmonizada no middleware e conferida por HTTP (`415`). O adaptador global `Image.open` do Ultralytics tentava instalar `pi-heif` ao receber conteúdo inválido; a entrada passou a usar diretamente os decodificadores JPEG/PNG, e o launcher desativa auto instalação. Nenhuma dependência nova foi instalada para essa correção.

Context7 foi consultado, mas a inicialização do conector falhou. Foram consultados a [documentação oficial de tipos BentoML](https://docs.bentoml.org/en/latest/build-with-bentoml/iotypes.html) e o código da versão instalada 1.4.39. A adaptação usa o descriptor nativo, entrada raiz posicional exclusiva e saída estruturada; o OpenAPI efetivamente servido foi conferido.

Os checkpoints aprovados e os 58 arquivos do manifesto de entrega E2 foram conferidos por hash, com resultados em `frozen-inputs.json`. Não houve alteração de pesos, mapas, transformações ou código científico congelado. Os arquivos novos do serviço e evidências desta entrega ficam identificados em `DELIVERY_CHECKSUMS.json`.

## Próxima parada

A Etapa 5 está entregue para aprovação do fluxo integrado, conforme a parada explícita do plano. A Etapa 7 — bundle imutável, Compose CPU/CUDA, implantação da versão empacotada e rollback — permanece pendente. Iniciar localmente o serviço para esta conferência não conclui esse empacotamento operacional.
