# Etapa 7 — versão inicial empacotada e em operação local

Entrega técnica de 17/11/2026. O usuário aprovou o fluxo integrado da Etapa 5 e autorizou esta etapa; [registro da aprovação](../../evidence/approvals/stage5-flow-20261117.json). A versão final `v1-initial-20261117-r3` está em execução CPU no Docker, com o modelo inicial aprovado `v1-e1-20261117T175200Z`. O retreino permanece previsto para execução manual posterior.

**API e documentação:** http://127.0.0.1:23000. **Reconhecimento:** `POST /v1/recognize`. **Guia completo:** [OPERACAO.md](OPERACAO.md).

## Identidade da entrega

| Artefato | Identificação |
|---|---|
| Pacote final | `releases/v1-initial-20261117-r3`, 126 arquivos indexados + `RELEASE.json`; 71.513.781 bytes no total |
| SHA-256 de `RELEASE.json` | `ebdced78f7bc3847261794bbe6abf0f8eed9e5a08304e7d9d964dcab15d9e4ac` |
| Imagem API | `sha256:835c7311e32c67c2296b9547d60d3a1627a4e217dfb43e7d6b9050dc32cd7058` |
| Tamanho da imagem API | 4.092.882.670 bytes |
| Detector aprovado | `5a0c93fa86f7249197c3eb83391a2e1717602d1b8409f67bc2d6159a57af1d21` |
| Classificador aprovado | `73e3c50ff1e77bf0c96d546b137e34e27ffac5df5efa84046d0aac3365f9f864` |
| Ambiente | Linux amd64, Python 3.11.17, uv 0.12.23, Torch 2.14.1+cu126, TorchVision 0.29.1+cu126, BentoML 1.4.39 |

O pacote reúne pesos, arquitetura, mapas, pré-processamento, política congelada, aprovações, protocolo/relatório E2, código e lock Linux. O dataset contribui somente com o mapa de rótulos; imagens de treino, auditoria, optimizer states e credenciais ficam fora do pacote. A imagem executa com usuário `10001:10001`, raiz somente leitura, volume de estado gravável e secrets montados somente para leitura. Não há bind mount de pastas de treinamento, código do host ou auditoria. O manifesto é verificado antes de iniciar o BentoML e novamente ao carregar os modelos.

Python e uv foram fixados por digest; a ativação usa o ID imutável da imagem. O empacotador recusa sobrescrever um release, e o gerenciador recusa reconstruir um release que já concluiu o build. A revisão `r2`, com os mesmos pesos, foi conservada para a conferência real de rollback.

## Execução conferida

1. Compose CPU e override CUDA passaram em `docker compose config --quiet`.
2. A revisão `r2` iniciou em CPU sem reserva de GPU e passou no healthcheck; requisições retornaram `200 ok` e `200 needs_review`.
3. A revisão final `r3` iniciou com reserva de uma GPU NVIDIA, identificou `cuda:0`/RTX 2060 e respondeu às mesmas imagens. Tipos, valores e decisões de aceite foram iguais aos exemplos CPU, com diferenças numéricas de caixas de até 0,00003052 pixel. [Observações](../../evidence/stage7-20261117/cpu-cuda-observations.json).
4. O comando `rollback` restaurou a imagem exata `sha256:1ac69a3c8c00e1f524da6889e5ab54f07ad7451a78518c17ed5bb66e96f35a82` da revisão `r2`, em CPU. O container ficou saudável e respondeu novamente `200 ok`. [Registro](../../evidence/stage7-20261117/rollback.json).
5. A revisão final `r3` foi reativada em CPU. Conferidos `200 ok`, `404 no_candidates`, `422 too_many_dice` e JPEG com EXIF (`200 needs_review`). [Estado final](../../evidence/stage7-20261117/final-activation.json), [container](../../evidence/stage7-20261117/final-container.json) e [OpenAPI servido](../../evidence/stage7-20261117/openapi.json).

O serviço faz warmup dos dois componentes antes de concluir a inicialização. O healthcheck verifica readiness HTTP, identidade do pacote e acesso ao bucket. CPU e GPU usam a mesma imagem e os mesmos pesos/política; o [override NVIDIA](https://docs.docker.com/compose/how-tos/gpu-support/) apenas configura o dispositivo e a reserva.

## Persistência e recuperação

MinIO publicado em `127.0.0.1:29000`, console em `127.0.0.1:29001`, bucket `dice-originals-v1` e volume `dice-v1_originals`. Foram conferidas ausência de política pública e de lifecycle de expiração. [Inspeção final](../../evidence/stage7-20261117/bucket-final.json). As credenciais ficam em `runtime/v1/secrets`, montadas como [secrets do Compose](https://docs.docker.com/compose/how-tos/use-secrets/).

Quatro originais enviados em CPU/CUDA foram recuperados após recriações dos containers e rollback. Todos mantiveram bytes e SHA-256 exatos. [Inspeção](../../evidence/stage7-20261117/persistence-after-rollback.json). A instância anterior da Etapa 5 e seus dados foram preservados separadamente.

Arquivos de recuperação criados em `runtime/v1/recovery/v1-initial-20261117-r3`:

| Arquivo | Bytes |
|---|---:|
| `api.tar` | 4.092.903.936 |
| `minio.tar` | 99.887.104 |
| `bundle.tar.gz` | 59.921.294 |

[Índice de hashes e IDs](../../evidence/stage7-20261117/recovery-index.json). A leitura de `minio.tar` foi conferida com `docker image load`, que recuperou o mesmo ID `sha256:a3475aaa2694039238a5728913f45f58c45539012c74228796408fb8d9147ba6`, sem alterar containers ou volumes. O guia descreve a recarga das imagens e a conservação dos volumes, secrets e histórico. Essa conservação é necessária para recuperar os uploads.

## Tempos para comparação

| Operação | Duração monotônica | Warmup incluído |
|---|---:|---:|
| Primeira tentativa, interrompida por TLS | 27,03 s | — |
| Build `r2`, com downloads de dependências | 865,72 s — 14min25,72s | — |
| Build final `r3`, com cache | 212,16 s — 3min32,16s | — |
| Ativação CPU `r2` até saudável | 31,31 s | 0,268 s |
| Ativação CUDA `r3` até saudável | 34,37 s | 2,106 s |
| Rollback CPU `r2` até saudável | 29,70 s | 0,219 s |
| Ativação CPU final `r3` até saudável | 28,92 s | 0,227 s |

[CSV de tempos](../../evidence/stage7-20261117/TEMPOS_OPERACAO.csv), [histórico de ativações](../../evidence/stage7-20261117/activation-history.jsonl), registros JSON e logs de build acompanham esta entrega. O host já possuía cache Docker; o primeiro build concluído incluiu os downloads dos pacotes Linux. O build final reutilizou as dependências. Não comparar esses dois builds como execuções com condições iguais.

Exemplos HTTP antes da exportação de imagens: CPU 273,44/250,19 ms e CUDA 300,28/198,89 ms para as duas imagens de conferência. São observações funcionais de poucas requisições, com corpos e custos distintos, sem benchmark de carga ou conclusão de superioridade de dispositivo. O `total` inclui decodificação, persistência, adaptação PNG sem perda e inferência. Os tempos de treinamento continuam em **5.094,24 s — 1h24min54,24s**, conforme a referência E1; nenhum treinamento foi executado nesta etapa.

Os relógios UTC do host e da VM Docker apresentaram defasagem aproximada de 16 segundos. Durações foram medidas com relógio monotônico em cada ambiente; não foram calculadas subtraindo carimbos entre host e container. Warmup faz parte da ativação e não deve ser somado novamente.

## Revisão, limitações e encerramento

Houve revisão independente somente leitura do empacotamento e da operação. Foram corrigidos a recuperação da última implantação saudável após falha de ativação e o uso da identidade histórica exata no rollback. A revisão por tag do uv foi fixada também por digest, e a imagem MinIO local foi exportada para recuperação. [Registro da revisão](../../evidence/stage7-20261117/code-review.json).

O primeiro build falhou ao validar o certificado do proxy. O gerenciador passou a montar somente a cadeia pública confiada pelo Windows como secret temporário de build, mantendo verificação TLS e hashes do lock, conforme o [suporte de certificados do uv](https://docs.astral.sh/uv/concepts/authentication/certificates/). A cadeia não entra na imagem. Nenhuma dependência foi instalada de forma dinâmica durante as requisições.

Observação menor: o Ultralytics informou fallback de sua pasta de configurações para `/tmp/Ultralytics`; essa pasta é temporária. Os mapas, pesos, transformações e parâmetros efetivos permanecem no pacote verificado, e o estado operacional fica no volume `/state`. Não houve falha funcional associada a esse aviso.

Conferências realizadas: builds reais, validação Compose, compilação Python, leitura de código, inicialização/healthcheck, HTTP, S3, troca CPU/CUDA, rollback e recuperação da imagem MinIO. Não foram criados ou executados testes automatizados nem comandos Git de mutação. Os checkpoints aprovados e os 58 arquivos da entrega E2 mantiveram seus hashes; [conferência final](../../evidence/stage7-20261117/frozen-inputs-after.json).

As entregas técnicas do plano estão concluídas. Resta a confirmação final do usuário, conforme a parada da Etapa 7. O modelo permanece aprovado como versão inicial, com as limitações científicas E2 já registradas e retreino manual futuro previsto.
