# Operação local da V1 inicial

Execute os comandos em `C:\MLops-dice-viwer\dice-viwer`, com Docker Desktop disponível. A versão técnica final é `v1-initial-20261117-r3`; seus pesos são os do modelo inicial aprovado `v1-e1-20261117T175200Z`. O retreino será uma execução manual futura, com novos dados/receita e aprovação próprios.

## Iniciar em CPU

```cmd
.venv\Scripts\python.exe deploy\v1\manage.py activate v1-initial-20261117-r3
```

Essa operação usa o ID imutável da imagem construída, verifica o manifesto completo e aguarda o healthcheck. O padrão usa CPU, um worker e uma inferência por vez. Os dois modelos recebem warmup antes de o serviço ficar pronto. Nenhuma instalação de driver NVIDIA é necessária para o modo CPU.

Endereços publicados somente em loopback:

| Serviço | Endereço |
|---|---|
| API e documentação interativa | http://127.0.0.1:23000 |
| Reconhecimento | `POST http://127.0.0.1:23000/v1/recognize` |
| Readiness BentoML | http://127.0.0.1:23000/readyz |
| OpenAPI | http://127.0.0.1:23000/docs.json |
| MinIO S3 | http://127.0.0.1:29000 |
| Console MinIO | http://127.0.0.1:29001 |

A instância de conferência da Etapa 5 usa portas e volume próprios; seus originais permanecem preservados. A aplicação consumidora deve usar a porta 23000 para a versão empacotada.

## Enviar uma imagem

Na documentação interativa, escolha `image/png` para PNG ou `image/jpeg` para JPEG no seletor do corpo da requisição. Envie o arquivo binário direto. O tipo de conteúdo deve corresponder aos bytes reais; a extensão do nome não altera o formato do arquivo.

```cmd
curl.exe -X POST http://127.0.0.1:23000/v1/recognize -H "Content-Type: image/png" --data-binary @runtime/stage5/needs_review.png
```

O serviço valida, persiste o original e então executa a inferência. O contrato e os limites da Etapa 5 continuam vigentes. `404 no_candidates` mantém a imagem salva; excesso de candidatos também preserva a referência confirmada. Scores e aceite automático mantêm as limitações medidas em E2.

## Usar NVIDIA/CUDA

Com o suporte NVIDIA do Docker/WSL2 disponível:

```cmd
.venv\Scripts\python.exe deploy\v1\manage.py activate v1-initial-20261117-r3 --cuda
```

O override reserva uma GPU. O gerenciador só registra sucesso se o serviço estiver saudável e o dispositivo efetivo for CUDA. A troca substitui a implantação do serviço completo, preservando volumes, credenciais e modelo aprovado. Para voltar a CPU, execute novamente o comando de ativação sem `--cuda`.

## Conferir operação e parar

```cmd
.venv\Scripts\python.exe deploy\v1\manage.py status
curl.exe -i http://127.0.0.1:23000/readyz
docker ps
docker logs --tail 100 dice-v1-api-1
```

`status` apresenta o registro da última ativação saudável. O estado atual do container aparece em `docker ps`; o healthcheck Docker combina readiness HTTP, identidade da versão e acesso ao bucket. Os logs têm referências de requisição/imagem, versão e dispositivo, sem os bytes dos uploads ou credenciais.

Para parar os serviços mantendo os volumes:

```cmd
docker stop dice-v1-api-1 dice-v1-minio-1
```

Reinicie usando o comando `activate` correspondente ao modo desejado.

## Retornar à implantação anterior

```cmd
.venv\Scripts\python.exe deploy\v1\manage.py rollback
```

O retorno usa o `image_id`, SHA do manifesto e modo CPU/CUDA históricos. Se uma ativação falhar após alterar containers, a recuperação usa a última implantação saudável registrada em `pending.json`. Uma ativação bem-sucedida limpa essa pendência e preserva a implantação anterior em `active.json`.

Na primeira implantação, sem histórico saudável anterior, não existe alvo anterior. A versão inicial deve ser conservada para recuperação e comparação futura. As revisões `r2` e `r3` usam os mesmos pesos; uma troca entre elas verifica o retorno da imagem/configuração inteira, sem representar comparação entre modelos retreinados.

Também é possível reativar explicitamente uma versão já construída e aprovada:

```cmd
.venv\Scripts\python.exe deploy\v1\manage.py activate v1-initial-20261117-r3
```

Não substitua arquivos dentro de uma pasta de release. O empacotador recusa diretórios existentes, e o gerenciador recusa reconstruir um release já concluído. Uma mudança exige novo identificador e manifesto.

## Persistência e recuperação das imagens Docker

Os originais ficam no bucket privado `dice-originals-v1`, volume `dice-v1_originals`, sem lifecycle de expiração. O volume `dice-v1_service-state` guarda o estado de inicialização. Credenciais locais ficam nos arquivos de `runtime/v1/secrets`, montados como secrets; seus valores não entram na imagem, no manifesto ou nos registros de implantação.

Preservar uma versão construída:

```cmd
.venv\Scripts\python.exe deploy\v1\manage.py preserve v1-initial-20261117-r3
```

Os arquivos ficam em `runtime/v1/recovery/v1-initial-20261117-r3`: `api.tar`, `minio.tar`, `bundle.tar.gz` e `RECOVERY.json`, com hashes e IDs. A exportação desta versão faz parte da entrega; o comando recusa sobrescrever um diretório de recuperação existente. A imagem MinIO tem origem local; esse arquivo permite recarregá-la sem depender de um nome no Docker Hub.

Para recarregar as imagens no Docker local:

```cmd
docker image load -i runtime\v1\recovery\v1-initial-20261117-r3\api.tar
docker image load -i runtime\v1\recovery\v1-initial-20261117-r3\minio.tar
.venv\Scripts\python.exe deploy\v1\manage.py activate v1-initial-20261117-r3
```

Conserve também a pasta `releases`, os registros e secrets de `runtime/v1`, e os volumes Docker. A recuperação dos uploads depende desses volumes e das credenciais. Os arquivos `api.tar` e `minio.tar` preservam as imagens de software; os dados dos volumes têm ciclo de backup próprio.

## Build reproduzível e tempos

O lock Linux fica em `deploy/v1/environment/uv.lock`, com as mesmas versões principais avaliadas e PyTorch/TorchVision CUDA 12.6. A imagem CPU também usa esses wheels, funcionando sem reservar GPU. Python, uv e imagens Docker são identificados por versão/digest ou ID imutável.

O build em Windows exporta somente certificados públicos de confiança do sistema e monta essa cadeia temporariamente como secret do BuildKit, para validação TLS do proxy. A cadeia não entra no pacote ou na imagem resultante. A validação dos hashes dos pacotes permanece ativa.

Tempos de build ficam em `runtime/v1/builds/*.json`; ativações e warmups em `runtime/v1/history.jsonl` e `active.json`. As durações usam relógios monotônicos. Os carimbos UTC do host e da VM Docker apresentaram defasagem de aproximadamente 16 segundos nesta execução; use os campos de duração em vez de subtrair carimbos entre esses ambientes. Os tempos de treinamento continuam em `reports/training/v1-e1-20261117T175200Z`; relatórios E2 e de serviço têm protocolos próprios e devem ser comparados separadamente.
