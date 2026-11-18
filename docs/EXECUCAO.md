# Executar o modelo aprovado

Para a aplicação com logging, características de imagens, Prometheus e carga,
use [o guia de observabilidade](OBSERVABILIDADE.md) e `just observe`. Os comandos
`just up`/`tools/model.py run` abaixo preservam a execução da release original.

## Atalhos com Just

Com Just instalado, execute as receitas na raiz do repositório. `just` lista os comandos. O launcher Python continua disponível separadamente.

```cmd
just restore
just up dice-experimento 24000
just health 24000
just recognize "C:/imagens/dado.png" 24000 image/png
just status dice-experimento
just logs dice-experimento
just stop dice-experimento
```

`just cuda` inicia com NVIDIA/CUDA; `just cpu` equivale a `just up`. Preserve o mesmo projeto e porta usados na subida. `status`, `logs` e `stop` usam o estado criado pelo launcher portátil; `stop` preserva volumes e credenciais. Para JPEG, use `image/jpeg` como terceiro argumento de `recognize`. A resposta inclui o código HTTP e o corpo, inclusive para `404 no_candidates`.

O padrão é projeto `dice-model` e porta `23000`. Se a implantação original já ocupar essa porta, utilize uma porta livre, como no exemplo. É possível selecionar outro interpretador pela variável de ambiente `PYTHON`.

## Launcher Python

Pré-requisitos: Python 3.11.8 ou posterior para o launcher, Docker Desktop/Engine com Compose e containers Linux amd64. O build usa a internet para baixar imagens e dependências fixadas. O host não precisa instalar PyTorch, uv, o dataset ou o ambiente da auditoria. A execução padrão usa CPU.

Na raiz de um clone, sem preparar ambiente Python:

```cmd
python tools/model.py run
```

O comando valida as três partes do dump, restaura `releases/v1-initial-20261117-r3`, verifica os 126 arquivos do pacote, obtém o MinIO público fixado por digest, constrói a imagem e aguarda o healthcheck. O projeto Compose padrão é `dice-model`; as credenciais são geradas sob `runtime/portable/dice-model/secrets` e montadas por arquivo. A primeira construção pode demorar vários minutos e usa aproximadamente 4 GB para a imagem, além do cache e dos dados persistidos.

- Documentação/API: http://127.0.0.1:23000
- MinIO: http://127.0.0.1:23001
- Console MinIO: http://127.0.0.1:23002

Para evitar conflito com uma implantação existente:

```cmd
python tools/model.py run --project dice-experimento --port 24000
```

Para NVIDIA/CUDA, com suporte NVIDIA no Docker:

```cmd
python tools/model.py run --cuda
```

Para somente restaurar e verificar os pesos e configurações, sem Docker:

```cmd
python tools/model.py restore
```

Para conferir uma imagem própria:

```cmd
curl.exe -X POST http://127.0.0.1:23000/v1/recognize -H "Content-Type: image/png" --data-binary "@C:\caminho\imagem.png"
```

Na documentação interativa, selecione `image/png` para PNG e `image/jpeg` para JPEG. O corpo é binário direto. A imagem válida é persistida antes da inferência. O modelo é uma versão inicial com limitações E2 conhecidas; scores não representam probabilidades calibradas. O retreino permanece manual e futuro.

O dump está em `models/v1-initial-20261117-r3`, com hashes de cada parte, do arquivo completo e do manifesto da versão. Os pesos FP32 em safetensors, mapas, política, ambiente e fontes de inferência estão no pacote. As imagens Docker de recuperação local não são necessárias no Git. Nenhum dataset externo ou credencial local é necessário para iniciar a versão restaurada.

O launcher mantém seu estado em `runtime/portable/<projeto>`. Os comandos históricos de `deploy/v1/manage.py` continuam disponíveis para a implantação original e usam `runtime/v1`; são estados distintos. Não troque o projeto de uma implantação existente sem conferir seus volumes e credenciais.
