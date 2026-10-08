# RPG Dice Viewer

Aplicação acadêmica para demonstrar, localmente, a leitura de **um dado de RPG em uma fotografia** usando um modelo de visão computacional já treinado. A aplicação recebe a imagem pelo **Swagger** e apresenta o **tipo do dado** e o **valor previsto**.

O projeto é executado com Docker e utiliza uma GPU NVIDIA. Não é necessário instalar Python, bibliotecas de aprendizado de máquina ou ferramentas de desenvolvimento além das indicadas abaixo.

> **Importante:** o modelo produz previsões, que podem estar incorretas. Esta aplicação foi preparada para demonstração local, não para disponibilização como serviço público.

## 1. Preparação do computador

Este guia considera um computador **Windows 11 de 64 bits (Intel/AMD), versão 23H2 (build 22631) ou mais recente**, com **pelo menos 8 GB de RAM**, processador compatível com SLAT, virtualização habilitada no BIOS/UEFI e **GPU NVIDIA compatível** com as bibliotecas CUDA 12.6 utilizadas pelo contêiner. Esses são requisitos do Docker Desktop com backend WSL2; confira os [requisitos oficiais do Docker Desktop para Windows](https://docs.docker.com/desktop/setup/install/windows-install/). Reserve, como recomendação prática, cerca de **30 GB de espaço livre** para os downloads e a construção da imagem Docker. A primeira instalação e construção precisam de internet e podem baixar vários gigabytes.

### 1.1. Instalar o Git

1. Baixe o [Git para Windows](https://git-scm.com/install/windows).
2. Execute o instalador, mantendo as opções padrão caso não tenha necessidades específicas.
3. Abra o **PowerShell** e verifique:

```powershell
git --version
```

### 1.2. Instalar ou atualizar o driver NVIDIA

1. Baixe o driver correspondente à sua placa no [site oficial da NVIDIA](https://www.nvidia.com/Download/index.aspx).
2. Instale-o e reinicie o computador, se necessário.
3. No PowerShell, confira se a placa é reconhecida:

```powershell
nvidia-smi
```

O comando deve exibir informações da GPU. Não é necessário instalar o CUDA Toolkit no Windows: as bibliotecas usadas pelo modelo ficam na imagem Docker.

### 1.3. Instalar o WSL2

Abra o **PowerShell como administrador** e execute:

```powershell
wsl --install
```

Reinicie o computador quando solicitado e conclua a configuração inicial da distribuição Linux, caso apareça. Depois, abra novamente o PowerShell e execute:

```powershell
wsl --update
wsl --set-default-version 2
wsl --version
```

O Docker Desktop requer **WSL 2.1.5 ou mais recente**. Confira a versão no resultado de `wsl --version`; se estiver desatualizada, execute `wsl --update` e reinicie o computador. Consulte os [requisitos oficiais de WSL2 para o Docker Desktop](https://docs.docker.com/desktop/features/wsl/).

Mais informações: [instalação do WSL pela Microsoft](https://learn.microsoft.com/windows/wsl/install).

> Se o WSL2 não iniciar, verifique se a virtualização está habilitada no BIOS/UEFI do computador.

### 1.4. Instalar o Docker Desktop

1. Baixe e instale o [Docker Desktop para Windows](https://docs.docker.com/desktop/setup/install/windows-install/).
2. Abra o Docker Desktop e aguarde o mecanismo de execução iniciar.
3. Em **Settings → General**, habilite **Use the WSL 2 based engine**.
4. Utilize **Linux containers**, não Windows containers.

Abra o PowerShell e verifique:

```powershell
docker version
docker compose version
```

Em `docker version`, devem aparecer as informações de **Client** e **Server**. Se somente o Client aparecer, confira se o Docker Desktop está aberto e funcionando.

A integração de GPU com Docker Desktop/WSL2 é descrita na [documentação oficial do Docker](https://docs.docker.com/desktop/features/gpu/).

## 2. Baixar e iniciar a aplicação

> **Atenção a quem prepara a apresentação:** os arquivos da implantação precisam estar publicados no repositório remoto antes que o comando `git clone` permita reproduzir a versão descrita neste guia. Este README não confirma que essas alterações já foram publicadas.

Com o Docker Desktop em execução, abra o **PowerShell** na pasta onde deseja guardar o projeto e execute:

```powershell
git clone https://github.com/deadcube04/Rpg-dice-viwer.git
cd Rpg-dice-viwer
docker compose up -d --build
```

Na primeira execução, o Docker fará o download das dependências, construirá a imagem e iniciará o serviço. Não feche o Docker Desktop.

**Não é necessário** criar um arquivo `.env`, importar o modelo manualmente, iniciar o MinIO/MLflow ou instalar Python, PyTorch, BentoML, Node.js e CUDA Toolkit no computador.

### Verificar se a aplicação está pronta

Ainda na pasta do projeto, execute:

```powershell
docker compose ps
```

Aguarde até o serviço `dice-service` aparecer com o estado **healthy**. Para acompanhar a inicialização:

```powershell
docker compose logs -f dice-service
```

Pressione **Ctrl+C** para sair da visualização dos logs. Isso **não** interrompe o serviço.

Também é possível verificar a API com:

```powershell
curl.exe http://localhost:3000/health
```

Quando estiver pronta, a resposta deve indicar `status: ready`, o modelo `dice_bundle:e33njkwb56yiiaa2`, armazenamento local disponível e informações da GPU.

## 3. Enviar uma foto pelo Swagger

Com o serviço em estado **healthy**, abra no navegador:

**http://localhost:3000/docs**

1. Expanda **POST `/v1/predictions` — Enviar foto para o modelo**.
2. Clique em **Try it out**.
3. No campo **file**, escolha uma fotografia do dado.
4. Clique em **Execute**.
5. Na seção **Response body**, observe `die_type` (tipo) e `value` (valor previsto).

Uma resposta de exemplo é:

```json
{
  "die_type": "D20",
  "value": 10
}
```

Neste exemplo, o modelo prevê um dado de **20 faces (D20)** com resultado **10**. Os tipos previstos pela aplicação são **D6, D8, D10, D12 e D20**. O **D4 não faz parte** do escopo. Para o D10, o símbolo zero corresponde ao valor 10.

**Recomendações para a fotografia:**

- Envie uma imagem **JPEG ou PNG**, com tamanho máximo de **10 MiB** e resolução de até **25 megapixels**.
- Fotografe **apenas um dado**, com boa iluminação, nitidez e ocupando boa parte da imagem.
- Evite imagens vazias ou com vários dados. O modo de apresentação classifica a **foto inteira** e não verifica automaticamente a quantidade de dados presentes.

O resultado é uma previsão do modelo existente e não uma garantia de acerto. O Swagger não exige etapa de revisão ou feedback.

### Alternativa: enviar a imagem pelo PowerShell

Se preferir, use o comando abaixo, substituindo o caminho pelo da sua foto:

```powershell
curl.exe -X POST http://localhost:3000/v1/predictions -F "file=@C:\Fotos\dado.jpg"
```

## 4. Parar, iniciar e consultar o serviço

Execute os comandos a seguir **dentro da pasta do repositório**, com o Docker Desktop aberto:

| O que fazer | Comando |
| --- | --- |
| Parar o serviço | `docker compose stop` |
| Iniciar novamente | `docker compose start` |
| Reiniciar o serviço | `docker compose restart dice-service` |
| Ver os últimos logs | `docker compose logs --tail 100 dice-service` |
| Remover o contêiner (sem apagar o volume) | `docker compose down` |
| Criar/iniciar novamente | `docker compose up -d` |
| Reconstruir após atualizar os arquivos | `docker compose up -d --build` |

## 5. Registros e backup

A aplicação usa **armazenamento local** em um volume do Docker chamado `demo-state` (normalmente `dice-viewer-demo_demo-state`). Os registros de previsões, feedback e imagens ficam no volume, incluindo:

- Banco SQLite: `/state/feedback.db`.
- Imagens: `/state/images/predictions/`.

Os registros permanecem após parar, reiniciar ou recriar o contêiner, **desde que o volume não seja removido**. O Git não transfere esses registros entre computadores.

Para copiar os registros para a pasta do projeto, execute:

```powershell
docker compose stop dice-service
docker compose cp dice-service:/state ./registros-apresentacao
docker compose start dice-service
```

A pasta `registros-apresentacao/` é ignorada pelo Git. **Não publique imagens pessoais, bancos de dados, arquivos `.env`, ambientes virtuais, caches ou imagens Docker.**

> **Cuidado:** não execute `docker compose down -v` se quiser preservar o histórico. A opção `-v` remove os volumes associados.

## 6. Solução de problemas

| Problema | Como verificar ou corrigir |
| --- | --- |
| `docker version` mostra apenas Client ou não conecta ao Server | Abra o Docker Desktop, aguarde o engine iniciar e confirme o uso de contêineres Linux. |
| Erro relacionado à GPU NVIDIA | Verifique `nvidia-smi`, atualize o driver, execute `wsl --update` e confirme o backend WSL2 do Docker Desktop. |
| GPU incompatível ou erro de CUDA | Consulte os logs e a compatibilidade da placa com as bibliotecas CUDA 12.6 da imagem. Instalar CUDA Toolkit no Windows não é uma correção automática. |
| Porta 3000 já utilizada | Libere a porta ou altere a publicação no `compose.yaml` para `127.0.0.1:3002:3000`. Nesse caso, abra `http://localhost:3002/docs`. |
| Swagger não abre | Confira `docker compose ps` e aguarde `healthy`; depois consulte `docker compose logs --tail 100 dice-service`. |
| Swagger exibe uma versão antiga | Recarregue `http://localhost:3000/docs` com **Ctrl+F5**. |
| Falha ao baixar dependências ou construir a imagem | Verifique conexão com a internet, proxy, certificados e espaço livre. Não desative a verificação TLS. |
| Bundle ausente ou hash incorreto | Restaure o arquivo original do modelo e reconstrua a imagem, sem substituí-lo por outro modelo. |
| Erro de armazenamento | Verifique espaço disponível e permissões do volume Docker; consulte os logs. |
| HTTP **422** ao enviar a foto | Confira se o campo usado é `file`, se a imagem é JPEG/PNG válido e se respeita os limites de tamanho e resolução. |
| Tipo ou valor previsto incorreto | Experimente outra foto do mesmo dado com enquadramento, iluminação e nitidez melhores. A previsão pode falhar. |

Se o erro persistir, consulte os logs:

```powershell
docker compose logs --tail 100 dice-service
```

## 7. Arquivos necessários no repositório

A implantação descrita depende dos arquivos do projeto, incluindo:

```text
Rpg-dice-viwer/
├── .dockerignore
├── .gitignore
├── Dockerfile
├── compose.yaml
├── compose.mlops.yaml
├── deployment/
│   └── requirements-serving.txt
├── artifacts-next/
│   └── dice_bundle-e33njkwb56yiiaa2.bentomodel
├── service.py
└── src/
    └── dice_viewer/
        ├── deployment.py
        ├── service.py
        ├── storage.py
        ├── operations.py
        └── inference.py
```

Essa listagem destaca os arquivos relevantes e **não representa toda a estrutura do repositório**. Os demais módulos Python necessários também devem permanecer versionados. O arquivo `HANDOFF_README.md` é apenas material de apoio à documentação e não é exigido para executar o serviço.

O modelo empregado é o bundle **`dice_bundle:e33njkwb56yiiaa2`**, armazenado em `artifacts-next/dice_bundle-e33njkwb56yiiaa2.bentomodel`. Durante o build, a integridade do arquivo é verificada e ele é importado para a imagem, sem treinamento ou alteração dos pesos.

## 8. Limitações e configuração opcional

O modo padrão da API usa o leitor treinado diretamente sobre a imagem inteira de um único dado. **Ele não confirma a existência do dado nem conta quantos dados aparecem**. Uma imagem vazia ou com vários dados pode produzir uma previsão mesmo sem ser uma entrada adequada. A precisão geral desse modo de apresentação ainda não foi medida.

O serviço padrão executa apenas `dice-service`, disponível em `127.0.0.1:3000` **na própria máquina**, com Swagger e recursos locais. Depois que a imagem Docker estiver construída e o serviço iniciado, a demonstração pode funcionar sem internet.

A configuração MLOps anterior permanece separada em `compose.mlops.yaml`, com requisitos próprios, e **não é necessária para esta apresentação**. Não execute as duas configurações na mesma porta. Há também um fluxo técnico de detecção e leitura (`?details=true`) usado por ferramentas de estudo; ele não faz parte das etapas do tutorial do Swagger.
