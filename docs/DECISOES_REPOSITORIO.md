# Decisões da revisão final do repositório

## Documentação

`docs` é o diretório canônico de consulta. O README da raiz funciona como índice. Planos antes externos ao Git foram incorporados em `docs/plans`; os caminhos externos mantêm redirecionamentos para não quebrar links pessoais. A operação da auditoria fica em `docs/audit`. Relatórios e evidências citadas ficam em `docs/reports` e `docs/evidence`.

Os originais experimentais em `reports`, `data` e `releases` são preservados e ignorados pelo Git. Movê-los invalidaria referências e hashes aprovados. Cópias de consulta têm links relativos atualizados, sem reescrever resultados. A skill da auditoria continua como recurso operacional em seu diretório.

## Limpeza

O código da demonstração anterior, seu bundle, Dockerfile e Composes antigos foram retirados da árvore ativa e preservados em `runtime/cleanup/legacy`. Não eram usados pela cascata aprovada nem pelo serviço V1. O inventário está em `runtime/cleanup/inventory.json`. Os arquivos já removidos da antiga pasta `audit-ui` pertencem à reorganização anterior; a interface atual está em `audit/audit-ui`.

O entry point compatível `service.py` aponta para o serviço V1; o projeto BentoML aponta para `recognition_service:DiceRecognitionService`. Os comandos de empacotamento e implantação atuais ficam em `deploy/v1` e `tools/model.py`.

Dados, backups, exports de revisão, checkpoints de treino, releases congelados e arquivos de recuperação não são apagados nesta limpeza. Os ignores removem esses artefatos, ambientes, node_modules, builds, caches, logs e credenciais do conjunto proposto para Git/Docker. Arquivos já rastreados continuam dependendo de um commit do usuário para registrar a remoção; alterar gitignore não os remove do histórico.

## Dump e reprodução

Distribuímos o pacote aprovado completo, sem reconverter os pesos ou alterar seu manifesto. O arquivo de 59.921.294 bytes foi dividido em três partes, de no máximo 24 MiB, adequadas ao Git comum e ao limite de upload pelo navegador. [Limites oficiais do GitHub](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github).

O restaurador usa somente Python padrão, valida tamanhos e hashes, recusa links e caminhos inseguros no tar e confere o manifesto antes de publicar o diretório restaurado. A operação Docker usa os fontes dentro da versão restaurada; não depende de caminhos da máquina de treinamento.

A imagem MinIO local da implantação inicial não existe em registry público. Para novos clones, o launcher obtém uma imagem pública MinIO por digest e injeta seu ID no Compose. Uma implantação portátil tem projeto, portas, secrets e volumes próprios, preservando a instância já aprovada.

O modelo, as métricas E1/E2, tempos de treinamento e aprovações permanecem preservados. Retreino posterior será manual, com nova versão de dados, receita e aprovação. Não há aprendizagem automática a partir de uploads.

## Escopo da verificação

Compilação Python, inspeção do conjunto candidato ao Git, links locais, hashes do dump, restauração em cópia isolada e conferência operacional. Não criar ou executar testes automatizados. Não executar `git add`, commit, push ou mudanças de índice/histórico. A preparação deixa os arquivos revisáveis para o usuário versionar.
