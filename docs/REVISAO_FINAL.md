# Revisão final do repositório — 17/11/2026

O conjunto candidato ao Git tem aproximadamente 65,5 MB, incluindo o dump de 59,9 MB. Nenhum arquivo excede 25 MiB. As partes do modelo são marcadas como binárias em `.gitattributes`. Dados, credenciais, ambientes, caches, builds e imagens Docker permanecem ignorados. Uma busca pelos segredos conhecidos do ambiente local não encontrou correspondências nos arquivos candidatos; isso é uma conferência limitada a esses valores conhecidos.

Doze arquivos da demonstração anterior foram retirados da árvore ativa e arquivados em `runtime/cleanup/legacy`. A configuração antiga de exemplo também foi preservada antes da atualização. Os arquivos da auditoria atual e do treinamento continuam disponíveis. A revisão não apagou dados, backups, exports ou pesos aprovados.

## Conferência a partir dos arquivos versionáveis

Foi materializada uma cópia isolada em `runtime/checkout-review`, contendo código, configuração e dump, sem `.venv`, dataset, checkpoints avulsos, credenciais ou pasta `releases` preexistente. O restaurador foi executado com o Python leve da auditoria, sem instalar bibliotecas ML nesse ambiente.

A restauração validou hashes das três partes, do arquivo completo e dos 126 arquivos indexados no pacote. O manifesto restaurado manteve SHA-256 `ebdced78f7bc3847261794bbe6abf0f8eed9e5a08304e7d9d964dcab15d9e4ac`.

O launcher obteve a imagem pública MinIO por digest, construiu a API usando cache Docker disponível e iniciou o projeto isolado `dice-portable-check` nas portas 24000–24002. O build levou 6,30 s com cache, e a ativação até saudável levou 48,02 s. Esses tempos não representam uma primeira construção sem cache.

Uma requisição real PNG retornou HTTP 200, `status: ok`, tipo `d10_tens` e valor 40, com persistência confirmada antes da inferência. A API aprovada original continuou respondendo readiness 200 na porta 23000. Os containers temporários da conferência foram parados; os volumes e os registros locais foram preservados. Nenhum teste automatizado foi criado ou executado.

O Docker pode gerar um novo ID de imagem em uma reconstrução, mesmo com o pacote verificado idêntico. A cópia isolada gerou `sha256:9dbcfdb64c168a21104e3dd52df14237502cae8946eaf7c60a07722485b6e4ca`; o manifesto e os pesos permaneceram iguais. A implantação original usa seu ID registrado, não a tag mutável.

## Documentação e integridade

Os links locais de `docs` foram conferidos. As evidências referenciadas foram copiadas para `docs/evidence`; relatórios congelados de origem não foram editados. Os 58 arquivos da entrega E2 continuaram com seus hashes originais. O pacote r3 e seus arquivos de recuperação permaneceram intactos.

O modelo permanece aprovado como versão inicial. As limitações E2 e os tempos de treinamento estão preservados para comparação. O retreino posterior continua manual, sem agendamento ou atualização automática de pesos.

Nenhum comando de alteração do índice, commit, push ou histórico Git foi executado. A preparação está pronta para revisão e versionamento pelo usuário. A relação de remoções antigas de `audit-ui` já existia antes desta revisão; a interface atual está em `audit/audit-ui`.
