# Entendimento do negócio — RPG Dice Viewer

Este documento registra a justificativa da observabilidade com base na
[arquitetura existente](plans/ARQUITETURA_V1_RECONHECIMENTO_DADOS_RPG.md) e no
contrato implementado. O arquivo não existia antes desta revisão: não representa
aprovação histórica nem uma nova avaliação científica do modelo.

## Problema e decisão apoiada

O serviço recebe uma fotografia estática de dados físicos de RPG e devolve
detecções, tipos, valores e decisões de revisão. Ele apoia a leitura dos dados
e explicita quando o resultado precisa de verificação humana. A observabilidade
permite investigar condições da captura, incerteza e indisponibilidade; não
troca limiares, aumenta confiança ou decide retreinar.

O contrato admite até oito resultados e tipos d4, d6, d8, d10, d10_tens, d12
e d20. HTTP 404/no_candidates é resultado válido do domínio; 200/needs_review
também não é falha técnica. Scores do modelo inicial não são probabilidades
calibradas. Não foi inventado um SLA de produção sem evidência de negócio.

## Relação com CRISP-DM

| Etapa | Decisão nesta entrega |
| --- | --- |
| Entendimento do negócio | Apoiar leitura de dados, revisão humana e investigação de falhas |
| Entendimento dos dados | Registrar condições de captura, complexidade e resultados para estudar distribuições |
| Preparação | Usar RGB já orientado pelo decoder; extrair características de amostra separada |
| Modelagem | Preservar detector, classificador, pesos e política aprovados, sem treino novo |
| Avaliação | Conferir rastreabilidade e integridade; carga sintética não mede acurácia ou drift |
| Implantação | CPU por padrão, JSONL local, retenção, métricas e carga repetível |

## Escolha das variáveis

| Variável | Entrada/saída | Justificativa e análise futura |
| --- | --- | --- |
| Timestamp UTC, request_id, image_id | Contexto | Relacionar imagem persistida, pedido e resposta; agrupar janelas |
| Bytes, dimensões, pixels e proporção | Entrada | Captura e tamanho afetam custo e condições do reconhecimento; histogramas/boxplots por versão |
| Brilho e contraste | Entrada | Capturas escuras, claras ou com pouco contraste podem dificultar a leitura das inscrições |
| Médias/desvios RGB | Entrada | Caracterizar variações de cor, fundo e material sem registrar pixels |
| Variância do Laplaciano | Entrada | Descrever detalhe/nitidez; não é score calibrado de qualidade ou aceite |
| Contagem e bbox | Saída | Caracterizar quantidade, escala e disposição dos candidatos e sua relação com custo |
| Shape, die_type e face_value | Saída | Estudar composição das previsões; previsão não equivale a rótulo verdadeiro |
| Confianças de detecção/tipo/valor | Saída | Observar fontes distintas de incerteza, sem tratá-las como probabilidades calibradas |
| Aceite automático e motivos de revisão | Saída | Medir cobertura de aceite e causas de intervenção humana sob política fixa |
| HTTP, error_code e response_completed | Operação | Separar resultado de domínio, entrada inválida, saturação e falha técnica |
| Tempos das etapas, total, HTTP e cliente | Operação | Localizar custo sem confundir escopos de medição |
| Versões, device e feature version | Contexto | Comparar regimes compatíveis e identificar alterações de implementação |

Variáveis numéricas permitem histogramas e boxplots. Tipos, motivos e códigos
pedem frequências categóricas. face_value deve ser analisado por die_type:
misturar d4 e d20 pode alterar a distribuição sem mudança na qualidade.
Dados da mesma imagem e reenvios são dependentes, não amostras independentes.

## Formato e referência futura

JSONL permite eventos independentes por linha, append-only e leitura durante
a carga. request é a unidade de pedido; die é ligado ao pedido;
worker_completion registra término tardio sem duplicar a requisição.
Características possuem versão e estado de ausência distinto de zero.
Nitidez depende da reamostragem: não comparar versões diferentes diretamente.

Esta entrega salva os dados para KS/PSI futuros, sem implementá-los. Uma
referência poderá vir do conjunto científico de treino identificado pelos
manifestos ou de um conjunto manual independente, cuja composição/período
precisarão ser registrados. Os exemplos sintéticos testam o software e não
foram declarados referência representativa de produção ou treino.

Distribuição diferente não prova concept drift ou queda de acurácia. Afirmar
qualidade preditiva exige rótulos independentes e protocolo adequado. Não
escolher limiares, modelos ou bins retrospectivamente para favorecer resultados.

## Privacidade e escopo

O JSONL não guarda imagem, base64, EXIF, nome do cliente, query, cabeçalhos,
credenciais ou mensagens arbitrárias de erro. Guarda descritores numéricos
e saídas permitidas. A persistência privada do upload original em MinIO já
fazia parte da inferência, não é armazenamento novo de observabilidade.

O domínio é visão computacional de dados de RPG. Tokens de LLM, prompts e
sinais clínicos não existem no contrato e não foram inventados para preencher
exemplos de outro domínio. Não há retreino, dashboards ou alertas automáticos.

Definições e comandos: [guia de observabilidade](OBSERVABILIDADE.md).
