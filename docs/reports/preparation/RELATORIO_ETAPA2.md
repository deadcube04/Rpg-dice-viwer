# Etapa 2 — dataset preparado e aprovado

**Data:** 17/11/2026. **Estado:** dataset e limitações aprovados pelo usuário; Etapa 3 executada. O treinamento continua aguardando aprovação da receita da Etapa 3.

## Aprovação e versão

O usuário aprovou o snapshot de auditoria `20261117T160833997109Z`, autorizou a Etapa 2 e, após a apresentação dos resultados, aprovou o dataset preparado e suas limitações.

| Identificador | Valor |
|---|---|
| Manifesto de revisão — SHA-256 | `7b23477513dedc0d25de0740177b3f453055656662c5ad216c7e8e5a26166763` |
| Índice do dataset preparado — SHA-256 | `5ecbe03114abb68790e2319e6e7b26f745c422735d48895a86d2761b6c0d1a73` |
| Dataset | `data/datasets/v1-20261117-7b23477513de` |
| Aprovação posterior | [dataset-approval.json](../../evidence/additional/data/preparation/dataset-approval.json) |

Os arquivos congelados mantêm seu conteúdo original. A aprovação posterior está registrada separadamente e vinculada ao hash do índice. O campo `approved: false` do snapshot de auditoria descreve o estado na exportação e não é reescrito.

## Partições e formatos

| Destino | Imagens | Recortes do classificador |
|---|---:|---:|
| Treino E1 | 3.846 | 3.846 |
| Validação E1 | 969 | 966 |
| Avaliação final E2 | 518 | 1.927 |
| Negativas separadas | 4 | 0 |
| Quarentena técnica | 324 | — |
| Acima de oito dados | 32 | — |
| Duplicatas exatas removidas dos conjuntos de modelo | 3 | — |
| Fonte temporariamente excluída | 1.219 | — |

A validação corresponde a **20,12%** das imagens E1 exportadas. A divisão agrupada usou seed 42; a integridade dos grupos prevalece sobre o balanceamento por classe. Os três exemplos de validação sem alvos utilizáveis de tipo ou valor continuam disponíveis ao detector, sem recortes de classificação.

Foram gerados manifesto canônico, mapas, partições, caixas YOLO de classe única `die` e recortes com máscaras independentes. E2 contém cenas de dois a oito dados e permanece reservada à avaliação final. Recortes de cenas de grupo não entram no treino E1.

## Agrupamento e quarentena

Os critérios incluem igualdade de arquivos e pixels, famílias de aumentos e sequências inferidas pelos nomes, sessões inferidas com intervalo adjacente de até 30 minutos e candidatos perceptuais a quase duplicatas. Esses candidatos são mantidos juntos conservadoramente; não foram apresentados como duplicatas semanticamente confirmadas.

| Motivo da quarentena | Imagens |
|---|---:|
| Anotações conflitantes em pixels idênticos | 76 |
| Imagens individuais relacionadas a grupos reservados para E2 | 156 |
| Supervisão incompleta de caixas do corpo | 92 |

`roboflow/d4-turbo-rad-v4` permanece excluída por decisão do usuário, com inventário e revisões preservados. Essa exclusão não remove o tipo d4 do vocabulário.

## Limitações de cobertura aprovadas

- Sem supervisão conjunta no treino E1: **d8:2, d8:5, d10:1, d10:7, d12:1, d12:3 e d12:10**. As mesmas combinações estão ausentes da validação E1.
- A validação não possui suporte conjunto para 22 combinações de tipo e token.
- D4 tem 28 rótulos de tipo e somente 21 exemplos conjuntos no treino; a validação contém três imagens de d4.
- A modalidade permanece desconhecida quando assim anotada; não se presume cobertura de d6 por pontos a partir do tipo.
- Os agrupamentos protegem as relações conhecidas pelos critérios adotados; não demonstram generalização a dados físicos ou coleções inéditas.

## Evidências

Foram conferidos **17.450 arquivos** e **6.739 recortes**, sem sobreposição conhecida entre treino, validação e E2. Quatro recortes foram inspecionados visualmente; isso não constitui uma nova auditoria visual integral.

- [Integridade final da Etapa 2](../../evidence/additional/data/preparation/stage2-verification.json).
- [Nova conferência de integridade antes da Etapa 3](../../evidence/additional/data/preparation/stage3-dataset-integrity.json).
- [Cobertura por tipo e token](../../evidence/additional/data/datasets/v1-20261117-7b23477513de/reports/joint_support.csv).
- [Resumo do dataset](../../evidence/additional/data/datasets/v1-20261117-7b23477513de/reports/summary.json).
- [Critérios e evidências de agrupamento](../../evidence/additional/data/datasets/v1-20261117-7b23477513de/manifests/group_evidence.jsonl).
- [Relatório textual original](../../evidence/additional/data/preparation/RELATORIO_ETAPA2.txt).

A tentativa parcial com sufixo `-interrupted` foi preservada para rastreabilidade e não deve ser usada para treinamento. A versão válida é a indicada na tabela de identificação.
