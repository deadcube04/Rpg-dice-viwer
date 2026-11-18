# Revisão de aderência à especificação acadêmica

Esta revisão complementa as sete etapas; os relatórios incrementais descrevem
o estado observado em cada etapa e não foram reescritos retroativamente.

## Correções após a auditoria

- README autossuficiente: pré-requisitos verificáveis, subida, primeiro envio,
  dois terminais, Get-Content -Wait, carga com seed, resultados esperados,
  interpretação básica de métricas, testes e parada sem apagar registros.
- ENTENDIMENTO_DO_NEGOCIO.md: contextualização CRISP-DM atual e matriz que
  justifica cada variável. Não foi inventado um CRISP-DM histórico inexistente.
- Logging externo: extensão service_class do BentoML envolve o ASGI completo;
  admissão 429, timeout 504 e erro 500 também são registrados uma vez.
  Controle de tráfego, rotas, corpos e códigos HTTP não foram alterados.
- http_observation_version=2 identifica o novo escopo de tempo; registros
  antigos sem esse campo permanecem legíveis e não são reescritos.
- A integração agora exige contagem exata de todos os requests, inclusive
  rejeições sem ID público; não descarta a exigência de JSONL para 429.
- Guia inclui exemplo numérico de buckets, média e taxa de erro e explica
  por que boxplot exato requer observações individuais.

## Verificação e entrega

Testes específicos exercitam as camadas reais BentoML sem carregar pesos:
429, 504 e 500 são preservados e geram um registro. Testes normais verificam
que a instrumentação não duplica requests/dados nem expõe texto de exceções.
As evidências e o manifesto de hashes são atualizados para os fontes revisados.
O JSONL ativo continua no runtime; a captura do professor fica em docs/evidence.

Código, documentação e evidências precisam ser incorporados ao Git para
chegarem num clone. Arquivo não ignorado não significa arquivo commitado.
Esta revisão prepara um conjunto concreto para versionamento; commit/push
continuam sujeitos à autorização explícita previamente exigida pelo usuário.

## Limites que permanecem

Não há como registrar como pedido de inferência um erro de startup ou de
protocolo que nunca entrou no ASGI. Falha do próprio disco pode perder eventos,
conforme documentado. O ID interno de um 429 não é um ID público inventado.
O framework ainda mantém séries internas por caminho, embora a exposição seja
filtrada. O texto atende ao requisito de análise; planilha é opcional e não
foi apresentada como requisito obrigatório ou como artefato inexistente.

Roteiro de avaliação: [README](../README.md), [negócio](ENTENDIMENTO_DO_NEGOCIO.md)
e [guia completo](OBSERVABILIDADE.md).
