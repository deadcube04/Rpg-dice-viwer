# Resultados da primeira entrega local

Avaliação concluída em 05/10/2026. Código, dados e artefatos ficam em `dice-viwer`; a API local usa a imagem `dice-viewer-mlops:4i3hx46a625w2aa2` e o modelo `dice_bundle:s7tgkawa62epoaa2`.

## Dados e protocolo

- Fonte exclusiva: dieCamera, manifesto SHA-256 `85f0af97b99f89d24baf8021b79451f1fd2743049d6fa52b3ab5747d8cbcc44c`.
- Recortes confiáveis: 655 treino, 84 validação, 134 teste. O split por foto tem SHA-256 `136bb9c368866e362152af278767772ad6570cacabab33d7886a49b133688b5d`.
- Detector: YOLO11n, parada antecipada após 65 épocas. Nas 52 fotos de validação, mAP@0,5 = 0,989 e mAP@0,5:0,95 = 0,814.
- Leitores comparados na mesma validação. Ambos acertaram 1 de 7 fotos com um dado. Nos 84 dados, EfficientNet-B0 acertou 62 (73,8%) e MobileNetV3 Small, 39 (46,4%). O desempate documentado favoreceu EfficientNet-B0 antes da latência. Esse protocolo foi congelado antes do teste reservado.
- A validação não ofereceu evidência suficiente de falso aceite inferior a 5%. O limiar 1,01 encaminha todas as predições para revisão. A taxa real de falso aceite é **indefinida**, pois houve zero aceites.

## Teste reservado

| Medida | Resultado |
| --- | ---: |
| Fotos com um dado, tipo e face corretos | 4/31 = 12,9% (IC Wilson 95%: 5,1–28,9%) |
| Dados individuais, tipo e face corretos | 21/134 = 15,7% (IC Wilson 95%: 10,5–22,8%) |
| Detector, revocação com IoU ≥0,5 | 97,8% |
| Leitor sobre recortes corretos | 23/134 = 17,2% |
| D6 | 8/26 |
| D8 | 5/25 |
| D10 | 0/14 |
| D12 | 3/11 |
| D20 | 5/58 |
| Aceitação automática | 0; cobertura 0% |

O conjunto contém 31 fotos de um dado, 34 de vários dados e 1 sem dado. Há dois D10 com face `0` normalizada para valor `10`; nenhum foi lido corretamente. O detector encontrou zero dados na foto sem dado. Uma foto com dois dados teve apenas um detectado, mas ainda foi enviada à revisão por baixa confiança. As contagens e matrizes de confusão completas estão em `artifacts/evaluation.json`, rastreado pelo DVC.

Há uma mudança forte entre as câmeras de validação e teste: a leitura sobre recortes verdadeiros cai de 77,4% na validação para 17,2% no teste. O detector mantém revocação alta, então o leitor e a mudança visual das câmeras são a principal limitação observada. Esta é uma inferência a partir das métricas, não um diagnóstico causal isolado.

## Serviço e metas

Depois de 30 chamadas de aquecimento e 200 chamadas medidas na imagem com GPU RTX 2060:

| Medida | Resultado |
| --- | ---: |
| Inferência p50 / p95 | 45,3 / 68,6 ms |
| Chamada HTTP p50 / p95 | 83,7 / 130,8 ms |

| Meta do trabalho | Estado |
| --- | --- |
| Acerto conjunto ≥95% | **Não atingida**: 12,9% por foto com um dado |
| Falso aceite <5% | **Não comprovada**: zero aceites, taxa indefinida |
| p95 de inferência ≤1 s | **Atingida**: 68,6 ms no contêiner local |
| Redução ≥50% no tempo mediano | **Pendente**: requer estudo humano com leitura e registro manual e via API nas mesmas fotos |
| Teste em dataset independente | **Não atendido**: as câmeras de teste pertencem ao dieCamera |

A API retornou revisão para fotos com zero e vários dados detectados e armazenou uma correção humana. Após reiniciar o contêiner, o SQLite ainda continha a correção, vinculada à versão do modelo, e a imagem correspondente permanecia no MinIO. O rollback para uma versão anterior não pôde ser exercitado porque esta é a primeira imagem promovida; seu digest já está em `artifacts/releases.jsonl` para a próxima promoção.

Em uma checagem operacional adicional, um D4 fora do escopo recebeu uma leitura forçada como D8, porém com decisão `review_required`. Isso mostra que o leitor não identifica explicitamente tipos desconhecidos; a segurança atual depende do limiar sem aceites. Não havia exemplos rotulados como ilegíveis no conjunto reservado para medir esse caso separadamente.

Não é adequado ativar aceitação automática com esses resultados. O serviço está disponível para revisão humana e coleta de feedback, sem retreino automático.
