# Evidências para avaliação — Observabilidade RPG Dice

Esta pasta acompanha a entrega e é versionável. runtime continua ignorado
porque contém estado operacional e credenciais; as capturas abaixo não dependem
dele para serem lidas pelo professor. Nenhum arquivo desta pasta é log ativo.

| Arquivo/diretório | Evidência |
| --- | --- |
| recognition.jsonl | Captura do histórico instrumentado de desenvolvimento; inclui resultados por dado, erros e características quando disponíveis |
| recognition-final.jsonl | Captura da implantação portátil final, com imagens sintéticas e conferências integradas |
| metrics.prom | Exposição Prometheus filtrada da implantação final |
| load-sequential/ | Seis tentativas medidas, seed 42, concorrência 1, com imagem de dado e imagem vazia; mais um warmup |
| load-concurrent/ | Mesmo plano de entradas, concorrência 3; inclui rejeições de capacidade observadas |
| load-final/ | Vinte tentativas medidas, seed 42, concorrência 1, usando somente imagens sintéticas; mais um warmup |
| input-manifest.json e inputs/ | Imagens sintéticas pretas/brancas para reproduzir o fluxo sem dataset privado; não são imagens extraídas dos logs |
| latency-final.png | Histograma e boxplot das 20 latências medidas de load-final, sem warmup |
| validation.json | Resultados da validação, escopo e estatísticas da captura final |
| manifest.json | SHA-256 das evidências e fontes selecionados; identifica a release científica usada |

JSONL usa um objeto por linha. Comece por event=request; associe event=die por
request_id/image_id. Não conte worker_completion como nova requisição. Algumas
linhas históricas anteriores à Etapa 3 não têm image_features; não preencher
ausências com zero. As tentativas do cliente incluem a fase para excluir warmup.

Os arquivos de carga guardam hashes, tipos e tamanhos, sem caminhos de imagens.
metrics.prom é acumulado do processo no instante do scrape. O JSONL portátil
também contém requisições anteriores ao reinício; seus totais globais podem
diferir dos contadores reiniciados. Os runs e as chamadas integradas devem
ser reconciliados pelos IDs e pela janela de execução, não por igualdade
entre todos os registros históricos e um contador de um único processo.
As duas primeiras cargas usaram um recorte local: ele não é incluído aqui e
suas classificações não podem ser reproduzidas sem os mesmos bytes. A carga
final usa somente os exemplos sintéticos incluídos. O corpo de imagens,
credenciais, tokens e mensagens arbitrárias de dependências não são capturados.
As capturas não são amostra aleatória de produção nem avaliação de acurácia.

Os textos foram preservados em UTF-8 com finais LF na cópia de entrega para
manter os hashes em clones Windows/Linux. Registros do runtime não foram
reescritos. O manifesto detecta alterações; não é assinatura de autenticidade.

Verificar as evidências no PowerShell, com Python padrão:

```powershell
@'
import hashlib, json
from pathlib import Path
root = Path('docs/evidence/observability')
manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
for name, expected in manifest['evidence_files'].items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest() == expected, name
print('Hashes das evidencias: OK')
'@ | python -
```

Esperado: Hashes das evidencias: OK. O guia de
[observabilidade](../../OBSERVABILIDADE.md) apresenta os comandos de execução,
fórmulas, populações, denominadores e limitações. A validação manual final do
usuário continua necessária antes de considerar a entrega aprovada.
