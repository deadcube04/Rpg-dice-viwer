# Etapa 5 — Persistência, rotação e retenção

Escopo: manter JSONL local no diretório atual; rotacionar segmentos fechados
sem reescrever registros; aplicar prazo e teto de bytes somente nesses segmentos.
Não alterar originais MinIO, contrato, pesos, modelo ou métricas.

Plano: testes de rotação por tamanho/dia, retenção por idade/teto, proteção do
ativo/arquivos alheios/symlinks, persistência ao reabrir e falhas; implementar
armazenamento separado do schema; integrar ao writer; validar e documentar.

Padrões propostos: 10 MiB por segmento, 30 dias e 1 GiB total para arquivos
gerenciados; fsync a cada lote. A limpeza ocorre após cada append bem-sucedido.
Sem tráfego, nenhum agendamento externo é criado e a expiração aguarda a próxima
escrita. Um lote não é dividido entre arquivos; o tamanho por segmento é um
limiar flexível, mas lotes acima do teto total são rejeitados com diagnóstico.

## Política implementada

| Variável no processo da API | Padrão | Regra |
| --- | --- | --- |
| DICE_OBSERVABILITY_DIR | DICE_STATE_DIR/observability | Diretório local do logger |
| DICE_LOG_MAX_BYTES | 10485760 | Limiar por segmento (10 MiB), inteiro positivo |
| DICE_LOG_RETENTION_DAYS | 30 | Prazo em dias corridos, de 1 a 36500 |
| DICE_LOG_MAX_TOTAL_BYTES | 1073741824 | Teto lógico total (1 GiB), pelo menos MAX_BYTES |

Sem DICE_STATE_DIR, o diretório padrão é runtime/observability na raiz.
Variáveis são lidas na criação do writer. Para mudar um contêiner, recrie-o
com a configuração desejada; mudar o ambiente do PowerShell não altera o
processo já iniciado. Configuração inválida emite observability_config_invalid
sem imprimir seu valor e aplica os três limites padrão conjuntamente.

O arquivo ativo continua `recognition.jsonl`. Antes de escrever um lote,
o logger fecha o segmento existente se ele for não vazio e ultrapassaria o
limiar, se a data UTC do último write mudou, ou se a última linha ficou
incompleta. O arquivo fechado recebe nome como:

`recognition.20261118T014753.123456Z.0123456789abcdef0123456789abcdef.jsonl`

O timestamp corresponde ao mtime anterior em UTC; o sufixo é um UUID gerado.
O conteúdo não é alterado. O novo lote inteiro vai para recognition.jsonl
por O_APPEND, com lock, flush, fsync e close. Uma cauda incompleta fica
preservada no segmento antigo: leitores devem rejeitar/ignorar aquela linha
incompleta explicitamente. Não há reparo silencioso ou truncamento de registros.

A limpeza, sob o mesmo lock, examina apenas arquivos regulares com o padrão
exato dos segmentos do logger. Symlinks, outros JSONL, arquivos de modelos,
imagens, startup e métricas não são candidatos. Não há varredura recursiva.
O ativo nunca é excluído pela retenção. Primeiro são eliminados segmentos
mais antigos que o prazo, e os mais antigos restantes são removidos se
necessário para respeitar o teto (soma dos bytes do ativo e segmentos próprios).
Idade usa o mtime do último write; o conteúdo das linhas não é reprocessado.

Assim, 30 dias é uma janela máxima pretendida, não uma garantia mínima:
o teto pode encurtar o histórico. A política não afeta originais MinIO e
não cria lifecycle naquele bucket. Um lote pode exceder 10 MiB sem ser
dividido, desde que caiba no teto total. Acima do teto total, o lote é
rejeitado antes de alterar arquivos, com observability_write_failed.

Falhas de write/fsync/rotação não mudam a resposta da inferência. Um fsync
que falha pode deixar o lote já escrito; não há retry que duplique eventos.
Falha de limpeza emite apenas observability_retention_failed; o lote escrito
é mantido, mas o teto/prazo podem ficar temporariamente excedidos.

## Arquivos desta etapa

- Novo: `src/dice_viewer/recognition/log_storage.py`.
- Alterado: `src/dice_viewer/recognition/observability.py` (JsonlWriter delega
  append/rotação/expiração e lê limites do ambiente).
- Novo: `tests/test_log_storage.py`.
- Novo: este documento.

O compose existente já oferece volume persistente service-state em /state;
não foram criados serviços, volumes adicionais ou dependências externas.
A conferência continua usando o diretório Windows já montado no contêiner
dice-observability-stage4. O JSONL não foi copiado para entrega nem retirado
do gitignore, conforme a instrução mais recente de deixá-lo no local atual.
Os fontes, testes e relatórios anteriores continuam sem commit.

## Evidências

- Suíte completa Windows: 43 testes executados, 42 passaram e um foi pulado
  por ausência de privilégio para criar symlink. O aviso de inicialização
  asyncio do teste de métricas permanece informativo.
- Testes de armazenamento em diretórios temporários Linux também executados,
  incluindo criação real de symlink; nenhum caso pulado nesse ambiente.
- Cobertura de tamanho, virada UTC, reabertura, idade, teto, batch acima do
  orçamento, fsync/falha, cauda incompleta, arquivos alheios, symlinks,
  configuração por ambiente e fallback seguro.
- Revisão independente identificou overflow para retenção exagerada, capaz
  de impedir aplicação do teto. O caso falhou em teste antes da correção;
  agora valores acima de 36500 dias são inválidos.
- O JSONL real tinha 32848 bytes antes do reinício. Depois de reiniciar e
  enviar uma imagem, tinha 36013 bytes, com SHA-256 do prefixo original
  idêntico e todas as linhas JSON válidas. Chamadas posteriores acrescentam
  mais linhas; o tamanho atual não é fixo.
- A imagem real retornou 200/needs_review; a comparação com a API aprovada
  preservou contrato e resultados. Métricas continuaram parseáveis e com
  incremento unitário, sem contar scrapes.
- Nenhuma expiração/rotação foi forçada sobre a evidência real; cenários
  destrutivos foram exercitados somente em diretórios temporários.

## Validação manual — Windows PowerShell

```powershell
Set-Location C:\MLops-dice-viwer\dice-viwer
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
just health 24000
```

Esperado: 43 testes, OK (skipped=1 no Windows sem privilégio de symlink);
readiness HTTP 200. A proteção de symlink foi conferida em Linux.

Confira persistência/append sem apagar ou editar o arquivo real:

```powershell
$log = '.\runtime\observability-stage2\state\observability\recognition.jsonl'
$antes = [IO.File]::ReadAllBytes((Resolve-Path $log).Path)
just recognize "data/datasets/v1-20261117-7b23477513de/classifier/crops/e2/00305b126c3d649706836060.png" 24000 image/png
$depois = [IO.File]::ReadAllBytes((Resolve-Path $log).Path)
$depois.Length -gt $antes.Length
[Convert]::ToBase64String($depois[0..($antes.Length-1)]) -eq [Convert]::ToBase64String($antes)
Get-Content $log | Select-Object -Last 3 | ForEach-Object { $_ | ConvertFrom-Json } | Select-Object event,request_id
```

Esperado: inferência 200/needs_review, True para aumento e preservação do
prefixo, depois um request e dois die com mesmo ID. Isso pressupõe que o
arquivo não cruze tamanho/data durante o comando; se houver rotação, os
bytes antigos estarão no segmento fechado e devem ser conferidos ali.

Para inspecionar os arquivos persistidos:

```powershell
Get-ChildItem '.\runtime\observability-stage2\state\observability' -Filter '*.jsonl' | Select-Object Name,Length,LastWriteTimeUtc
```

Esperado: recognition.jsonl e eventuais segmentos fechados. Sem alcançar
10 MiB ou mudar o dia UTC, é normal existir apenas o ativo.

Esta demonstração cria um diretório novo e separado para conferir rotação
e teto com quatro lotes pequenos; não toca na evidência real:

```powershell
@'
import json, os, sys, uuid
from pathlib import Path
from datetime import datetime, timezone, timedelta
sys.path.insert(0, 'src')
from dice_viewer.recognition.log_storage import AppendOnlyLog, LogPolicy
root = Path('runtime') / ('retention-demo-' + uuid.uuid4().hex)
now = datetime.now(timezone.utc)
active = root / 'recognition.jsonl'
store = AppendOnlyLog(active, LogPolicy(max_bytes=12, max_total_bytes=24), clock=lambda: now)
for n in range(1, 5):
    store.append((json.dumps({'n': n}) + '\n').encode())
    os.utime(active, (now.timestamp(), now.timestamp()))
    now += timedelta(seconds=1)
print('Diretorio:', root)
for path in sorted(root.glob('*.jsonl')):
    print(path.name, path.read_text().strip())
print('Bytes gerenciados:', sum(p.stat().st_size for p in root.glob('*.jsonl')))
'@ | .\.venv\Scripts\python.exe -
```

Esperado: dois arquivos, contendo n=3 no segmento fechado e n=4 no ativo;
18 bytes ao todo, abaixo do teto 24. Os lotes n=1 e n=2 foram eliminados
pelo teto somente nesse diretório novo. O teste de expiração por idade usa
relógio controlado e não modifica o relógio do sistema.

```powershell
just restore
git diff --check
git diff --exit-code -- src/dice_viewer/recognition/contract.py src/dice_viewer/evaluation/cascade.py
git status --short
```

Esperado: release verificada; comandos de diff sem saída/código 0; arquivos
desta etapa e das etapas anteriores ainda não versionados/modificados.
Nenhum commit, push ou merge. A API temporária permanece disponível para
avaliação; para pará-la: docker stop dice-observability-stage4.

## Limitações e riscos

- Um processo/writer por caminho; lock é entre threads desse writer.
  Não há coordenação para múltiplos processos, proteção contra adulteração
  externa, backup, armazenamento remoto ou garantia de auditoria inviolável.
- fsync reduz perda de conteúdo, mas não garante durabilidade do rename e
  entradas do diretório em crash/pane; não há fsync de diretório nem transação
  entre arquivos. A última linha pode ficar parcial.
- fsync, scan e remoção são síncronos; filesystem lento pode atrasar próximas
  requisições. O teto usa bytes lógicos, não blocos/overhead do filesystem.
- Se o disco já estiver cheio, write/fsync pode falhar antes da limpeza.
  Pode ser necessário liberar espaço externamente. Não há recuperação
  automática, timer sem tráfego ou garantia de teto sob falhas de I/O.
- Exclusão por prazo/teto é definitiva. Antes da entrega ao professor,
  preparar uma captura versionável das evidências nas etapas finais, sem
  depender da janela operacional como arquivo permanente de entrega.
- Contêineres/volumes/diretórios devem ser preservados pelo operador;
  docker stop/restart preserva o bind mount atual, mas excluir o diretório
  no host ou remover volumes de produção exclui a persistência correspondente.

Etapa 5 encerrada para validação manual. Não iniciar a Etapa 6 sem aprovação
explícita do usuário.
