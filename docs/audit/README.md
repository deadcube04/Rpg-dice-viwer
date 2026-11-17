# Auditoria de imagens RPG Dice

Subprojeto independente em `audit/`, separado do código e do ambiente de reconhecimento. Arquitetura de referência: [ARQUITETURA_V1_RECONHECIMENTO_DADOS_RPG.md](../plans/ARQUITETURA_V1_RECONHECIMENTO_DADOS_RPG.md). Plano: [PLAN.md](../plans/PLAN.md).

## Estrutura

```text
audit/
  audit-ui/                 React + Vite + TypeScript, package.json e bun.lock
  audit-back/
    src/dice_audit/          API, SQLite e importadores; não importa dice_viewer
    scripts/                Migração de caminhos do cache
    pyproject.toml          Pacote rpg-dice-audit e comando dice-audit
    uv.lock                 Dependências Python fixadas
    requirements.txt        Alternativa de instalação mínima com pip
    run.py                  Launcher local sem depender de instalação do pacote
    .venv/                  Ambiente Python próprio (ignorado)
    runtime/                SQLite, cache, inventário, backups e exports (ignorados)
  skills/review-dice-images/SKILL.md
  scripts/reorganize-legacy.ps1
  start-audit.cmd
  README.md
```

A auditoria usa apenas a biblioteca padrão Python e Pillow 12.3.0. Não depende de YOLO, Torch, BentoML, MinIO ou do pacote Python do reconhecimento. O ZIP original permanece externo e imutável em C:/diecamera-frames/data.zip.

## Executar neste computador

Abra `audit/start-audit.cmd` ou, a partir da pasta audit/audit-back:

```bat
.venv\Scripts\python.exe run.py serve
```

A UI compilada e a API ficam em **http://127.0.0.1:8765**. O banco é audit-back/runtime/audit-v1.sqlite3. A porta 8765 e o contrato /api permanecem iguais após a extração.

## Preparar um ambiente novo

Python 3.11, uv e Bun:

```bat
cd audit-back
uv sync --locked --python 3.11
cd ../audit-ui
bun install --frozen-lockfile
bun run build
cd ../audit-back
uv run --locked dice-audit import
uv run --locked dice-audit serve
```

Importar é necessário somente em runtime novo; não reimporte para iniciar uma sessão de revisão existente. O ZIP esperado tem SHA-256 eeeb6c2873fc513bba6211a7df6d17ed33939cc0030b28a0da54d483f2a91562. A importação verifica esse hash e conserva todas as 6.915 entradas de imagem, inclusive duplicatas.

Alternativa sem uv, em audit-back: crie .venv com `py -3.11 -m venv .venv`, instale `-r requirements.txt` e execute `.venv\Scripts\python.exe run.py serve`. Esse launcher usa o src local, sem importar o projeto principal.

## Desenvolvimento separado

Backend: execute `uv run --locked dice-audit serve` em audit-back.
Frontend: execute `bun run dev` em audit-ui. Vite usa 127.0.0.1:5173 com proxy /api para 127.0.0.1:8765. Para servir outro build, o backend aceita `--ui-dist <diretório>`; também aceita `--runtime`, `--archive` e `--port`. Comandos: import, serve, status, backup e export.

O backend inicia e atende /api independentemente do frontend. Se o build não estiver disponível, a página raiz responde 503 com a instrução para compilá-lo. Todos os caminhos padrão são relativos à instalação de código do subprojeto, sem depender do cwd do reconhecimento.

## Revisão e automação

A [skill review-dice-images](../../audit/skills/review-dice-images/SKILL.md) descreve o ciclo por imagem. Também está instalada em C:/Users/PC/.codex/skills/review-dice-images; use `$review-dice-images` no prompt do agente, informando filtros e limites da execução.

A UI mostra ID e revisão no cabeçalho, permite digitar para buscar opções existentes nos selects e selecionar instâncias pelo nome “Selecionar instância N”. Não pede responsável e usa quatro radios: Pendente, Revisada, Sem dados e Excluída. Zoom, deslocamento e tela cheia operam na imagem principal, sem recorte ampliado.

Confira toda a cena e cada caixa de corpo; numeral não equivale a corpo. Preserve tokens 0/00/10 e máscaras por instância. Use desconhecido/ilegível/ambiguidade quando a imagem não sustentar o rótulo. Registre modalidade de d6 e convenções d4/d10. Sem dados exige zero instâncias; Excluída exige motivo. Salvar Pendente apenas guarda o trabalho incompleto.

Salvar e próxima só avança após gravação. Ctrl/Cmd+S salva; Ctrl/Cmd+Shift+S salva e avança. Rascunhos locais e proteção ao sair ajudam a retomada; confirmar salvamento no servidor continua necessário. Em 409, compare com o histórico antes de conciliar. Uma duplicata nunca confirma outra imagem automaticamente.

Revisões podem ser realizadas por agente/automação, conforme autorização do usuário. Revisar/exportar não aprova o dataset nem inicia treinamento. Etapas posteriores continuam exigindo sinal explícito.

## Persistência e recuperação

SQLite usa WAL, transações, controle de revisão e histórico. Os botões Criar backup e Exportar revisões trabalham somente com o runtime deste subprojeto. Snapshots incluem hash e approved=false e não contêm splits definitivos.

Para restaurar, pare a API, preserve o runtime atual e restaure um backup como audit-v1.sqlite3 em um runtime separado. Informe --runtime ao reiniciar. O cache gerado deve acompanhar esse runtime; o ZIP original também permanece necessário para consultar originais. Quando um backup antigo contém cache_path absoluto anterior à mudança, a API tenta o cache local runtime/images/{image_id}.png, sem alterar anotações ou snapshots históricos.

## Migração já realizada

A extração moveu a UI, backend/importadores, skill, launcher e runtime da raiz para audit. Criou ambiente novo em audit-back/.venv; o ambiente antigo foi guardado em runtime/legacy-env para recuperação, sem ser usado na execução atual. O banco legado audit.sqlite3, backups, exports e interface antiga também foram preservados dentro do runtime.

Antes da movimentação foi criado backup. A migração operacional `scripts/relocate_cache_paths.py` atualizou cache_path em frames/revisions e verificou um SHA-256 do conteúdo excluindo apenas esse campo: as anotações, IDs, decisões, revisões e histórico permaneceram iguais. Relatórios ficam em runtime/migrations. Snapshots históricos não foram reescritos, para preservar seus hashes.

O script audit/scripts/reorganize-legacy.ps1 é somente para migrar a estrutura antiga, não para iniciar a aplicação ou repetir a revisão. A organização atual já está migrada.

## Verificação

Build/typecheck do frontend, compilação Python e operação da API devem ser conferidos dentro destes dois projetos. Não criar ou executar testes automatizados. O backend possui somente Pillow e rpg-dice-audit em seu ambiente, isolado do ambiente de modelos. Bun audit anterior retornou {} para as versões atuais; nenhuma dependência frontend foi alterada na reorganização.

Validação da extração realizada: uv run --locked dice-audit status apresentou 6.915 imagens e a revisão existente; build/typecheck Vite e compileall Python passaram. A API e a imagem revisada foram carregadas no navegador pelo novo serviço. Digests da migração confirmaram preservação de frames e histórico, exceto cache_path. Nenhum teste automatizado foi criado ou executado.

## Escopo ativo da revisão

Por orientação do usuário, `roboflow/d4-turbo-rad-v4` fica fora da fila, dos filtros de origem, das sugestões de duplicatas e do progresso por enquanto. Suas 1.219 imagens e revisões permanecem no inventário, nos backups e nos snapshots completos. O escopo ativo contém 5.696 imagens; isso não remove o tipo d4 das demais fontes.

A barra divide decisões concluídas pelo total ativo, independentemente dos filtros. Exibe duas casas decimais e atualiza a cada cinco segundos enquanto a página está visível, além da atualização imediata após salvar. `inventory_total` e `ignored_sources` na API distinguem inventário completo e escopo ativo.
