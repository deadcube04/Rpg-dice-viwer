---
name: review-dice-images
description: Use quando um agente ou uma automação receber a tarefa de revisar imagens e anotações na UI local RPG Dice em 127.0.0.1:8765. Exclusiva para revisão visual desse dataset; não se aplica a revisão de código, treinamento ou aprovação de modelos.
---

# Revisar imagens na UI RPG Dice

Revise visualmente uma imagem de cada vez e grave somente o que a imagem sustenta. O usuário autoriza revisão por agente/automação; uma anotação herdada não é evidência de que você conferiu a imagem. Uma revisão concluída também não aprova o dataset para treinamento.

## Preparar a execução

1. Leia o escopo recebido: filtros/fonte, limite de imagens ou tempo, e se deve continuar de uma execução anterior. Use os limites que o usuário/automação forneceu. Sem limite, trabalhe na fila pendente enquanto houver tempo disponível; pare em um ponto salvo e registre a retomada.
2. Abra **http://127.0.0.1:8765** com as ferramentas de navegador/computer-use disponíveis e leia a documentação dessas ferramentas. Se não for possível ver a imagem ou controlar a UI, relate o bloqueio; não substitua a revisão visual por cópia de labels, palpites ou gravações em lote.
3. Se o serviço estiver parado e a tarefa autorizar operar o projeto local, execute `C:/MLops-dice-viwer/dice-viwer/audit/start-audit.cmd`. Não reimporte o ZIP, não restaure backups nem altere o banco para iniciar uma revisão.
4. Na fila, use **Estado → Pendente**. Ajuste apenas os demais filtros pedidos. Para digitar em um select: abra o campo pelo nome, digite a opção, selecione o resultado exato e confira o valor selecionado. Busca não confirma a escolha. O select aceita as opções do vocabulário, sem criar classes novas.
5. Se a última imagem retomada estiver fora do filtro ou já estiver concluída, abra explicitamente a primeira pendente exibida na fila. Registre **Imagem** (ID), caminho e **Revisão** no cabeçalho. Use identidade, não posição, para acompanhar o trabalho.

## Ciclo repetível de uma imagem

### 1. Conferir a cena inteira

- Use **Ajustar** para começar com a cena inteira. Confira todos os dados visíveis, inclusive sem face legível, parcialmente ocultos ou cortados. O número de caixas existentes não prova a quantidade real de dados.
- Use **Zoom +**, **Zoom −**, **Deslocar**, **Tela cheia** e **Mostrar/Ocultar caixas** quando necessário. Não existe painel de recorte ampliado; o zoom é na imagem principal.
- Compare caixas com corpos físicos. Uma caixa marcada **Somente numeral** não delimita automaticamente o corpo do dado. Se houver dado visível nessa fonte, anote seu corpo a partir da imagem; não apenas troque a categoria mantendo a caixa pequena do numeral.
- Em **Selecionar / mover**, arraste uma caixa para movê-la e o canto inferior direito para redimensioná-la. Em **Criar caixa**, arraste sobre um dado ausente. **Remover instância** remove somente a anotação selecionada. As coordenadas x1/y1/x2/y2 também são editáveis em pixels após EXIF; não use coordenadas da captura de tela como pixels originais.
- Caixas devem cobrir a região visível do corpo, sem sombra, estar dentro da imagem e ter área positiva. Preserve dados acima de oito para revisão explícita; não remova objetos só para caber no limite futuro de inferência.

### 2. Conferir cada instância

Selecione **Selecionar instância 1**, depois 2, 3 etc. Confira **Instância selecionada: N de M** antes de editar. Adições/remoções podem renumerar os botões: releia a lista após mudar a quantidade.

| Controle | Ação de revisão |
|---|---|
| Tipo do dado | Escolha d4/d6/d8/d10/d10_tens/d12/d20 conforme evidência. Use Desconhecido quando não for determinável. d10_tens é o dado de dezenas, não um dado de cem faces. |
| Token textual da face | Leia a face de resultado, não um numeral lateral. Selecione a opção exata, preservando 0, 00 e 10. O campo Ilegível / desconhecido deixa o token ausente. |
| Valor normalizado | Confira a exibição: 0, 00 e 10 produzem 10; demais tokens mantêm o inteiro. Não calcule soma da cena ou composição percentual. |
| Valor confirmado nesta instância | Marque somente se a face e a convenção tornam o resultado inequívoco. Uma proposta antiga não confirmada exige sua nova leitura visual. |
| Resultado ambíguo | Marque quando não houver resultado inequívoco: por exemplo, dado apoiado/inclinado ou 6/9 sem orientação resolvível. Isso desabilita a confirmação do valor. |
| Oclusão / Corte na borda | Marque cada condição observada, sem inventar geometria da parte invisível. |
| Região anotada | Corpo do dado para uma caixa de corpo; Somente numeral apenas para região numérica. Numeral não supervisiona detecção do corpo. |
| Modalidade da face | Pontos (d6), Numerais ou Não identificada conforme a imagem. Contagem dos pontos é o token do d6. |
| Convenção d4/d10 ou coleção | Registre a convenção quando identificável. d4 pode ser lido na ponta ou na base; não escolha um numeral arbitrariamente. |

Se algo não puder ser determinado, mantenha a instância com campos desconhecidos/máscaras apropriadas e explique a limitação em **Observações**. Uma revisão pode estar completa mesmo contendo valores desconhecidos; isso é diferente de deixar trabalho incompleto.

### 3. Escolher um dos quatro radios

| Radio em Decisão da imagem | Quando usar |
|---|---|
| **Pendente** | A conferência ainda está incompleta, há impedimento de execução ou você vai interromper antes de terminar. Guarda o rascunho e continua na fila pendente. |
| **Revisada** | Você conferiu toda a cena, geometria e todas as instâncias; limitações de tipo/valor foram explicitamente marcadas. Exige ao menos uma instância. |
| **Sem dados** | Você inspecionou a cena e confirmou ausência de dados. Remova todas as instâncias antes de salvar. TXT vazio ou ausência de labels não prova uma negativa. |
| **Excluída** | A imagem é inadequada para uso, por exemplo arquivo ilegível ou conteúdo alheio ao escopo. Preencha **Motivo da exclusão / decisão**. Não exclua apenas por dificuldade, duplicação ou valor ambíguo. |

Não há campo de responsável. Não confunda este grupo de radios com o filtro **Estado** da fila. Registre motivos concretos; não declare confiança, calibração ou aprovação humana que não ocorreu.

### 4. Salvar e verificar antes de repetir

1. Anote o ID e a revisão atuais, a decisão escolhida e quantas instâncias foram conferidas.
2. Prefira **Salvar e próxima** (Ctrl/Cmd+Shift+S). Aguarde o botão sair de carregamento e a interface atualizar. Não edite durante gravação.
3. Se o ID mudou e não há erro, a gravação terminou antes do avanço. Se permaneceu o mesmo, confirme que **Revisão** aumentou ou que a UI informou o fim da fila. **Salvar** (Ctrl/Cmd+S) também aumenta a revisão, mantendo a imagem aberta. Confirme o resultado visível; clicar não é conclusão.
4. Se precisar esclarecer uma gravação após desconexão, pode consultar, somente para leitura, `GET /api/frames/{image_id}` e `/history`. Compare decisão, revisão e anotações com o que acabou de conferir. Não use PUT/API/SQL para confirmar imagens em lote ou contornar a UI.
5. Repita na imagem seguinte. Ao salvar Pendente, registre seu ID como não concluído para evitar repetir indefinidamente a mesma imagem na mesma execução. Cada duplicata continua exigindo revisão própria; a comparação por hash não aprova a anotação.

## Falhas e retomada

- **400 ou validação:** leia a mensagem e corrija a causa visível (caixa inválida, token incompatível, exclusão sem motivo, negativa com instâncias). Não altere um rótulo verdadeiro apenas para passar na validação; marque desconhecido/ambiguidade quando essa for a evidência.
- **409 / revisão desatualizada:** pare de salvar essa imagem. Recarregue; o rascunho permanece acessível. Compare com **Histórico de revisões**. Use **Conciliar sobre revisão atual** somente se puder resolver cada diferença; caso contrário, preserve o rascunho e reporte o ID/conflito. Não sobrescreva outra revisão às cegas.
- **Falha de rede ou gravação sem resposta:** confira por leitura se a revisão já foi persistida antes de repetir o clique. Se não puder confirmar, registre resultado incerto e interrompa; não conte como concluída.
- **Aviso de rascunho local:** confirme o salvamento no servidor antes de sair. Não dependa somente de localStorage para recuperar trabalho.
- **Fim do tempo/limite:** finalize a imagem atual se houver condições; caso contrário salve Pendente com observação ou preserve o rascunho e registre o ID. Não confirme automaticamente o restante da fila.
- **Fila vazia:** confira filtros e progresso; “nenhuma imagem neste filtro” não significa que as 6.915 entradas estão concluídas.

## Encerrar e reportar

Informe: quantidade de imagens efetivamente conferidas/salvas nesta execução, decisões aplicadas, IDs pendentes/conflitantes/incertos e próximo ponto de retomada. Diferencie seus números do progresso global mostrado pela UI; outras execuções podem ter trabalhado no mesmo banco.

Se solicitado, crie backup/snapshot pelos botões da UI. Exportar não aprova dados. Esta skill termina na revisão e registro: não prepara splits, não treina, não promove modelo e não inicia etapas posteriores.

Referência de convenções: `C:/MLops-dice-viwer/ARQUITETURA_V1_RECONHECIMENTO_DADOS_RPG.md`, seções 4, 11 e 12, considerando a autorização posterior de revisão por agente. Operação da ferramenta: `C:/MLops-dice-viwer/dice-viwer/docs/audit/README.md`.
