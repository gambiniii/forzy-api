# Handoff — agente, atribuição por componente e o que vem agora

Documento de passagem. Escrito para quem for continuar o trabalho, humano ou
assistente. Cobre o que mudou nos dois repositórios, o estado atual verificado, as
armadilhas que já custaram tempo, e o que fica pendente.

Leia junto com [`fase2-modelo-3d-e-atribuicao.md`](./fase2-modelo-3d-e-atribuicao.md),
que tem o detalhe técnico completo. Este aqui é o resumo operacional.

---

## 1. Estado atual, verificado

Bateria de 11 testes no agente, todos passando, nenhum fallback:

| cenário | tempo | tool usada |
|---|---|---|
| Componente do motor (tela principal) | 6,6 s | `get_motor_part_info` |
| Visão geral do sistema | 11,1 s | `get_system_overview` |
| O que o ML detecta agora | 8,6 s | `get_ml_atribuicao` |
| Métricas dos modelos | 7,1 s | `get_ml_metricas` |
| Relatório PDF / Excel / Word | ~6 s cada | `generate_report` |
| Componente e leituras (mini chat) | 5,4 / 6,3 s | `get_motor_part_info`, `get_db_leituras` |
| Recusa de pista interna vs externa | 4,3 s | nenhuma, responde de conhecimento |
| Recusa de rotação por minuto | 4,3 s | nenhuma |

Os três relatórios baixam com o content-type correto. Excel com 3 abas, Word com
15 parágrafos e 3 tabelas, PDF com seções e tabela de leituras.

Build do frontend: `tsc -b` e `npm run build` ambos com exit 0.

---

## 2. O que mudou nesta rodada

### O agente não era carregado no startup

O `main.py` subia o poller e o scheduler, mas não o agente. Ele era construído de
forma preguiçosa na **primeira mensagem** que o usuário enviasse depois de cada
reinício, e essa mensagem pagava a montagem inteira do vectorstore e do agente.
Qualquer falha nesse caminho chegava ao usuário como "erro de conexão", sem
nenhuma pista.

Agora o startup dispara o aquecimento em background. Medido: o agente fica pronto
sozinho em cerca de 6 s, sem ninguém chamar `/chat/warmup`. Roda em background
para não segurar o startup, e quem chamar o chat antes de terminar continua
funcionando pelo caminho preguiçoso de sempre.

### As mensagens de erro do chat mentiam

O assistente e o mini chat mostravam a mesma mensagem para qualquer falha. Isso
escondia a causa, e as três causas possíveis pedem ações diferentes:

| causa | mensagem agora |
|---|---|
| servidor fora do ar | "Não foi possível alcançar a API. Verifique se o servidor está no ar." |
| agente ainda carregando | "O agente não respondeu em 120s. Ele pode estar carregando os modelos, tente de novo em alguns segundos." |
| erro dentro do agente | a mensagem que a API devolveu, tal como veio |

`sendChatMessage` ganhou timeout explícito de 120 s via `AbortController`. Antes
não havia nenhum. O limite é generoso de propósito: o agente leva mais de 10 s
quando encadeia várias tools.

### Unidades erradas nos relatórios

Os três formatos rotulavam a coluna `rpm` como "RPM" e a `vibracao` como
"Vibração (mm/s)". É o contrário: **`rpm` guarda a velocidade de vibração em mm/s
e `vibracao` guarda a aceleração em g.** O PDF saía com "RPM 4.300" para uma
leitura de 4,3 mm/s.

Era o mesmo defeito já corrigido nas tools de banco, mas os relatórios não tinham
recebido a correção — e o relatório é justamente o artefato que sai da aplicação e
vai para outra pessoa ler.

O "RPM nominal" da tabela de **especificações** continua, renomeado para "Rotação
nominal (rpm)": ali é rotação de verdade, dado de placa, não leitura de sensor.

---

## 3. Armadilhas que já custaram tempo

Estas quatro merecem atenção porque são silenciosas: elas não dão erro, dão
resultado errado.

### `tsc --noEmit` é falso positivo neste projeto

O `tsconfig.json` é só um arquivo de referências. Verificado com
`npx tsc --noEmit --listFilesOnly`: **zero arquivos**. O comando compila nada e sai
com sucesso sempre. Eu confiei nele por boa parte de uma sessão e o build de
produção estava quebrado esse tempo todo.

**Use `npx tsc -b` ou `npm run build`.**

### As colunas de `leitura_sensor` têm nomes que enganam

```
leitura_sensor.rpm         -> VELOCIDADE de vibração em mm/s   (não é rotação)
leitura_sensor.vibracao    -> ACELERAÇÃO em g                  (não é velocidade)
leitura_sensor.temperatura -> °C da carcaça
```

Os três canais vêm do mesmo acelerômetro IO-Link. Os limites ISO 10816 de 2,8 e
4,5 mm/s valem para a **velocidade**, nunca para a aceleração em g. Esse mesmo
erro já apareceu em três lugares diferentes: nas tools de banco, no gerador de
relatórios e nos limites do componente 3 no banco.

E **não existe sensor de rotação** neste sistema. A rotação nominal de 3600 rpm
síncrona é dado de placa.

### Timestamps do banco têm fuso, os do CSV não

O banco devolve `timestamptz`. O CSV histórico não tem fuso.
`pd.to_datetime(...).astype("datetime64[us]")` levanta `TypeError` na entrada com
fuso, então testes sobre o CSV passam enquanto o endpoint quebra em produção.

Normalize com `pd.to_datetime(x, utc=True).dt.tz_localize(None)`.

### Adicionar coluna ao modelO ORM sem migration quebra o endpoint

O SQLAlchemy seleciona **todas** as colunas mapeadas em qualquer consulta. Se a
coluna não existir no banco, a query falha inteira. Um `getattr` defensivo no
código protege o acesso ao atributo, mas não a query.

Isso já aconteceu: acrescentei `acel_atencao` e `acel_critico` ao
`ComponenteLimite` e o `GET /componentes/{id}/limites` passou a devolver 500 até a
migration rodar. **Existem três colunas na mesma situação hoje** — ver seção 5.

---

## 4. O que existe agora e como usar

### Treinar os modelos

```bash
python -m ml_module.forzy.train        # cerca de 220 s
```

Salva em `ml_module/forzy/saved/`. O `detectores.joblib` (23 MB) está no
`.gitignore` de propósito: a inferência reconstrói o autoencoder do `state_dict`
em `modelos.json`, e a Mahalanobis com o Isolation Forest só servem para a
comparação da bancada, cujo resultado já está em `metricas.json`. Verificado:
renomear o joblib não afeta a inferência.

### Endpoints

```
GET  /diagnosticos/componente/{id}/atribuicao?janela_min=15
GET  /diagnosticos/modelos/metricas
```

Os dois são **consultivos**: não sobrescrevem `overall_status`, que continua
vindo do pipeline existente.

### Ver e corrigir os limites

```bash
python scripts/corrigir_limites.py --ver       # só mostra
python scripts/corrigir_limites.py --migrar    # só o schema
python scripts/corrigir_limites.py --aplicar   # schema + recalibra
```

O schema **já foi aplicado**. Os valores **não**.

### O agente

20 tools. As 5 novas desta fase:

| tool | para que serve |
|---|---|
| `get_motor_part_info(peca)` | ficha de uma peça: função, componentes internos, modos de falha |
| `diagnose_by_signals(vel, acel, temp)` | dado o padrão de sinais, componentes candidatos + causas WEG |
| `get_motor_baselines()` | referências medidas + limites normativos + fenômenos que parecem falha |
| `get_ml_atribuicao(componente_id)` | o que os modelos estão detectando agora |
| `get_ml_metricas()` | métricas da bancada de injeção |

---

## 5. O que fica pendente

### Bloqueante: recalibrar os limites

Os valores no banco deixam os dois motores permanentemente críticos, o que torna
o destaque no modelo 3D inútil — se tudo está sempre vermelho, o vermelho não
informa nada.

```
componente 2 (S1): vib_critico = 4,5 mm/s   mas a baseline SAUDÁVEL medida é 6,68
componente 3 (S2): vib_critico = 0,02       escala de ACELERAÇÃO contra VELOCIDADE
```

O componente 3 fica crítico **até parado**, porque a velocidade com o motor
desligado já é 0,049 mm/s, maior que 0,02. O componente 2 cruza o limite no
instante em que liga.

Rode `python scripts/corrigir_limites.py --aplicar`. Os valores propostos vêm da
baseline de cada motor mais 2 e mais 4 desvios, e estão declarados no cabeçalho do
script como **priors**, não como ajuste estatístico — com 19 minutos de regime e
zero falhas no histórico não existe como calibrar de outra forma.

### Três colunas no ORM sem migration

`confidence`, `threshold_status` e `threshold_message` entraram em
`src/models/diagnostico.py` mas só `breached_metrics` ganhou arquivo `.sql`. Hoje
funciona porque foram aplicadas à mão no banco, mas **um deploy do zero sobe a
tabela sem elas e todo `save_diagnostico` quebra**. É o mesmo tipo de problema
descrito na seção 3.

### Quatro tools do agente estão inertes

`get_sensor_status`, `get_ml_analysis`, `get_active_alerts` e
`get_maintenance_history` chamam a API interna e dependem de `AGENT_API_TOKEN`,
que **não existe em nenhum lugar do projeto** — não está no `.env`, no
`.env.example` nem no `docker-compose.yml`. Sempre devolvem 401.

Marquei as docstrings para o agente preferir as tools de banco, que cobrem a mesma
informação lendo o Postgres direto. Para ativar de verdade: definir
`AGENT_API_TOKEN` com um token de usuário admin, ou reescrever essas quatro para
ler o banco como as outras fazem.

### Levar o canal a-Peak até o banco

O fator de crest é o indicador **precoce** de rolamento e é o que dá a evidência
mais forte para acender o mancal. Ele existe nos bytes 8 e 9 do array PDI, mas o
poller consome `GET /get_s1`, que devolve só Velocidade, Aceleração e Temperatura,
e `leitura_sensor` não tem coluna de pico.

Enquanto isso não existir, a inferência **estima** o crest pela mediana saudável
medida (3,22) e declara essa limitação no campo `limitacoes` da resposta. Não é
invenção silenciosa, mas é evidência enfraquecida.

### Tabela de segmentos: unificar a fonte da verdade

O catálogo das 23 peças vive hoje em dois arquivos que precisam ficar em sincronia
à mão:

- `forzy-web/src/config/motorParts.ts`
- `forzy-api/rag_module/tools/motor_parts_tool.py`

Com a `migration 006_motor_segmento.sql` no lugar, a fonte da verdade passa a ser
o banco e os dois arquivos viram cache. Os campos que o catálogo tem e que a
tabela precisa acomodar: id do segmento no OBJ, nome oficial WEG, sistema
funcional, descrição, componentes internos que a peça representa, modos de falha
com assinatura e precocidade, e a marcação de se a peça pode acender.

### Bugs de produção documentados mas não corrigidos

Estes mudam comportamento e merecem verificação isolada, por isso não foram
tocados:

1. **O gate operacional lê a leitura mais antiga da janela.**
   `gate_velocity_snapshot` indexa `vel_series` sobre um `raw_df` ordenado
   decrescente, então decide "operando ou desligado" com um dado de até 83 minutos
   atrás. Afeta especificamente o `diagnostico_scheduler`, que é o que roda a cada
   5 minutos.
2. **Janelas móveis treinadas a 30 s / 2 min / 5 min, servidas a 8 / 33 / 83 min.**
   `ROLLING_WINDOWS` conta amostras, não tempo, e o poller mudou de intervalo.
   16,7 vezes mais longas em produção.
3. **`predict_single` usa uma lista de features diferente da do treino.**
   Coincidem em 43 por acidente. Adicionar qualquer feature a `build_features`
   derruba o scheduler com `ValueError` no scaler. Vale um teste de regressão
   travando os 43 nomes na ordem.
4. **RUL desligado desde a V2 das features**, e o R² de 0,9994 é vazamento de
   alvo: o label é fabricado do próprio IF+LSTM e reinjetado como feature. Não
   ressuscitar sem refazer os labels.
5. **`port2` é cópia bit a bit de `port1`** em produção, então 21 das 43 features
   são duplicatas e o posto efetivo é 22.

---

## 6. O que não dá para fazer, e por quê

Isso está aqui para ninguém prometer o que o sistema não entrega.

1. **Detectar BPFO, BPFI, BSF ou FTF.** É Nyquist, não falta de esforço. O sensor
   entrega um escalar agregado por leitura, não forma de onda. As frequências do
   rolamento 6206 a 3525 rpm ficam entre 23 e 319 Hz e exigiriam amostragem
   mínima de 638 Hz. A nossa é 1,67 Hz no arquivo histórico e 0,1 Hz em produção.
   Faltam 191 vezes.
2. **Distinguir pista interna de externa, esfera ou gaiola.** Mesma
   impossibilidade: essa distinção *é* a assinatura de frequência.
3. **Distinguir o rolamento do lado acionado do não acionado.** Um sensor por
   motor.
4. **Apontar uma peça entre as 23 com evidência.** O teto honesto são grupos
   funcionais, e é assim que a interface apresenta.
5. **Qualquer classificador supervisionado de falha.** Zero rótulos, histórico
   100% saudável.
6. **Threshold calibrado estatisticamente.** Todo limiar é prior declarado.

E a ressalva que precisa acompanhar qualquer número de desempenho: **as falhas da
bancada são injetadas, não observadas.** Isso prova que o detector reconhece
desvios com a assinatura física esperada. Não prova que detectaria uma falha real,
que tem componentes não modelados. As métricas são um limite superior otimista.
