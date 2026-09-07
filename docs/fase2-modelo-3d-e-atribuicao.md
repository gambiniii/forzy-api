# Fase 2 — Modelo 3D segmentado, modelos de ML e atribuição por componente

Documentação do que foi implementado. Cobre os dois repositórios, `forzy-api` e
`forzy-web`.

O objetivo da fase era fechar o circuito **dado → ML → frontend → peça vermelha no
modelo 3D → explicação do agente**. Está fechado e verificado ponta a ponta.

---

## 1. As 23 peças do modelo 3D, identificadas e nomeadas

O arquivo `forzy-web/public/3d/Engine1.obj` tem 23 objetos, todos chamados
`empty_2` até `empty_24`. Agora cada um tem nome oficial WEG, função, sistema
funcional, modos de falha e a lista de componentes internos que representa.

### Como foram identificados

Por **geometria**, não por inspeção visual. O modelo está em milímetros e em
escala 1:1, e a prova é direta: os pés ficam em Y=−100 e a linha de centro do eixo
em Y=0, logo a altura de eixo é 100 mm, que é exatamente a carcaça 100L do
datasheet do produto 13887610. O eixo do motor corre em X com a ponta em +X, e a
caixa de ligação fica em +Z, confirmando a "posição esquerda" do datasheet.

### Duas correções da Fase 1

| Segmento | Rótulo anterior | O que é de fato | Evidência |
|---|---|---|---|
| `empty_6` | carcaça/estator | **Tampa da caixa de ligação** | Chapa de 29,8 mm de espessura no ponto mais externo em Z, empilhada sobre `empty_5` (junta) e `empty_4` (caixa oca). O logo WEG que causou a confusão fica nela. |
| `empty_24` | ponta do eixo | **Chaveta** | 45×8×7 mm com apenas 12 faces, assentada no rasgo do eixo. 8×7 mm é a chaveta DIN 6885 normalizada para eixo Ø28–30. |

A carcaça é `empty_2`, o maior volume do modelo, com 18010 faces e casca oca de
raio interno 84,5. O eixo é `empty_23`, cilindro escalonado Ø30→Ø28 com 90 mm
livres.

### As 6 peças que o diagnóstico pode destacar

| Segmento | Peça | Sinais | Componentes internos que representa |
|---|---|---|---|
| `empty_2` | Carcaça | temperatura forte | Estator bobinado, núcleo, isolamento classe F, pés, placa de identificação |
| `empty_7` | Tampa dianteira | velocidade e aceleração fortes | **Rolamento 6206** do lado acionado, V-Ring, arruela ondulada |
| `empty_13` | Tampa traseira | aceleração forte | **Rolamento 6206** do lado não acionado, V-Ring, centrífugo, platinado |
| `empty_19` | Tampa defletora | temperatura forte | Ventilador, grelha, fluxo de ar IC411 |
| `empty_23` | Eixo | velocidade forte | Rotor, núcleo do rotor, assento do rolamento |
| `empty_4` | Caixa de ligação | temperatura média | Capacitor de partida e permanente, placa de bornes, aterramento |

Os outros 17 são detalhe visual e **nunca acendem sozinhos**: tampa da caixa de
ligação, junta, olhal de içamento, chaveta, dois drenos e onze parafusos. Não há
sinal que isole um parafuso específico.

### O interior do modelo é vazio

Todos os vértices no vão X entre −89 e +89 têm raio maior que 84,6. Não existe
malha para rotor, estator, rolamentos, ventilador, capacitor, centrífugo,
platinado, bornes, V-rings nem aterramento. Cada peça interna é representada pelo
proxy externo que fisicamente a contém — o rolamento do lado acionado está
literalmente dentro da tampa dianteira, o ventilador dentro da defletora. É isso
que o campo `componentesInternos` expressa.

### Onde isso vive

- `forzy-web/src/config/motorParts.ts` — catálogo das 23 peças
- `forzy-web/src/config/motorSegmentMap.ts` — mapa métrica → segmentos
- `forzy-api/rag_module/tools/motor_parts_tool.py` — mesma base para o agente

O rolamento **6206** está confirmado na Tabela 8.1 do manual WEG: carcaça 100,
2 polos, 5 g de graxa, relubrificação a cada 25.000 horas em 60 Hz.

---

## 2. Os modelos de ML

Pacote novo `forzy-api/ml_module/forzy/`, independente do pipeline antigo.

```bash
python -m ml_module.forzy.train     # cerca de 220 s
```

### Arquitetura: 6 modelos, não um global

São **2 motores × 3 regimes** (parado, transiente, operação).

**Por regime**, porque as distribuições não se sobrepõem. Parado e operação estão
separados por duas ordens de grandeza em velocidade, e um modelo único aceitaria
tudo que existe entre eles — que é justamente onde vive a perda de acionamento.

**Por ativo**, porque as assinaturas diferem. Medido: o S2 opera 4,39 °C mais
quente e vibra 0,182 mm/s a mais que o S1 em regime, com p menor que 1e-10.

O **transiente é pontuado mas não alarma**, por projeto: em partida e parada por
inércia a vibração varia legitimamente em uma ordem de grandeza.

### O quarto canal escondido

O CSV traz um array de 16 bytes de Process Data In por porta, além das três
colunas já convertidas. Três valores para dezesseis bytes significa que sobra
informação. A engenharia reversa, validada contra as próprias colunas
convertidas, encontrou:

| bytes | grandeza | confere com | acerto medido |
|---|---|---|---|
| 0-1 big-endian ÷ 100 | v-RMS [mm/s] | Velocidade | 77,5% |
| 4-5 big-endian ÷ 100 | a-RMS [g] | Aceleração | 98,3% |
| 8-9 big-endian ÷ 100 | **a-Peak [g]** | **nenhuma** | não exportado |
| 13 | Temperatura [°C] | Temperatura | 99,2% |

O a-Peak permite o **fator de crest**, que sobe antes do RMS quando um rolamento
começa a deteriorar. Mediana medida em regime saudável: 3,39 no S1 e 3,19 no S2,
dentro da faixa de 3 a 4 esperada de máquina em bom estado. Varredura dos 16
bytes confirmou que não existe um quinto canal.

### Os três detectores

| detector | PR-AUC | ROC-AUC | Recall@FP1% | FP no controle |
|---|---|---|---|---|
| Mahalanobis robusta | 0,7889 | 0,7806 | 0,5375 | 0,00% |
| Isolation Forest | 0,7660 | 0,7518 | 0,4445 | 0,00% |
| **Autoencoder 16→8→4→8→16** | **0,8841** | **0,8646** | **0,7035** | **0,46%** |

O autoencoder é o escolhido. A rede é deliberadamente pequena: com 1.100 a 2.100
amostras por ativo e regime, uma rede maior decoraria o conjunto e reconstruiria
bem até as anomalias.

O erro é agregado por **média mais máximo**, não EQM puro. O EQM dilui desvio
concentrado: um sensor travado desloca uma feature das 16 e o erro dela chega a
132 vezes o normal, mas diluído em 16 termos mal move a média.

### Bancada de injeção de falhas

O histórico real é 100% saudável, sem um único rótulo. A validação é perturbando o
dado de teste com degradações fisicamente calibradas: 6 modos × 3 severidades.

Quatro cuidados obrigatórios: série contínua e não janelas soltas; perturbar os
canais brutos e recalcular as features do zero; **requantizar** depois de injetar,
senão o detector acusa a falha pela perda de quantização em vez da física; e
descartar a zona cinza de 60 s após cada episódio.

Os episódios são ancorados **dentro da partição de teste do regime de operação**.
Sem isso caem no trecho de treino e a bancada sai vazia — foi o que aconteceu nas
duas primeiras execuções.

### Atribuição: casamento de padrão com sinal

```
pontuação = Σ(z[f] · peso[f]) / Σ|peso[f]|  −  média(|z[f]|) das estáveis
```

Ranquear por magnitude bruta não funciona, e isso foi medido. Os vetores de
z-score dos modos injetados no regime de operação:

| modo | v_rms | v_roll_max | a_rms | crest | razao_av |
|---|---|---|---|---|---|
| Desbalanceamento | +22,2 | +23,4 | +2,5 | −0,5 | **−15,0** |
| Rolamento | +2,2 | +8,4 | +3,2 | +4,4 | **+3,6** |
| Deriva de ganho | +6,1 | +8,0 | +4,2 | +0,3 | **+0,4** |
| Perda de acionamento | −18,3 | −19,3 | −11,9 | +0,2 | −0,3 |

`razao_av`, a razão entre aceleração e velocidade, é o discriminador. Despenca no
desbalanceamento, porque a energia vai para 1× a rotação e a velocidade sobe muito
mais que a aceleração. Sobe no rolamento, porque impacto de alta frequência é
aceleração. E fica em zero na deriva de ganho, porque todos os canais escalam
juntos e a razão se preserva.

O termo das **estáveis** penaliza movimento onde a falha deveria deixar quieto. É
o que separa sensor travado, em que o nível fica no valor nominal, de perda de
acionamento, em que o nível desaba.

### Resultado verificado

| falha injetada | atribuição | peças que acendem |
|---|---|---|
| Desbalanceamento | Desbalanceamento do rotor | eixo, tampa dianteira, carcaça |
| Sobreaquecimento | Problema térmico ou elétrico | carcaça, defletora, caixa de ligação |
| Rolamento | Degradação de rolamento | as duas tampas, onde ficam os mancais |
| Sensor travado | Falha de instrumentação | **nenhuma** |
| Deriva de calibração | Deriva de calibração do sensor | **nenhuma** |
| Perda de acionamento | Perda de acionamento | caixa de ligação, tampa traseira |
| Controle saudável | sem atribuição | nenhuma |

Os dois modos de instrumentação não acendem peça de propósito: não são falha do
motor, e trocar rolamento por causa de um sensor descalibrado seria o pior erro
possível.

Existe um **estado explícito de "sem atribuição"** quando nenhum eixo domina. Num
histórico 100% saudável, forçar uma atribuição sempre elegeria alguma peça, e ela
estaria errada.

### Duas guardas físicas obrigatórias

**Heat soak.** Depois de uma corrida longa a temperatura da carcaça continua
subindo com o motor já parado: medido 38 °C no fim da corrida chegando a um pico
de 46 °C cerca de 7,9 minutos depois de desligar. Existe uma região perfeitamente
saudável do espaço de operação com temperatura de 46 a 47 °C e vibração de
0,04 mm/s.

**Deriva térmica do platô.** Durante a corrida a vibração cai de 6,95 para
6,42 mm/s enquanto o motor aquece, a −0,038 mm/s por minuto. Queda de vibração
nunca indica degradação.

---

## 3. Backend

### Endpoints novos

```
GET /diagnosticos/componente/{id}/atribuicao?janela_min=15
GET /diagnosticos/modelos/metricas
```

O primeiro roda o detector na janela atual e devolve regime, score normalizado,
severidade, falha provável, componentes candidatos, segmentos 3D a destacar e as
features com maior desvio. O segundo devolve as métricas da bancada.

Ambos são **consultivos**: não sobrescrevem `overall_status`, que continua vindo
do pipeline existente.

### Limites de aceleração

`componente_limite` ganhou `acel_atencao` e `acel_critico` (migration 005). Sem
eles a métrica `aceleracao` nunca entrava em `breached_metrics` e os rolamentos
jamais podiam acender — justamente o componente que a aceleração melhor denuncia.

Defaults 0,63 e 0,78 g, que é a baseline medida em operação (0,498 g) mais 2 e
mais 4 desvios. São **priors declarados**, não ajuste estatístico.

Não use os 2 g e 4 g do documento de governança: aqueles valores se referem à
banda de alta frequência da cláusula prospectiva, não à aceleração agregada que o
sensor entrega hoje. Com eles o alarme nunca dispararia.

---

## 4. Frontend

### Tooltip rico com fixação por clique

`forzy-web/src/pages/machine-detail/SegmentTooltip.tsx`

Card HTML **fora do Canvas**, de propósito. O `<Html>` do drei escalaria o texto
com o zoom da câmera, seria recortado quando a peça está na borda, e usa
`pointerEvents: none`, o que impediria clicar no botão.

Dois modos de conteúdo, para não poluir:
- peça sem anomalia: nome, sistema funcional e função
- peça com anomalia: acrescenta o problema, as causas prováveis filtradas pelas
  métricas que de fato estouraram, os componentes envolvidos e o botão do agente

Passar o mouse mostra o card. **Clicar fixa**, e a partir daí o hover não troca
nem fecha — é o que permite levar o mouse até o botão. Sai clicando em outra peça,
clicando no vazio do visualizador, clicando fora dele, apertando Escape ou no X do
card. O clique no vazio usa o `onPointerMissed` do react-three-fiber.

### Duas fontes de destaque, combinadas

`combinarHighlight()` em `diagnosticoUtils.ts` funde:

1. `breached_metrics` — regra determinística do Metric Contract, auditável
2. Atribuição do detector de novidade — z-score por feature contra o baseline

**Regra de prioridade:** quando as duas apontam o mesmo segmento, a severidade
mais alta vence e as duas mensagens aparecem. O limite determinístico vem primeiro
no texto porque é o que o operador consegue auditar; o ML entra como evidência
adicional, nunca sozinho substituindo a regra.

### Handoff para o agente

O botão monta a pergunta com o contexto da peça, da métrica estourada e da
atribuição do modelo, e navega para o assistente via `location.state`. O
assistente lê o prefill, dispara a mensagem na montagem e limpa o state da rota
para não reenviar ao voltar. O `machine_id`, que era fixo em `"1"`, passou a vir do
componente real.

---

## 5. O agente

Ganhou 5 tools novas, indo de 15 para 20:

| tool | o que faz |
|---|---|
| `get_motor_part_info(peca)` | ficha de uma peça: função, componentes internos, modos de falha |
| `diagnose_by_signals(vel, acel, temp)` | dado o padrão de sinais, componentes candidatos + causas oficiais WEG |
| `get_motor_baselines()` | valores de referência medidos + limites normativos + fenômenos que parecem falha |
| `get_ml_atribuicao(componente_id)` | o que os modelos estão detectando agora |
| `get_ml_metricas()` | métricas da bancada de injeção |

A base de conhecimento inclui a tabela oficial **PROBLEMAS X SOLUÇÕES** do manual
WEG 50033244, capítulo 10, transcrita.

### Limites do que o agente pode afirmar

O prompt agora proíbe explicitamente:

- afirmar detecção de BPFO, BPFI, BSF ou FTF
- distinguir pista interna de externa, esfera ou gaiola
- distinguir o rolamento do lado acionado do não acionado
- apontar uma peça isolada como culpada
- tratar temperatura alta como falha quando o motor foi desligado há menos de
  15 minutos (heat soak)
- tratar queda de vibração como degradação

E deixa claro que **não existe sensor de rotação**: a rotação nominal de 3600 rpm
síncrona é dado de placa, não leitura.

---

## 6. O que é impossível com estes dados

Isso está documentado porque foi pedido explicitamente e porque é o que impede o
sistema de mentir.

1. **Detectar BPFO, BPFI, BSF ou FTF.** É Nyquist, não falta de esforço. O sensor
   entrega um escalar agregado por leitura, não forma de onda. As frequências do
   rolamento 6206 a 3525 rpm ficam entre 23 e 319 Hz e exigiriam amostragem
   mínima de 638 Hz. A nossa é 1,67 Hz no arquivo histórico e 0,1 Hz em produção.
   Faltam 191 vezes.
2. **Distinguir pista interna de externa, esfera ou gaiola.** É a mesma
   impossibilidade: essa distinção *é* a assinatura de frequência.
3. **Distinguir o rolamento do lado acionado do não acionado.** Um sensor por
   motor, e `port2` é espelho de software de `port1` em produção.
4. **Apontar uma peça entre as 23 com evidência.** O teto honesto são grupos
   funcionais.
5. **Qualquer classificador supervisionado de falha.** Zero rótulos.
6. **Fator de crest em produção.** O a-Peak só existe no arquivo histórico; o
   poller recebe apenas Velocidade, Aceleração e Temperatura, e `leitura_sensor`
   não tem coluna de pico.
7. **Threshold calibrado estatisticamente.** Cerca de 19 minutos de regime
   estacionário por motor, uma condição de carga, zero falhas. Todo limiar é
   prior declarado.

### Ressalva sobre as métricas

As falhas são **injetadas, não observadas**. Isso prova que o detector reconhece
desvios com a assinatura física esperada e quantifica a margem entre normal e
anômalo. **Não prova** que detectaria uma falha real, que tem componentes não
modelados: espectro, modulação, interação com a carga. As métricas são um **limite
superior otimista**.

---

## 7. Correções de defeitos encontrados no caminho

### No modelo de dados e nos regimes

**A rampa de parada dura 121 a 127 s**, não 15. A regra original descartava 15 s no
fim de cada corrida, o que deixava os 45 s finais rotulados como operação com
velocidade média de 4,84 mm/s contra um platô de 6,61 — 27% abaixo. Contaminava o
treino com a desaceleração. Com descarte de 90 s o desvio em operação caiu de
0,94 para 0,25 mm/s.

**O corte treino/teste era global no transiente.** Os transientes estão espalhados
pelas 4 horas e o estado térmico muda ao longo da sessão, então os últimos 30%
ficavam fora da distribuição de treino. O falso positivo medido nesse regime era
de 62%; por bloco contíguo caiu para 0%.

### No caminho de inferência

**Timestamp com fuso derrubava o endpoint.** `pd.to_datetime(...).astype("datetime64[us]")`
levanta `TypeError` na entrada com fuso. O banco devolve `timestamptz` e o CSV não
tem fuso, então os testes sobre o CSV passavam enquanto o endpoint dava erro em
toda chamada real. Verificado depois da correção: as duas entradas dão o mesmo
score de 3,4293.

**O descarte de cabeça e cauda vazava para a inferência.** Aquele descarte é
higiene do conjunto de treino. Aplicado em produção, marcaria a última amostra de
qualquer janela como transiente em 100% dos casos. Criada `classificar_atual()`,
que decide o regime pela física.

**Componente sem modelo caía na baseline do S1.** O componente 1 é o motor da
FIAP, que é outro motor físico. Agora recusa explicitamente.

### Nas tools do agente

**As unidades estavam trocadas e induziam alucinação.** As tools de banco
imprimiam a aceleração em g com rótulo de mm/s, e a velocidade em mm/s como se
fosse rotação. O agente lia literalmente "o motor gira a 0,04 RPM e a vibração é
desconhecida". Os limites ISO de 2,8 e 4,5 mm/s eram aplicados sobre aceleração
em g, o que não tem significado físico.

**Zero virava "n/d".** `f"{v or 'n/d'}"` transforma 0.0 em 'n/d' porque zero é
falsy — e zero é justamente a leitura de aceleração com o motor parado, o estado
mais comum do histórico.

**Caminho de CSV errado.** `sensor_tool.py` apontava para a raiz, mas o arquivo
mora em `sensor/`, então o fallback local falhava com `FileNotFoundError`.

**SQL com interpolação de string** em `get_sensor_trends` trocado por
`make_interval` com parâmetro de bind.

### No frontend

**O hover apagava o sinal.** O `handlePointerOver` pintava a peça de verde
incondicionalmente, então uma peça vermelha de anomalia virava verde justamente
quando o operador passava o mouse para inspecioná-la. Agora peças com diagnóstico
mantêm a cor da severidade e só brilham mais.

**Duas interfaces divergentes quebravam o build de produção.** `HighlightInfo` e
`SegmentHighlight` descreviam o mesmo dado e uma exigia um campo que a outra não
tinha. Uma agora estende a outra, então não podem mais divergir.

**`tsc --noEmit` é um falso positivo neste projeto.** O `tsconfig.json` é só um
arquivo de referências e compila zero arquivos, saindo com sucesso sempre. O
comando que vale é **`tsc -b`** ou `npm run build`.

### No banco

**Migration 005 aplicada** (só o schema). As colunas foram acrescentadas ao modelo
ORM e o SQLAlchemy seleciona todas as colunas mapeadas, então o endpoint
`GET /componentes/{id}/limites` devolvia 500 enquanto a coluna não existisse.

---

## 8. Reprodutibilidade

Semente 42 em três lugares: `random_state` do scikit-learn,
`np.random.default_rng` e `torch.manual_seed`. Duas execuções na mesma máquina dão
resultado idêntico.

Os números deste documento são de torch 2.13.0+cpu, sklearn 1.9.0, pandas 3.0.5,
numpy 2.5.2. Entre máquinas, o Isolation Forest e a atribuição por feature são
idênticos; Mahalanobis varia na terceira casa e o autoencoder na segunda, conforme
a versão da biblioteca.

`detectores.joblib` (23 MB) está no `.gitignore`. A inferência reconstrói o
autoencoder do `state_dict` em `modelos.json`; Mahalanobis e Isolation Forest só
servem para a comparação da bancada, cujo resultado já está em `metricas.json`.
Verificado: renomear o joblib não afeta a inferência.
