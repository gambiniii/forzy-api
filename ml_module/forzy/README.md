# Detecção de novidade Forzy — 6 modelos, atribuição por componente

Pipeline que lê a telemetria real dos dois motores WEG W22, treina detectores de
novidade **apenas em dado saudável** e atribui a anomalia a um conjunto de
componentes físicos, que é o que faz uma peça do modelo 3D ficar vermelha.

```bash
python -m ml_module.forzy.train     # treina e salva em ml_module/forzy/saved/
```

## Por que 6 modelos e não um

São **2 motores × 3 regimes**.

**Por regime**, porque as distribuições não se sobrepõem. Parado e operação estão
separados por duas ordens de grandeza em velocidade. Um modelo único aceitaria
tudo que existe entre eles, que é justamente onde vive a perda de acionamento.

**Por ativo**, porque as assinaturas diferem. Medido nos dados: o motor S2 opera
4,39 °C mais quente e vibra 0,182 mm/s a mais que o S1 em regime, com p < 1e-10.

O regime **transiente é pontuado mas não alarma**, por projeto. Durante partida e
parada por inércia a vibração varia legitimamente em uma ordem de grandeza.
Modelá-lo mesmo assim é necessário porque uma perda parcial de acionamento
derruba a velocidade exatamente para essa faixa.

## Os módulos

| arquivo | o que faz |
|---|---|
| `etl.py` | decodifica os 4 canais do PDI e trata o latch do a-Peak |
| `regimes.py` | segmenta parado, transiente e operação |
| `features.py` | as 16 features, todas com janela temporal |
| `scaling.py` | padronização robusta com piso e clipping |
| `detectors.py` | Mahalanobis robusta, Isolation Forest e autoencoder |
| `injection.py` | bancada de injeção de falhas, 6 modos × 3 severidades |
| `calibration.py` | limiar por validação cruzada com dobras contíguas |
| `attribution.py` | do desvio nos sinais para os componentes candidatos |
| `train.py` | orquestra tudo e salva os artefatos |
| `infer.py` | o que a API chama em produção |

## O quarto canal escondido

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
começa a deteriorar: o defeito de pista gera impactos curtos de alta energia que
quase não alteram o valor eficaz. É o indicador precoce clássico, disponível sem
custo de instrumentação.

Varredura dos 16 bytes confirmou que **não existe um quinto canal**: os demais
são constantes nas duas portas.

## Armadilhas tratadas

**O a-Peak é latched.** O sensor recalcula o pico a cada 9,02 s (medido) enquanto
os outros canais atualizam a cada 0,6 s. Entre refreshes ele *segura* o valor, e
os zeros com eixo girando são blackouts de partida, concentrados no transiente
(61,3% deles) em rajadas de até 52 s. Tratamento: com aceleração acima de 0,05 g,
zero é ausência e é preenchido para frente por no máximo 30 s.

**Timestamps duplicados.** Sete registros repetem o instante exato, o que quebra
indexação por rótulo no pandas. Toda a lógica é posicional, com numpy.

**O tipo é `datetime64[us]`, não `[ns]`.** Converter com `.astype("int64")`
produz valores mil vezes menores. Sempre `.total_seconds()` sobre a diferença.

**Janelas temporais, nunca por contagem de amostras.** Com amostragem irregular
(mediana 0,60 s, p99 31 s, lacunas de até 228 s), uma janela de N amostras teria
duração física variável.

**A rampa de parada dura 121 a 127 s.** A especificação original descartava 15 s
no fim de cada corrida, o que deixava os 45 s finais rotulados como operação com
velocidade média de 4,84 mm/s contra um platô de 6,61 — 27% abaixo. Isso
contaminava o treino com a desaceleração. Aqui o descarte é de 90 s, e o desvio
em operação caiu de 0,94 para 0,25 mm/s.

## Padronização com piso e clipping

Dentro de um único regime várias features são quase constantes: a aceleração com
o motor parado tem desvio de 0,001 g, e dividir por ele transforma ruído de
quantização em z-scores de centenas. O piso é 5% do desvio global da feature e o
z-score final é clipado em 10.

## Calibração do limiar

Duas abordagens naturais falham, ambas medidas neste conjunto:

- **Corte cronológico simples** (últimos 20% do treino): em operação isso cai numa
  fatia de 2 min 42 s de uma única corrida, com desvio 14 vezes mais estreito que
  o restante. O limiar sai altíssimo e o detector deixa de alarmar.
- **Corte aleatório**: os pontos de calibração ficam intercalados no tempo com os
  de ajuste, praticamente idênticos. O modelo os reconstrói bem demais, o limiar
  sai baixo e o falso positivo real chega a 28%.

A solução é **validação cruzada com 5 dobras contíguas no tempo**. Cada dobra é
pontuada por um modelo ajustado sem ela.

## A agregação do erro: média mais máximo

Não é o erro quadrático médio puro. O EQM dilui desvio concentrado: um sensor
travado desloca uma feature das 16, cujo erro de reconstrução chega a 132 vezes o
normal, mas diluído em 16 termos mal move a média. Somar o máximo recupera esse
caso sem perder o anterior — a média responde a desvios difusos, o máximo a
desvios concentrados num único canal.

## Bancada de injeção de falhas

Sem rótulos não existe recall, F1 nem PR-AUC. A saída é perturbar o dado de teste
saudável com degradações fisicamente calibradas.

| modo | assinatura | perfil |
|---|---|---|
| F1 Desbalanceamento | velocidade até 14,7 mm/s, aceleração sobe pouco, crest cai | rampa |
| F2 Sobreaquecimento | temperatura +22 °C, vibração quase inalterada | rampa |
| F3 Rolamento | a-Peak dispara, crest de 3,2 a 6,8, velocidade só +18% | rampa com pulsos |
| F4 Sensor travado | canais congelam em valores nominais | degrau |
| F5 Deriva de calibração | ganho errado até +40%, valores plausíveis | degrau |
| F6 Parada não programada | acionamento se perde | degrau |

Quatro cuidados obrigatórios: série contínua e não janelas soltas; perturbar os
canais brutos e recalcular as features do zero; **requantizar** depois de injetar,
senão o detector acusa a falha pela perda de quantização em vez da física; e
descartar a zona cinza de 60 s após cada episódio, em que as janelas móveis ainda
carregam amostras contaminadas.

Os episódios são ancorados **dentro dos trechos de operação**. Uma falha de
rolamento não se manifesta com o eixo parado, e neste dataset a operação em
regime só acontece em dois trechos, a cerca de 42 min e 2 h do início.

## O que é impossível com estes dados

1. **Detectar BPFO, BPFI, BSF ou FTF.** É Nyquist, não falta de esforço. O sensor
   entrega um escalar agregado por leitura, não forma de onda. As frequências do
   rolamento 6206 a 3525 rpm ficam entre 23 e 319 Hz e exigiriam amostragem
   mínima de 638 Hz; a nossa é 1,67 Hz no arquivo e 0,1 Hz em produção.
2. **Distinguir pista interna de externa, esfera ou gaiola.** É a mesma
   impossibilidade: essa distinção *é* a assinatura de frequência.
3. **Distinguir o rolamento do lado acionado do não acionado.** Um sensor por motor.
4. **Apontar uma peça entre as 23 com evidência.** O teto honesto são grupos
   funcionais.
5. **Qualquer classificador supervisionado de falha.** Zero rótulos.
6. **Fator de crest em produção.** O a-Peak só existe no arquivo histórico; o
   poller recebe apenas velocidade, aceleração e temperatura.

## Ressalva sobre as métricas

As falhas são **injetadas, não observadas**. Isso prova que o detector reconhece
desvios com a assinatura física esperada e quantifica a margem entre normal e
anômalo. **Não prova** que detectaria uma falha real, que tem componentes não
modelados: espectro, modulação, interação com a carga. As métricas são um
**limite superior otimista**.

A base de treino são cerca de 19 minutos de regime estacionário por motor, num
único dia, sem variação de carga. A faixa de condições normais está subamostrada.

## Reprodutibilidade

Semente 42 em três lugares: `random_state` do scikit-learn,
`np.random.default_rng` e `torch.manual_seed`. Duas execuções na mesma máquina
dão resultado idêntico. Entre máquinas, o Isolation Forest e a atribuição por
feature são idênticos; Mahalanobis varia na terceira casa e o autoencoder na
segunda, conforme a versão da biblioteca.
