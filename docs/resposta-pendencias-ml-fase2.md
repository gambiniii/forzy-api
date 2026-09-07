# Resposta às 7 pendências de ML da Fase 2

Cada item foi verificado lendo o código e medindo nos dados reais. Onde a
conclusão diverge da proposta original, o motivo está explicitado.

---

## 1. "Falta o domínio de frequência (BPFO/BPFI/BSF/FTF)"

**Procede que não existe. E é impossível implementar — não difícil, impossível.**

Confirmado que `feature_engineering.py` só tem estatística no tempo: nenhuma FFT,
envelope, cepstrum ou banda de energia.

O motivo de ser impossível é Nyquist, não esforço de engenharia:

- Não temos forma de onda. O sensor entrega **um escalar agregado por canal por
  leitura** (v-RMS, a-RMS, a-Peak, temperatura). A série que temos é a série
  desses agregados.
- Taxa dessa série: mediana de 0,60 s no arquivo histórico, o que dá Nyquist de
  **0,83 Hz**. Em produção o poller roda a 10 s, Nyquist de **0,05 Hz**.
- O rolamento do W22 carcaça 100, 2 polos é o **6206** (Tabela 8.1 do manual WEG,
  5 g de graxa, relubrificação a 25.000 h). A 3525 rpm suas frequências
  características são:

  | frequência | valor | múltiplo da rotação |
  |---|---|---|
  | FTF (gaiola) | 23,3 Hz | 0,40 × |
  | BSF (esfera) | 135,8 Hz | 2,31 × |
  | BPFO (pista externa) | 209,6 Hz | 3,57 × |
  | BPFI (pista interna) | 319,1 Hz | 5,43 × |

  Para observar a maior delas seria preciso amostrar a **638 Hz**. Estão de 260 a
  6500 vezes acima do nosso Nyquist. Não há aliasing recuperável nem truque de
  amostragem comprimida: o sensor já integrou essa banda inteira num único RMS
  antes de nos entregar.

**Corolário: distinguir pista interna de externa, esfera de gaiola é impossível**
— essa distinção *é* a assinatura de frequência. Qualquer tela que afirme
"defeito na pista externa" com este pipeline está inventando.

### O que É possível e foi implementado

**Fator de crest** (`a_peak / a_rms`). O a-Peak estava escondido nos bytes 8-9 do
array PDI e não é exportado em nenhuma coluna do CSV. É o indicador precoce
clássico: o defeito de pista gera impactos curtos de alta energia que quase não
alteram o valor eficaz, então o crest sobe antes do RMS. Mediana medida em regime
saudável: 3,39 no S1 e 3,19 no S2, exatamente a faixa de 3 a 4 esperada de
máquina em bom estado.

**Ressalva importante:** o crest é **offline-only** hoje. O `forzy_poller` consome
`GET /get_s1`, que devolve apenas `{Velocidade, Aceleração, Temperatura}`, e
`leitura_sensor` não tem coluna para pico. Levar o a-Peak até o banco é
pré-requisito para usar crest em produção.

**Sobre exibir BPFO/BPFI na interface:** calcular os valores teóricos a partir da
geometria do rolamento e da rotação é legítimo como contexto educativo. O que não
pode é afirmar que foram *detectados*. O agente foi instruído a rotulá-los
explicitamente como "frequência de referência, não medida por este sensor".

---

## 2. "MSE colapsado sobre as features"

**Procede integralmente. Corrigido, e a correção é exatamente aditiva.**

`_lstm_reconstruction_error` fazia `np.mean` sobre os três eixos, colapsando num
escalar. Toda a informação de *qual* feature reconstruiu mal era jogada fora ali.

A correção devolve `(escalar, erro_por_feature)`. O escalar é **numericamente
idêntico** ao anterior, verificado com igualdade bit a bit: a média sobre todos os
elementos é a mesma coisa que a média das médias por feature, porque todas têm a
mesma contagem de amostras. Logo `reconstruction_error`, `threshold`, `severity`,
`confidence` e `combined_health` mantêm o mesmo valor e nenhum consumidor quebra.

**Duas armadilhas que precisaram entrar junto**, senão o ranking é lixo:

1. O erro por feature está em **espaço escalado**. Uma feature com faixa de treino
   minúscula é amplificada e lideraria o ranking em *todas* as amostras, inclusive
   nas saudáveis — mediria qual feature o autoencoder aprendeu pior, não qual está
   anômala. Por isso a pontuação exige o baseline salvo no treino e usa z-score.
2. O padding do LSTM era feito com **zeros em espaço bruto**. Zero mm/s e zero g
   não são estado neutro, são "motor parado", longe da variedade de treino em
   regime. Com `MIN_ROWS = 5`, um ciclo podia rodar com 5 linhas reais e 55 de
   zeros. Trocado por repetição da borda.

---

## 3. "Calibração de threshold para decidir se é rolamento"

**Procede que falta, mas o enquadramento estava errado.**

A temperatura foi **removida de propósito** das features na V2 do
`build_features`. O LSTM e o Isolation Forest em produção **literalmente não
enxergam temperatura**. Não existe threshold a calibrar sobre um sinal que não
entra no modelo, e qualquer conclusão de "é térmico" saindo do erro de
reconstrução seria fabricação.

O caminho implementado tem duas camadas separadas:

- **Evidência térmica vem do Metric Contract determinístico**, que já funciona e é
  auditável: `_classify_threshold` produz `breached_metrics` com limites por
  componente vindos do banco.
- **Evidência mecânica vem do detector de novidade**, com limiar calibrado por
  validação cruzada de 5 dobras contíguas no tempo.

**Sobre calibrar estatisticamente, é preciso dizer em voz alta que não dá.** São
cerca de 19 minutos de regime estacionário por motor, uma única condição de carga
e zero exemplos de falha. Qualquer limiar escolhido é um **prior declarado**, não
um ajuste. As duas únicas saídas reais são assumir o prior explicitamente e
rotulá-lo como tal, ou usar bancada de injeção de falhas. Ambas foram feitas.

---

## 4. "O CWRU está no repo mas não serve pronto"

**Procede, e há um achado mais grave: o CWRU nunca é carregado em produção.**

`_resolve_model_dir` só aceita o diretório CWRU se `scaler.n_features_in_ == 61`.
O scaler CWRU tem **512**. O teste nunca passa, para IF nem para LSTM, e
`load_all_models` sempre cai no diretório Forzy. Ou seja, **os F1 de 0,9757 e
0,9846 anunciados no health endpoint medem artefatos que não estão no caminho de
inferência.** A constante 61 também estava errada: `build_features` produz 43.

Sobre a opção (a), treinar um classificador multiclasse no `fault_type` do CWRU:
é tecnicamente viável mas **não transfere**. O CWRU é forma de onda de 512 pontos
a 12 ou 48 kHz de um mancal de bancada; nossos dados são quatro escalares
agregados a 1,67 Hz de um motor diferente. Um classificador treinado lá não tem
como pontuar features daqui — é exatamente o que a incompatibilidade de 512 contra
43 já demonstra. A opção (b), usar o CWRU só para validar a heurística, esbarra no
mesmo problema.

Foi feito o que de fato responde à pergunta: **bancada de injeção de falhas sobre
a nossa própria série**, com seis modos calibrados fisicamente.

---

## 5. "Só existe um sensor físico"

**Procede e foi confirmado.** `leituras_to_raw_df` é chamado sem `port2_rows`
tanto no scheduler quanto no router, então port2 é cópia bit a bit de port1 e 21
das 43 features são duplicatas exatas. O posto efetivo da matriz em produção é
**22, não 43**.

Consequência aceita no projeto: **não é possível distinguir o rolamento do lado
acionado do lado não acionado.** A atribuição aponta os dois mancais juntos, e o
tooltip declara isso ao operador em vez de fingir precisão que não existe.

O novo pipeline não usa separação por porta: trabalha com uma série por motor
físico, decodificada do PDI da porta correspondente.

---

## 6. "Backend: expor o novo sinal"

Feito por endpoint em vez de coluna nova, o que evita migration e mantém o
diagnóstico existente intocado:

- `GET /diagnosticos/componente/{id}/atribuicao` — roda o detector na janela atual
  e devolve regime, score normalizado, severidade, falha provável, componentes
  candidatos, segmentos 3D a destacar e as features com maior desvio.
- `GET /diagnosticos/modelos/metricas` — métricas da bancada de injeção.

É **consultivo**: não sobrescreve `overall_status`. As duas fontes de destaque
coexistem no frontend e a regra de prioridade está em `combinarHighlight`.

Também foi adicionado o que faltava para a aceleração entrar em
`breached_metrics`: `componente_limite` ganhou `acel_atencao` e `acel_critico`
(migration 005). Sem eles a métrica nunca era emitida e os rolamentos jamais
podiam acender.

---

## 7. "Frontend: preencher aceleracao em motorSegmentMap"

Feito, **mas com segmentos diferentes dos propostos**.

A sugestão era `empty_19`, a capa do ventilador. A geometria do OBJ mostra que os
alojamentos de rolamento estão nas **tampas**, não na defletora:

- `empty_7` (tampa dianteira): 1778 vértices com raio menor que 45 em X de 120,3 a
  130,0, com raio mínimo de 25,0 — um furo de Ø50, que é alojamento de mancal.
- `empty_13` (tampa traseira): furo de Ø62, exatamente o diâmetro externo do 6206.
- `empty_19` é uma cúpula que envolve a tampa traseira e abriga o **ventilador**.
  Ela é o proxy correto para **temperatura**, não para rolamento.

Então: `aceleracao: ["empty_7", "empty_13"]` e `empty_19` entrou em `temperatura`.

Foram corrigidos também dois erros da Fase 1: `empty_6` é a **tampa da caixa de
ligação** e não a carcaça (que é `empty_2`), e `empty_24` é a **chaveta** e não a
ponta do eixo (que é `empty_23`). O logo WEG que motivou o rótulo original fica
justamente na tampa da caixa de ligação.

A regra de prioridade quando as duas fontes divergem: a severidade mais alta
vence, as duas mensagens aparecem, e o limite determinístico vem primeiro no texto
porque é o que o operador consegue auditar. O ML entra como evidência adicional,
nunca sozinho substituindo a regra.
