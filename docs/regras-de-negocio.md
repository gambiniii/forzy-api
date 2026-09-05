# Regras de Negócio — Forzy Digital Twin

**Projeto:** Forzy Digital Twin — Monitoramento Preditivo de Motores Industriais
**Parceria:** FIAP × Promon
**Documento:** Regras de negócio e critérios de decisão do sistema

---

## 1. Contexto e Objetivo

O Forzy Digital Twin é um sistema de **monitoramento preditivo** para motores elétricos industriais. Sensores físicos instalados nos motores capturam continuamente vibração, velocidade e temperatura; esses dados alimentam modelos de Machine Learning que estimam a saúde do equipamento, detectam anomalias e projetam a vida útil restante — permitindo que a manutenção seja **planejada antes da falha ocorrer**, em vez de reativa.

O objetivo de negócio é reduzir paradas não planejadas e custos de manutenção corretiva, substituindo a inspeção periódica manual por um acompanhamento contínuo e automatizado da condição real de cada motor.

## 2. Estrutura Organizacional

O sistema organiza os ativos monitorados em três níveis hierárquicos:

- **Planta** — a instalação física/cliente (ex.: "Forzy - Promon"). Agrupa todas as máquinas de uma mesma unidade.
- **Máquina** — o motor em si (ex.: "Motor WEG W22 - Unidade S1"). Cada máquina pertence a uma planta.
- **Componente** — o conjunto monitorado dentro da máquina (hoje, o próprio motor). Cada máquina tem um componente associado, e é o componente que efetivamente recebe leituras de sensor e diagnósticos.

Essa separação existe porque, no mundo real, uma planta pode ter várias máquinas, e futuramente uma máquina poderia ter mais de um componente monitorado (ex.: motor + redutor).

## 3. Coleta de Dados dos Sensores

Cada motor monitorado possui sensores físicos que reportam, a cada ciclo de coleta:

- **Velocidade/Vibração** (mm/s) — indicador primário de desgaste mecânico (desalinhamento, desbalanceamento, folgas, falha de rolamento).
- **Temperatura** (°C) — indicador de sobrecarga térmica ou falha de refrigeração.

Essas leituras são armazenadas com carimbo de data/hora e associadas ao componente correspondente, formando um histórico contínuo usado tanto para visualização em tempo real quanto para alimentar os modelos de diagnóstico.

## 4. Estado Operacional do Motor

Antes de qualquer diagnóstico, o sistema verifica se o motor está **de fato operando**: se a velocidade média medida está **abaixo de um piso mínimo (0,3 mm/s)**, o motor é considerado **desligado**, e nenhuma análise de saúde é realizada — evitaria diagnosticar "normalidade" ou "falha" em um motor parado, o que não teria significado prático.

Quando o motor está desligado:
- Não é gerado health score, RUL ou classificação de risco.
- A recomendação exibida é "Motor fora de operação. Análise de anomalia não aplicável."

Quando o motor está operando, o pipeline de diagnóstico completo (seção 5) é executado.

## 5. Diagnóstico por Inteligência Artificial

Para cada componente em operação, três modelos de ML são combinados para produzir um diagnóstico:

| Modelo | O que avalia |
|---|---|
| **Isolation Forest** | Detecta se o padrão de vibração/temperatura atual é estatisticamente anômalo em relação ao comportamento histórico normal do motor. |
| **LSTM Autoencoder** | Avalia a severidade de um desvio, com base no erro de reconstrução de uma janela temporal recente — quanto maior o erro, mais grave o desvio do padrão esperado. |
| **XGBoost RUL** | Estima a Vida Útil Restante (Remaining Useful Life) em horas, com base na tendência de degradação observada. |

### 5.1 Saúde do Motor (Health Score)

O **health score** (0–100%) é a métrica central de saúde do motor. É calculado combinando as duas primeiras análises:

> `Saúde = 40% × (normalidade segundo Isolation Forest) + 60% × (normalidade segundo severidade LSTM)`

A severidade do LSTM é convertida em pontuação de normalidade segundo a tabela:

| Severidade LSTM | Peso de normalidade |
|---|---|
| Normal | 100% |
| Baixa | 50% |
| Média | 30% |
| Alta | 10% |
| Crítica | 0% |

### 5.2 Classificação de Risco (a partir do RUL)

A vida útil restante estimada (em horas) é traduzida em um nível de risco:

| RUL estimado | Nível de risco |
|---|---|
| Acima de 5.000h | Baixo |
| Entre 1.000h e 5.000h | Médio |
| Entre 100h e 1.000h | Alto |
| Abaixo de 100h | Crítico |

### 5.3 Status Geral do Motor

O status apresentado ao operador combina saúde e risco:

- **Crítico** — saúde abaixo de 40%, ou risco de RUL crítico.
- **Saudável** — saúde igual ou acima de 70% **e** risco de RUL baixo.
- **Atenção** — demais casos (saúde intermediária ou risco elevado sem ainda ser crítico).
- **Motor desligado** — quando não há operação (seção 4).

### 5.4 Detecção de Anomalias

Um diagnóstico é marcado como **anomalia** sempre que o Isolation Forest **ou** o LSTM Autoencoder identificam um desvio significativo — não é necessário que os dois modelos concordem; qualquer um dos dois sinalizando já é suficiente para o alerta, priorizando sensibilidade (não deixar passar um problema real) sobre a ausência de falsos positivos.

### 5.5 Recomendações Automáticas

Cada diagnóstico gera uma recomendação textual em linguagem natural, coerente com o status apurado (ex.: "Motor operando normalmente", "Padrão de operação irregular detectado na vibração. Agendar inspeção."), servindo como orientação direta de ação para a equipe de manutenção, sem exigir interpretação técnica dos números brutos.

### 5.6 Periodicidade do Diagnóstico

O diagnóstico não é recalculado a cada leitura individual — ele roda em um ciclo periódico (a cada 5 minutos), sempre considerando a janela mais recente de leituras acumuladas. Isso reflete a natureza do problema: o desgaste mecânico é um processo gradual, não instantâneo, e não faz sentido de negócio reavaliar a saúde do motor a cada poucos segundos.

## 6. Indicador de Conectividade (Online/Offline)

Além do estado operacional do motor (ligado/desligado), o painel exibe se a **conexão de dados** está ativa: se uma leitura nova chegou nos últimos 30 segundos, o motor aparece como **Online**; caso contrário, **Offline**.

Este indicador responde a uma pergunta diferente da do estado operacional: ele não diz se o motor está funcionando, mas se o sistema está **recebendo dados frescos dele agora**. Um motor pode estar Online (conexão viva) e ainda assim Desligado (parado fisicamente) — são informações independentes.

## 7. Histórico e Rastreabilidade

Todo diagnóstico gerado é armazenado permanentemente, formando um histórico auditável por motor. Isso permite:

- Visualizar a **tendência de saúde** do motor ao longo do tempo (se está piorando, estável ou se recuperando).
- Calcular a **proporção de tempo** que o motor passou em cada classificação (Normal / Atenção / Crítico).
- Identificar quando ocorreu a **última anomalia** registrada.
- Consultar o histórico de leituras brutas de sensor por período (última hora, 6h, 24h, 7 dias).

## 8. Especificações Técnicas do Motor

Cada motor cadastrado possui uma ficha técnica (placa de identificação): potência, tensão nominal, corrente nominal, rotação nominal, frequência, número de polos e rendimento. Essas informações contextualizam o diagnóstico — servem de referência de projeto do fabricante para a equipe técnica interpretar o comportamento observado, mas **não alimentam diretamente o cálculo do health score**, que se baseia exclusivamente no comportamento estatístico das leituras de vibração/temperatura ao longo do tempo.

## 9. Perfis de Acesso

O sistema distingue perfis de usuário (ex.: administrador, técnico, cliente) com diferentes níveis de permissão sobre operações de cadastro e edição de máquinas, componentes e atributos técnicos — o acompanhamento de leituras e diagnósticos é acessível a qualquer usuário autenticado.
