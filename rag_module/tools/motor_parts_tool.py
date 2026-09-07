"""
Base de conhecimento das peças do motor WEG W22 3cv monofásico + tool LangChain.

Espelha `forzy-web-master/src/config/motorParts.ts` (os 23 segmentos do modelo 3D)
e acrescenta o material que só faz sentido no agente: a tabela oficial
PROBLEMAS X SOLUÇÕES do manual WEG 50033244 (cap. 10) e os limites normativos.

ORIGEM DOS DADOS — tudo verificado em documento oficial ou medido no dataset:
  • Nomes das peças: vista explodida WEG 50106446 + geometria do Engine1.obj
    (modelo em mm, escala 1:1; altura de eixo 100 mm = carcaça 100L).
  • Rolamento: Tabela 8.1 do manual — carcaça 100, 2 polos → 6206, 5 g de graxa,
    relubrificação a cada 25.000 h (linha W22, 60 Hz).
  • Limites de vibração: Tabela 7.7 (ISO 20816-3, saída ≤ 300 kW, Grupo 2).
  • Baselines: medidos nas 3h57 de telemetria real dos dois motores.
"""

from __future__ import annotations

from langchain_core.tools import tool

# ── Baselines MEDIDOS na telemetria real (não são valores de norma) ──────────
BASELINE = {
    "OPERACAO": {
        "v_rms_mm_s": (6.504, 0.941),   # média, desvio
        "a_rms_g":    (0.484, 0.074),
        "crest":      (3.218, 0.812),
        "temp_c":     (34.41, 3.24),
        "n":          4448,
    },
    "TRANSIENTE": {"v_rms_mm_s": (1.974, 1.746), "a_rms_g": (0.143, 0.151), "temp_c": (32.02, 4.31), "n": 3808},
    "PARADO":     {"v_rms_mm_s": (0.049, 0.032), "a_rms_g": (0.000, 0.001), "temp_c": (36.50, 5.54), "n": 6110},
}

# ── As 23 peças ──────────────────────────────────────────────────────────────
PECAS: dict[str, dict] = {
    "empty_2": {
        "nome": "Carcaça",
        "grupo": "Sistema térmico / estrutural",
        "descricao": "Corpo do motor com aletas de refrigeração e pés B3D. Aloja o estator bobinado e é a principal superfície de troca de calor.",
        "internos": ["Estator bobinado", "Núcleo do estator", "Isolamento classe F", "Pés de fixação", "Placa de identificação"],
        "sinais": {"velocidade": "média", "aceleracao": "fraca", "temperatura": "forte"},
        "falhas": [
            ("Sobreaquecimento do enrolamento", "Temperatura sobe de forma sustentada com a vibração inalterada.", "tardio"),
            ("Dissipação prejudicada por sujeira", "Temperatura sobe gradualmente ao longo de dias, sem mudança na vibração.", "precoce"),
            ("Fixação frouxa / ressonância da base", "Picos de vibração intermitentes com o nível médio normal.", "precoce"),
        ],
        "destacavel": True,
    },
    "empty_7": {
        "nome": "Tampa dianteira (lado acionado)",
        "grupo": "Sistema mecânico rotativo",
        "descricao": "Fecha o motor do lado da ponta do eixo e aloja o rolamento dianteiro. É onde a norma manda medir vibração.",
        "internos": ["Rolamento dianteiro 6206", "Vedação V-Ring dianteira", "Arruela ondulada"],
        "sinais": {"velocidade": "forte", "aceleracao": "forte", "temperatura": "média"},
        "falhas": [
            ("Rolamento desgastado", "Aceleração sobe com picos curtos de alta energia, elevando o fator de crest ANTES do valor eficaz.", "precoce"),
            ("Lubrificação inadequada", "Temperatura do mancal sobe junto com a aceleração; ruído aumenta.", "tardio"),
            ("Desalinhamento do acoplamento", "Velocidade de vibração sobe de forma sustentada, concentrada em 1x a rotação.", "tardio"),
        ],
        "destacavel": True,
    },
    "empty_13": {
        "nome": "Tampa traseira (lado não acionado)",
        "grupo": "Sistema mecânico rotativo",
        "descricao": "Fecha o motor do lado do ventilador, aloja o rolamento traseiro e cobre o mecanismo de partida do motor monofásico.",
        "internos": ["Rolamento traseiro 6206", "Vedação V-Ring traseira", "Centrífugo (mecanismo de partida)", "Platinado"],
        "sinais": {"velocidade": "média", "aceleracao": "forte", "temperatura": "média"},
        "falhas": [
            ("Rolamento traseiro desgastado", "Aceleração sobe com crest crescente; velocidade quase inalterada no início.", "precoce"),
            ("Falha do mecanismo centrífugo", "Anormalidade só na partida: demora a atingir rotação e aquece mais nos primeiros minutos.", "precoce"),
        ],
        "destacavel": True,
    },
    "empty_19": {
        "nome": "Tampa defletora",
        "grupo": "Sistema térmico",
        "descricao": "Capa que protege o ventilador e direciona o ar sobre as aletas. Na refrigeração IC411 o ventilador é acoplado ao eixo: parado o motor, para a ventilação.",
        "internos": ["Ventilador", "Grelha de entrada de ar", "Fluxo de ar IC411"],
        "sinais": {"velocidade": "fraca", "aceleracao": "fraca", "temperatura": "forte"},
        "falhas": [
            ("Ventilação obstruída", "Temperatura sobe de forma sustentada com vibração normal. É a causa número um de sobretemperatura segundo o manual WEG.", "precoce"),
            ("Ventilador danificado ou desbalanceado", "Aceleração sobe moderadamente e a temperatura acompanha por perda de refrigeração.", "tardio"),
        ],
        "destacavel": True,
    },
    "empty_23": {
        "nome": "Eixo",
        "grupo": "Sistema de transmissão",
        "descricao": "Transmite o torque do rotor para a máquina acionada. Escalonado Ø30 no assento do rolamento e Ø28 na ponta. É o proxy visível do rotor.",
        "internos": ["Rotor (gaiola)", "Núcleo do rotor", "Assento do rolamento"],
        "sinais": {"velocidade": "forte", "aceleracao": "média", "temperatura": "fraca"},
        "falhas": [
            ("Desbalanceamento do rotor", "Velocidade sobe de forma sustentada e proporcional à rotação, com a aceleração subindo menos que a velocidade.", "tardio"),
            ("Eixo empenado ou desalinhado", "Vibração oscila periodicamente acompanhando a rotação; temperatura sobe por consequência.", "tardio"),
        ],
        "destacavel": True,
    },
    "empty_4": {
        "nome": "Caixa de ligação",
        "grupo": "Sistema elétrico",
        "descricao": "Aloja placa de bornes, aterramento e capacitores. Em motor monofásico é grande porque precisa acomodar os capacitores de partida e permanente.",
        "internos": ["Capacitor de partida", "Capacitor permanente", "Placa de bornes", "Aterramento", "Prensa-cabos"],
        "sinais": {"velocidade": "fraca", "aceleracao": "fraca", "temperatura": "média"},
        "falhas": [
            ("Capacitor degradado", "O motor demora a atingir a rotação nominal e aquece mais, com a vibração pouco alterada.", "tardio"),
            ("Conexão frouxa ou mau contato", "Temperatura sobe sem causa mecânica aparente; vibração normal.", "tardio"),
        ],
        "destacavel": True,
    },
    # Detalhes visuais — nunca acendem sozinhos
    "empty_6":  {"nome": "Tampa da caixa de ligação", "grupo": "Sistema elétrico", "descricao": "Chapa externa com o logo WEG em relevo que fecha a caixa de ligação e garante o IP55.", "internos": [], "sinais": {}, "falhas": [], "destacavel": False},
    "empty_5":  {"nome": "Junta da tampa da caixa de ligação", "grupo": "Vedação IP55", "descricao": "Lâmina de vedação entre a caixa de ligação e sua tampa.", "internos": [], "sinais": {}, "falhas": [], "destacavel": False},
    "empty_3":  {"nome": "Olhal de içamento", "grupo": "Estrutural", "descricao": "Ponto de suspensão para transporte e instalação. Não participa da operação.", "internos": [], "sinais": {}, "falhas": [], "destacavel": False},
    "empty_24": {"nome": "Chaveta", "grupo": "Sistema mecânico rotativo", "descricao": "Peça de 8x7 mm no rasgo do eixo que transmite torque ao elemento acoplado. Dimensão DIN 6885 para eixo Ø28-30.", "internos": [], "sinais": {}, "falhas": [("Folga ou desgaste da chaveta", "Aceleração sobe em impactos intermitentes; velocidade média normal.", "precoce")], "destacavel": False},
    "empty_9":  {"nome": "Dreno (tampa dianteira)", "grupo": "Vedação IP55", "descricao": "Escoa condensado do interior do motor.", "internos": [], "sinais": {}, "falhas": [], "destacavel": False},
    "empty_18": {"nome": "Dreno (tampa traseira)", "grupo": "Vedação IP55", "descricao": "Dreno do lado não acionado.", "internos": [], "sinais": {}, "falhas": [], "destacavel": False},
}

for _ids, _onde in (
    (("empty_8", "empty_10", "empty_11", "empty_12"), "tampa dianteira"),
    (("empty_14", "empty_15", "empty_16", "empty_17"), "tampa traseira"),
    (("empty_20", "empty_21", "empty_22"), "tampa defletora"),
):
    for _id in _ids:
        PECAS[_id] = {
            "nome": f"Parafuso de fixação ({_onde})",
            "grupo": "Estrutural",
            "descricao": f"Prende a {_onde} à carcaça. Soltura é causa reconhecida de ruído e vibração, mas nenhum sinal isola um parafuso específico.",
            "internos": [], "sinais": {}, "falhas": [], "destacavel": False,
        }

# ── Tabela oficial PROBLEMAS X SOLUÇÕES (manual WEG 50033244, cap. 10) ───────
PROBLEMAS_WEG = {
    "ruído elevado / vibração": [
        ("Defeito nos componentes de transmissão ou na máquina acionada", "Verificar transmissão, acoplamento e alinhamento"),
        ("Base desalinhada / desnivelada", "Realinhar e nivelar o motor e a máquina acionada"),
        ("Desbalanceamento dos componentes ou da máquina acionada", "Realinhar e nivelar"),
        ("Balanceamento diferente entre motor e acoplamento (meia chaveta vs. chaveta inteira)", "Refazer balanceamento"),
        ("Sentido de rotação errado", "Inverter o sentido de rotação"),
        ("Parafusos de fixação soltos", "Reapertar os parafusos"),
        ("Ressonância da fundação", "Verificar o projeto da fundação"),
        ("Rolamentos danificados", "Substituir o rolamento"),
    ],
    "aquecimento excessivo no motor": [
        ("Refrigeração insuficiente", "Limpar entradas e saídas de ar da defletora e da carcaça; verificar distâncias mínimas e a temperatura do ar na entrada"),
        ("Sobrecarga", "Medir a corrente do motor e, se necessário, diminuir a carga"),
        ("Excessivo número de partidas ou inércia da carga muito elevada", "Reduzir o número de partidas"),
        ("Tensão muito alta ou muito baixa", "Verificar a tensão de alimentação e a queda de tensão"),
        ("Interrupção de um cabo de alimentação", "Verificar a conexão de todos os cabos"),
        ("Desequilíbrio de tensão nos terminais", "Verificar fusíveis queimados, comandos errados, desequilíbrio da rede ou falta de fase"),
        ("Sentido de rotação incompatível com ventilador unidirecional", "Verificar o sentido conforme marcação do motor"),
    ],
    "aquecimento do mancal": [
        ("Graxa ou óleo em demasia", "Limpar o mancal e lubrificar conforme as recomendações"),
        ("Envelhecimento da graxa ou do óleo", "Limpar e relubrificar"),
        ("Graxa ou óleo não especificados", "Limpar e relubrificar com o especificado"),
        ("Falta de graxa ou óleo", "Lubrificar conforme as recomendações"),
        ("Excessivo esforço axial ou radial", "Reduzir a tensão nas correias; redimensionar a carga aplicada"),
    ],
    "motor não parte": [
        ("Interrupção na alimentação", "Verificar o circuito de comando e os cabos"),
        ("Fusíveis queimados", "Substituir os fusíveis"),
        ("Erro na conexão do motor", "Corrigir conforme o diagrama de conexão"),
        ("Mancal travado", "Verificar se o mancal gira livremente"),
    ],
}

# ── O que NÃO dá para afirmar com esta telemetria ────────────────────────────
LIMITACOES = """
LIMITAÇÕES REAIS DESTE SISTEMA — nunca afirme além disto:

1. NÃO é possível detectar BPFO, BPFI, BSF ou FTF (frequências características do
   rolamento). Motivo: Nyquist, não falta de esforço. O sensor entrega um escalar
   agregado por leitura, não forma de onda. As frequências do rolamento 6206 a
   3525 rpm ficam entre 23 e 319 Hz e exigiriam amostragem mínima de 638 Hz; a
   nossa é de 1,67 Hz no arquivo histórico e 0,1 Hz em produção.
2. Por consequência, NÃO é possível distinguir defeito de pista interna, pista
   externa, esfera ou gaiola — essa distinção É a assinatura de frequência.
3. NÃO é possível distinguir o rolamento do lado acionado do lado não acionado:
   há um único sensor por motor.
4. NÃO é possível apontar uma peça específica entre as 23 com evidência. O teto
   honesto são grupos funcionais (rolamentos/eixo, rotor, fixação, térmico).
5. NÃO existe classificador supervisionado de falha: o histórico é 100% saudável,
   sem nenhum rótulo. A detecção é de novidade, não de classe.
6. O fator de crest (a_peak/a_rms) só existe no arquivo histórico. O poller de
   produção recebe apenas Velocidade, Aceleração e Temperatura, e a tabela
   leitura_sensor não tem coluna de pico.
"""

FENOMENOS_NORMAIS = """
DOIS COMPORTAMENTOS QUE PARECEM FALHA E NÃO SÃO — medidos na telemetria real:

1. HEAT SOAK (pós-desligamento). Depois de uma corrida longa, a temperatura da
   carcaça CONTINUA SUBINDO com o motor já parado: medido 38 °C no fim da corrida
   subindo até um pico de 46 °C cerca de 7,9 minutos depois de desligar, uma
   sobre-elevação de +8 °C. Causa: parado o eixo, para o ventilador IC411, mas o
   calor acumulado nos enrolamentos segue migrando para a carcaça. Existe uma
   região perfeitamente saudável do espaço de operação com temperatura de 46-47 °C
   e vibração de 0,04 mm/s. Se a temperatura está alta com o motor recém-desligado,
   isso é resfriamento normal, não sobreaquecimento.

2. DERIVA TÉRMICA DO PLATÔ. Durante uma corrida longa a vibração CAI enquanto o
   motor aquece: medido 6,95 → 6,42 mm/s (−0,038 mm/s por minuto) com a carcaça
   indo de 32 a 37 °C. A explicação física provável é a viscosidade do lubrificante
   caindo com o aquecimento. Ressalva honesta: neste dataset tempo de funcionamento
   e temperatura são colineares dentro de uma corrida, então não dá para separar o
   efeito de um do outro. Queda de vibração NUNCA indica degradação.
"""


def _fmt_peca(seg: str, p: dict) -> str:
    linhas = [f"### {p['nome']}  (segmento 3D `{seg}`)", f"Sistema: {p['grupo']}", p["descricao"]]
    if p.get("internos"):
        linhas.append("Componentes internos que esta peça representa: " + ", ".join(p["internos"]))
    if p.get("sinais"):
        linhas.append("Relação com os sinais: " + ", ".join(f"{k} = {v}" for k, v in p["sinais"].items()))
    if p.get("falhas"):
        linhas.append("Modos de falha:")
        for nome, assin, prec in p["falhas"]:
            linhas.append(f"  - {nome} ({prec}): {assin}")
    linhas.append(f"Pode ser destacada no modelo 3D: {'sim' if p['destacavel'] else 'não (detalhe visual)'}")
    return "\n".join(linhas)


@tool
def get_motor_part_info(peca: str = "") -> str:
    """Retorna a ficha técnica de uma peça do motor WEG W22: função, componentes
    internos que ela abriga, modos de falha e como cada modo aparece nos sinais
    de velocidade de vibração, aceleração e temperatura.

    Use SEMPRE que o usuário perguntar sobre uma peça específica do motor, sobre
    o que pode estar causando uma anomalia, ou quando ele vier do modelo 3D
    perguntando sobre um componente destacado.

    Parâmetros:
      peca: nome da peça em português (ex: "rolamento", "carcaça", "ventilador",
            "eixo", "tampa dianteira", "caixa de ligação") ou o id do segmento
            3D (ex: "empty_7"). Deixe vazio para listar todas as peças.
    """
    termo = (peca or "").strip().lower()

    if not termo:
        destac = [f"- {p['nome']} (`{s}`) — {p['grupo']}" for s, p in PECAS.items() if p["destacavel"]]
        outras = [f"- {p['nome']} (`{s}`)" for s, p in PECAS.items() if not p["destacavel"]]
        return (
            "PEÇAS DO MOTOR WEG W22 MAPEADAS NO MODELO 3D (23 segmentos)\n\n"
            "Peças que o diagnóstico pode destacar:\n" + "\n".join(destac) +
            "\n\nDetalhes visuais (nunca acendem sozinhos):\n" + "\n".join(outras) +
            "\n\nPergunte por uma peça específica para ver a ficha completa."
        )

    achados = [
        (s, p) for s, p in PECAS.items()
        if termo in s.lower() or termo in p["nome"].lower()
        or any(termo in i.lower() for i in p.get("internos", []))
    ]
    if not achados:
        return (
            f"Nenhuma peça encontrada para '{peca}'. Peças disponíveis: "
            + ", ".join(sorted({p["nome"] for p in PECAS.values()}))
        )

    # De-duplica parafusos idênticos
    vistos, saida = set(), []
    for s, p in achados:
        if p["nome"] in vistos:
            continue
        vistos.add(p["nome"])
        saida.append(_fmt_peca(s, p))
    return "\n\n".join(saida)


@tool
def diagnose_by_signals(
    velocidade_alta: bool = False,
    aceleracao_alta: bool = False,
    temperatura_alta: bool = False,
) -> str:
    """Dada a combinação de sinais que saíram do limite, retorna os componentes
    candidatos ordenados por probabilidade, com a causa provável e a ação
    recomendada segundo o manual oficial WEG.

    Use quando o usuário perguntar "o que pode ser?" a partir de um padrão de
    leituras, ou quando vier do modelo 3D com uma peça destacada.

    Parâmetros (marque os que estão FORA do limite):
      velocidade_alta: velocidade de vibração acima do limite (mm/s)
      aceleracao_alta: aceleração acima do limite (g)
      temperatura_alta: temperatura acima do limite (°C)
    """
    if not any((velocidade_alta, aceleracao_alta, temperatura_alta)):
        return (
            "Nenhum sinal fora do limite. Sem desvio, não há atribuição de componente — "
            "e é importante dizer isso ao usuário em vez de eleger uma peça qualquer.\n\n"
            + FENOMENOS_NORMAIS
        )

    blocos: list[str] = []
    padrao = []
    if velocidade_alta: padrao.append("velocidade de vibração alta")
    if aceleracao_alta: padrao.append("aceleração alta")
    if temperatura_alta: padrao.append("temperatura alta")
    blocos.append("PADRÃO OBSERVADO: " + " + ".join(padrao) + "\n")

    if aceleracao_alta and not velocidade_alta:
        blocos.append(
            "CANDIDATO PRINCIPAL: rolamento em degradação inicial.\n"
            "A aceleração subir com a velocidade ainda normal é a assinatura clássica de defeito "
            "de pista começando: o defeito gera impactos curtos de alta energia que quase não "
            "alteram o valor eficaz da velocidade. É o indicador PRECOCE.\n"
            "Peças: Tampa dianteira (rolamento 6206 do lado acionado), Tampa traseira (rolamento "
            "do lado não acionado). Um único sensor não distingue os dois lados.\n"
            "Ação WEG: verificar o estado dos mancais observando ruídos e vibração não habituais, "
            "a temperatura do mancal e a condição do lubrificante."
        )
    if velocidade_alta and not aceleracao_alta:
        blocos.append(
            "CANDIDATO PRINCIPAL: desbalanceamento ou desalinhamento.\n"
            "A velocidade subir sem a aceleração acompanhar indica energia concentrada em 1x a "
            "rotação, e não impacto de alta frequência.\n"
            "Peças: Eixo (proxy do rotor), Tampa dianteira, Carcaça (fixação e ressonância)."
        )
    if velocidade_alta and aceleracao_alta:
        blocos.append(
            "CANDIDATO PRINCIPAL: degradação mecânica já estabelecida.\n"
            "Os dois canais subindo juntos indica que o defeito passou da fase incipiente.\n"
            "Peças: rolamentos (tampas dianteira e traseira), Eixo, Carcaça."
        )
    if temperatura_alta and not (velocidade_alta or aceleracao_alta):
        blocos.append(
            "CANDIDATO PRINCIPAL: problema térmico ou elétrico, NÃO mecânico.\n"
            "A vibração inalterada afasta falha mecânica severa.\n"
            "Peças: Tampa defletora (ventilação obstruída é a causa nº 1 segundo o manual), "
            "Carcaça (estator, sobrecarga), Caixa de ligação (capacitor, mau contato).\n"
            "ANTES DE ALARMAR: confirme se o motor não foi desligado há menos de 15 minutos — "
            "o heat soak faz a temperatura subir até 8 °C com o motor já parado, e isso é normal."
        )
    if temperatura_alta and (velocidade_alta or aceleracao_alta):
        blocos.append(
            "CANDIDATO PRINCIPAL: rolamento com atrito elevado.\n"
            "Vibração e temperatura subindo juntas é a cadeia clássica: desgaste → vibração → "
            "atrito → aquecimento.\n"
            "Peças: Tampa dianteira e Tampa traseira (rolamentos 6206)."
        )

    if temperatura_alta:
        blocos.append("CAUSAS OFICIAIS WEG — aquecimento excessivo:\n" + "\n".join(
            f"  - {c}: {a}" for c, a in PROBLEMAS_WEG["aquecimento excessivo no motor"]))
        blocos.append("CAUSAS OFICIAIS WEG — aquecimento do mancal:\n" + "\n".join(
            f"  - {c}: {a}" for c, a in PROBLEMAS_WEG["aquecimento do mancal"]))
    if velocidade_alta or aceleracao_alta:
        blocos.append("CAUSAS OFICIAIS WEG — ruído elevado e vibração:\n" + "\n".join(
            f"  - {c}: {a}" for c, a in PROBLEMAS_WEG["ruído elevado / vibração"]))

    blocos.append(LIMITACOES)
    return "\n\n".join(blocos)


@tool
def get_motor_baselines() -> str:
    """Retorna os valores de referência MEDIDOS na telemetria real dos motores
    Forzy, por regime operacional, e os limites normativos oficiais da WEG.

    Use quando o usuário perguntar se um valor está normal, qual o valor esperado,
    ou quando precisar contextualizar uma leitura. Estes baselines vêm de 3h57 de
    telemetria real, não de norma — e isso importa, porque o baseline saudável
    medido (6,50 mm/s) é MAIOR que o limite ISO de máquina pequena.
    """
    op = BASELINE["OPERACAO"]
    return f"""BASELINES MEDIDOS — 3h57min de telemetria real dos motores S1 e S2 (19/05/2026)

REGIME DE OPERAÇÃO (n={op['n']} leituras, ~19 min de regime estacionário por motor):
  Velocidade de vibração : {op['v_rms_mm_s'][0]:.3f} ± {op['v_rms_mm_s'][1]:.3f} mm/s
  Aceleração             : {op['a_rms_g'][0]:.3f} ± {op['a_rms_g'][1]:.3f} g
  Fator de crest         : {op['crest'][0]:.3f} ± {op['crest'][1]:.3f}  (faixa 3-4 = máquina sã)
  Temperatura            : {op['temp_c'][0]:.2f} ± {op['temp_c'][1]:.2f} °C

REGIME TRANSIENTE (n={BASELINE['TRANSIENTE']['n']}): velocidade {BASELINE['TRANSIENTE']['v_rms_mm_s'][0]:.3f} mm/s, temperatura {BASELINE['TRANSIENTE']['temp_c'][0]:.2f} °C
MOTOR PARADO (n={BASELINE['PARADO']['n']}): velocidade {BASELINE['PARADO']['v_rms_mm_s'][0]:.3f} mm/s, temperatura {BASELINE['PARADO']['temp_c'][0]:.2f} °C

DIFERENÇA ENTRE OS MOTORES (medida, estatisticamente significativa):
  O S2 opera 4,39 °C mais quente e vibra 0,182 mm/s a mais que o S1 em regime.
  São assinaturas distintas apesar de estarem na mesma bancada.

LIMITES OFICIAIS WEG — Tabela 7.7 do manual (ISO 20816-3, saída ≤ 300 kW, Grupo 2):
  Base rígida  : NORMAL ≤ 2,8 | ALARME > 3,5 | CRÍTICO > 5,6  mm/s RMS
  Base flexível: NORMAL ≤ 4,5 | ALARME > 5,6 | CRÍTICO > 8,9  mm/s RMS

ATENÇÃO AO INTERPRETAR: o baseline saudável medido (6,50 mm/s) já está acima do
crítico de base rígida. Ou a montagem amplifica (base flexível), ou o sensor não
reporta velocidade limitada à banda ISO. Por isso o sistema deve avaliar DESVIO
RELATIVO À BASELINE DO PRÓPRIO MOTOR, e não zonas ISO absolutas. Dizer que este
motor está em "zona D crítica" porque marca 6,5 mm/s seria um falso positivo.

ROLAMENTO (Tabela 8.1 do manual): carcaça 100, 2 polos → rolamento 6206,
5 g de graxa, relubrificação a cada 25.000 horas em 60 Hz.

{FENOMENOS_NORMAIS}"""
