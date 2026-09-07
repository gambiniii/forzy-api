"""
Atribuição — do desvio nos sinais para os componentes candidatos do motor.

Duas camadas:

1. Z-SCORE ROBUSTO POR FEATURE contra o baseline do regime da própria amostra.
   Serve para verificar se o modelo está respondendo à física ou a alguma pista
   espúria. Esta camada é DETERMINÍSTICA e idêntica entre máquinas, porque
   depende só do ETL e das medianas, não da rede neural. É o melhor teste de
   sanidade disponível.

2. MAPA FALHA → COMPONENTES, seguindo a regra que o próprio projeto estabeleceu:
   não tratar como relação 1:1. Um mesmo componente altera mais de uma variável,
   e uma variável é afetada por vários componentes. Logo:
       anomalia nos sensores → conjunto de possíveis falhas → componentes candidatos

O teto honesto são os GRUPOS FUNCIONAIS, não uma peça entre 23. Há um sensor por
motor, sem espectro de frequência e sem nenhum rótulo de falha no histórico.
"""

from __future__ import annotations

import numpy as np

from ml_module.forzy.features import FEATURES, GRUPO_FEATURE

# Segmentos do modelo 3D (public/3d/Engine1.obj), identificados por geometria.
SEG_CARCACA = "empty_2"
SEG_TAMPA_DIANT = "empty_7"
SEG_TAMPA_TRAS = "empty_13"
SEG_DEFLETORA = "empty_19"
SEG_EIXO = "empty_23"
SEG_CAIXA_LIG = "empty_4"

# ─────────────────────────────────────────────────────────────────────────────
# COMO A ATRIBUIÇÃO DECIDE
#
# Cada assinatura declara a DIREÇÃO esperada de cada feature, com sinal, e a
# pontuação é o casamento entre o padrão observado e esse gabarito:
#
#     pontuação = Σ(z[f] · peso[f]) / Σ|peso[f]|  −  média(|z[f]|) das estáveis
#
# Ranquear por magnitude bruta não funciona, e isso foi MEDIDO. Os vetores de
# z-score dos modos injetados, no regime de operação do MOTOR-01:
#
#   modo                  v_rms   v_roll_max   a_rms   crest   razao_av   v_roll_std
#   F1 desbalanceamento   +22.2      +23.4      +2.5    -0.5     -15.0       +2.1
#   F3 rolamento           +2.2       +8.4      +3.2    +4.4      +3.6       +0.8
#   F5 deriva de ganho     +6.1       +8.0      +4.2    +0.3      +0.4       +0.2
#   F6 perda acionamento  -18.3      -19.3     -11.9    +0.2      -0.3        0.0
#
# A leitura física é direta. `razao_av` (a_rms/v_rms) é o discriminador:
#   • desbalanceamento → DESPENCA, porque a energia vai para 1x a rotação e a
#     velocidade sobe muito mais que a aceleração
#   • rolamento        → SOBE, porque o impacto de alta frequência é aceleração
#   • deriva de ganho  → FICA EM ZERO, porque todos os canais escalam juntos e a
#     razão entre eles se preserva
#
# E `v_roll_std` com `v_delta` separam rampa de degrau: a deriva de calibração é
# um degrau (variabilidade inalterada), o desbalanceamento é uma rampa.
#
# Sem os pesos com sinal, F5 era atribuído a desbalanceamento: ele eleva a
# velocidade em 40% e, contra uma baseline de desvio 0,25 mm/s, isso dá z ≈ 6 e
# ganha de qualquer assinatura que só olhe magnitude.
# ─────────────────────────────────────────────────────────────────────────────
ASSINATURAS = {
    "desbalanceamento": {
        "titulo": "Desbalanceamento do rotor",
        # razao_av pesa DOBRO: a queda dela é o que separa desbalanceamento de
        # deriva de calibração, que também eleva a velocidade.
        "direcao": {"v_rms": 1.0, "v_roll_max": 1.0, "razao_av": -2.0, "crest": -0.5},
        "estaveis": [],
        "segmentos": [SEG_EIXO, SEG_TAMPA_DIANT, SEG_CARCACA],
        "componentes": ["Rotor", "Eixo", "Ventilador", "Acoplamento", "Fixação/pés"],
        "explicacao": (
            "Velocidade de vibração sobe de forma sustentada com a aceleração subindo menos. "
            "Energia concentrada em 1x a rotação, não em impacto de alta frequência."
        ),
        "precocidade": "tardio",
    },
    "rolamento": {
        "titulo": "Degradação de rolamento",
        # O crest sobe ANTES do valor eficaz, e razao_av sobe junto porque o
        # impacto de alta frequência é aceleração. O peso negativo em v_rms
        # afasta o desbalanceamento, em que a velocidade é que domina.
        "direcao": {"crest": 1.0, "crest_roll_mean": 1.0, "a_peak": 1.0,
                    "razao_av": 1.0, "v_rms": -0.3},
        "estaveis": [],
        "segmentos": [SEG_TAMPA_DIANT, SEG_TAMPA_TRAS],
        "componentes": [
            "Rolamento 6206 (dianteiro)", "Rolamento 6206 (traseiro)",
            "Lubrificante", "Vedação V-Ring", "Eixo",
        ],
        "explicacao": (
            "O fator de crest sobe ANTES do valor eficaz: o defeito de pista gera impactos "
            "curtos de alta energia que quase não alteram o RMS. É o indicador precoce clássico."
        ),
        "precocidade": "precoce",
    },
    "termico": {
        "titulo": "Problema térmico ou elétrico",
        "direcao": {"temp_slope": 1.0, "temp_roll_mean": 1.0, "temp_c": 1.0},
        # Falha mecânica severa mexe na vibração; se ela está quieta em qualquer
        # direção, o problema é térmico ou elétrico.
        "estaveis": ["v_roll_mean", "a_roll_mean"],
        "segmentos": [SEG_CARCACA, SEG_DEFLETORA, SEG_CAIXA_LIG],
        "componentes": [
            "Ventilador", "Tampa defletora / fluxo de ar", "Estator bobinado",
            "Carcaça (dissipação)", "Capacitor", "Placa de bornes",
        ],
        "explicacao": (
            "Temperatura sobe com a vibração quase inalterada, o que afasta falha mecânica "
            "severa. Ventilação obstruída é a causa número um segundo o manual WEG."
        ),
        "precocidade": "tardio",
    },
    "instrumentacao": {
        "titulo": "Falha de instrumentação",
        "direcao": {"frac_repetida": 1.0},
        # Sensor travado congela em valor NOMINAL: o nível não muda. Se ele
        # desabou, é perda de acionamento, não instrumentação.
        "estaveis": ["v_roll_mean", "a_roll_mean"],
        "segmentos": [],          # NÃO acende peça nenhuma
        "componentes": ["Sensor IO-Link", "Cabo do sensor", "Mestre IO-Link"],
        "explicacao": (
            "Os canais congelaram em valores dentro da faixa nominal. Não é falha do motor. "
            "Trocar rolamento por causa de um sensor travado seria o pior erro possível."
        ),
        "precocidade": "precoce",
    },
    "deriva_calibracao": {
        "titulo": "Deriva de calibração do sensor",
        # Erro de ganho move TODOS os canais na mesma proporção, então as razões
        # entre eles se preservam. É isso que separa deriva de desbalanceamento:
        # ambos elevam a velocidade, mas só o desbalanceamento derruba razao_av.
        "direcao": {"v_rms": 1.0, "a_rms": 1.0, "a_roll_mean": 1.0},
        "estaveis": ["crest", "razao_av", "v_roll_std", "v_delta"],
        "segmentos": [],          # não é falha do motor: NÃO acende peça
        "componentes": ["Sensor IO-Link", "Calibração do sensor", "Mestre IO-Link"],
        "explicacao": (
            "Todos os canais subiram na mesma proporção e as razões entre eles não mudaram. "
            "Isso é assinatura de ganho errado no sensor, não de degradação mecânica: uma "
            "falha real muda a relação entre velocidade e aceleração."
        ),
        "precocidade": "precoce",
    },
    "perda_acionamento": {
        "titulo": "Perda de acionamento",
        # Desvio NEGATIVO: o sinal desabou porque o acionamento sumiu.
        "direcao": {"v_roll_mean": -1.0, "a_roll_mean": -1.0, "v_rms": -1.0},
        "estaveis": [],
        "segmentos": [SEG_CAIXA_LIG, SEG_TAMPA_TRAS],
        "componentes": [
            "Capacitor", "Mecanismo centrífugo de partida", "Platinado",
            "Placa de bornes", "Alimentação",
        ],
        "explicacao": (
            "O acionamento caiu sem comando. Em motor monofásico o suspeito principal é o "
            "conjunto de partida (capacitor, centrífugo, platinado)."
        ),
        "precocidade": "tardio",
    },
}

# Limite abaixo do qual NÃO se atribui componente. Sem isso, num histórico 100%
# saudável o ranking sempre elegeria alguma peça — e ela estaria errada.
Z_MINIMO = 3.0


def z_por_feature(x: np.ndarray, mediana: np.ndarray, escala: np.ndarray) -> np.ndarray:
    """Z-score robusto de cada feature contra o baseline do regime da amostra."""
    return (np.asarray(x, dtype=float) - mediana) / np.maximum(escala, 1e-9)


def _pontuar_assinaturas(z: dict[str, float]) -> list[tuple[str, float]]:
    """Pontua cada assinatura por CASAMENTO DE PADRÃO COM SINAL.

        pontuação = Σ(z[f] · peso[f]) / Σ|peso[f]|  −  média(|z[f]|) das estáveis

    O primeiro termo mede o quanto o desvio observado aponta na direção que
    aquela falha produziria. O segundo penaliza movimento nas features que a
    falha deveria deixar quietas — é o que separa sensor travado (nível
    inalterado) de perda de acionamento (nível desabou).

    Queda de vibração nunca indica degradação por si só: durante a corrida ela
    cai porque o lubrificante aquece (medido: −0,038 mm/s por minuto enquanto a
    carcaça vai de 32 a 37 °C). Por isso apenas `perda_acionamento` tem pesos
    negativos, e ela exige que a queda seja de nível, não de tendência.
    """
    pontos = []
    for chave, meta in ASSINATURAS.items():
        direcao = meta.get("direcao") or {}
        peso_total = sum(abs(p) for p in direcao.values())
        if peso_total <= 0:
            continue
        casamento = sum(z.get(f, 0.0) * p for f, p in direcao.items()) / peso_total

        estaveis = [abs(z[f]) for f in meta.get("estaveis", []) if f in z]
        penalidade = float(np.mean(estaveis)) if estaveis else 0.0

        pontos.append((chave, float(casamento - penalidade)))
    pontos.sort(key=lambda p: -p[1])
    return pontos


def atribuir(
    x: np.ndarray,
    mediana: np.ndarray,
    escala: np.ndarray,
    score_norm: float,
    regime: str,
    nomes: list[str] | None = None,
) -> dict:
    """Atribui a anomalia a um conjunto de componentes candidatos.

    Devolve sempre uma estrutura completa, inclusive quando NÃO há atribuição —
    esse estado é obrigatório e protege a credibilidade do sistema.
    """
    nomes = nomes or FEATURES
    z = dict(zip(nomes, z_por_feature(x, mediana, escala)))

    top_features = sorted(z.items(), key=lambda kv: -abs(kv[1]))[:5]
    pontos = _pontuar_assinaturas(z)

    base = {
        "regime": regime,
        "score_normalizado": round(float(score_norm), 4),
        "top_features": [{"feature": f, "z": round(float(v), 3), "grupo": GRUPO_FEATURE.get(f, "?")}
                         for f, v in top_features],
    }

    # O motor parado não é falha. Com 42,5% das leituras nesse estado, atribuir
    # componente aqui geraria alarme toda vez que o motor fosse desligado.
    if regime == "PARADO":
        return {**base, "atribuido": False,
                "motivo": "Motor parado — análise de componente não aplicável."}

    if not pontos or pontos[0][1] < Z_MINIMO:
        return {**base, "atribuido": False,
                "motivo": ("Desvio distribuído entre as features, sem eixo dominante. "
                           "Anomalia sem atribuição de componente.")}

    # Instrumentação é avaliada PRIMEIRO e curto-circuita todo o resto: se o
    # sensor travou, qualquer outra conclusão é sobre um sinal que não existe.
    chave, pontuacao = pontos[0]
    for c, p in pontos:
        if c == "instrumentacao" and p >= Z_MINIMO:
            chave, pontuacao = c, p
            break

    meta = ASSINATURAS[chave]
    alternativas = [
        {"falha": ASSINATURAS[c]["titulo"], "pontuacao": round(p, 3)}
        for c, p in pontos[1:3] if p >= Z_MINIMO * 0.6
    ]

    return {
        **base,
        "atribuido": True,
        "falha": chave,
        "titulo": meta["titulo"],
        "pontuacao": round(pontuacao, 3),
        "segmentos": meta["segmentos"],
        "componentes_candidatos": meta["componentes"],
        "explicacao": meta["explicacao"],
        "precocidade": meta["precocidade"],
        "alternativas": alternativas,
    }
