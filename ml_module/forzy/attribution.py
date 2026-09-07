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

# Assinaturas esperadas por modo, na linguagem das features. Cada entrada lista
# as features que devem dominar o z-score e os componentes candidatos ORDENADOS
# por probabilidade.
ASSINATURAS = {
    "desbalanceamento": {
        "titulo": "Desbalanceamento do rotor",
        "features_chave": ["v_roll_std", "v_rms", "v_roll_max"],
        # O crest CAI no desbalanceamento: a energia vai para 1x a rotação, não
        # para impacto. Se ele subiu, a hipótese é outra.
        "features_contraste": ["crest", "crest_roll_mean"],
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
        "features_chave": ["crest", "crest_roll_mean", "a_peak", "razao_av"],
        # O que distingue rolamento de desbalanceamento NÃO é a magnitude, é a
        # RAZÃO: o crest sobe enquanto o nível de velocidade quase não muda.
        # Sem este contraste, um aumento de 18% na velocidade contra uma baseline
        # apertada (desvio de 0,25 mm/s) domina o z-score e a atribuição erra.
        "features_contraste": ["v_roll_mean", "v_rms"],
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
        "features_chave": ["temp_slope", "temp_roll_mean", "temp_c"],
        # Falha mecânica severa mexe na vibração; se ela está quieta, o problema
        # é térmico ou elétrico.
        "features_contraste": ["v_roll_mean", "a_roll_mean"],
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
        "features_chave": ["frac_repetida"],
        "features_contraste": [],
        "segmentos": [],          # NÃO acende peça nenhuma
        "componentes": ["Sensor IO-Link", "Cabo do sensor", "Mestre IO-Link"],
        "explicacao": (
            "Os canais congelaram em valores dentro da faixa nominal. Não é falha do motor. "
            "Trocar rolamento por causa de um sensor travado seria o pior erro possível."
        ),
        "precocidade": "precoce",
    },
    "perda_acionamento": {
        "titulo": "Perda de acionamento",
        "features_chave": ["v_roll_std", "a_roll_std", "v_delta"],
        "features_contraste": ["frac_repetida"],
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
    """Pontua cada assinatura por CONTRASTE, não por magnitude bruta.

        pontuação = média(z das features-chave) − média(z das features de contraste)

    Sem o termo de contraste, a assinatura de maior magnitude vence sempre e a
    atribuição erra. Caso medido: a injeção de rolamento eleva a velocidade em
    18%, o que contra uma baseline apertada (desvio de 0,25 mm/s em regime) dá
    z ≈ 4,8 — e a hipótese de desbalanceamento passava à frente da de rolamento,
    mesmo com o crest disparando. O que distingue os dois modos não é o tamanho
    do desvio, é a RAZÃO entre canais:

      • rolamento          → crest sobe, nível de velocidade quase não muda
      • desbalanceamento   → velocidade sobe, crest CAI (energia em 1x a rotação)
      • térmico            → temperatura sobe, vibração fica quieta

    Usa apenas desvio POSITIVO nas features de magnitude: queda de vibração nunca
    indica degradação — durante a corrida ela cai porque o lubrificante aquece
    (medido: -0,038 mm/s por minuto enquanto a carcaça vai de 32 a 37 °C).
    """
    pontos = []
    for chave, meta in ASSINATURAS.items():
        chaves = [z[f] for f in meta["features_chave"] if f in z]
        if not chaves:
            continue

        if chave == "perda_acionamento":
            # Aqui o desvio relevante é o colapso do sinal, em qualquer direção.
            positivo = float(np.mean([abs(v) for v in chaves]))
        else:
            positivo = float(np.mean([max(v, 0.0) for v in chaves]))

        contraste = [z[f] for f in meta.get("features_contraste", []) if f in z]
        penalidade = float(np.mean([max(v, 0.0) for v in contraste])) if contraste else 0.0

        pontos.append((chave, positivo - penalidade))
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
