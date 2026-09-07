"""
Segmentação dos três regimes operacionais — a estrutura mais importante destes dados.

PARADO e OPERACAO estão separados por duas ordens de grandeza em v-RMS e não se
sobrepõem: o histograma de v-RMS em escala log é claramente bimodal.

Definir regime só por limiar de valor NÃO funciona. OPERACAO exige três condições
simultâneas, e sem a condição (b) os acionamentos curtos entrariam como regime e
alargariam indevidamente o envelope do normal:
  (a) v-RMS suavizado (mediana móvel de 5 s) acima de 2,00 mm/s
  (b) o trecho contíguo acima do limiar durando pelo menos 60 s
  (c) a amostra não estando na subida inicial nem na descida final

CORREÇÃO MEDIDA NA MARGEM FINAL: a especificação original pedia 15 s de margem no
fim, mas a rampa de PARADA medida dura 121–127 s. Com 15 s, os 45 s finais ainda
rotulados OPERACAO têm v-RMS médio de 4,84 mm/s contra um platô de 6,61 — 27%
abaixo. Isso contamina o treino do detector de novidade com a desaceleração e é a
origem principal do desvio de 0,94 mm/s em OPERACAO. Por isso TAIL_SKIP_S = 90.

Toda a lógica é POSICIONAL (numpy), porque há timestamps duplicados no arquivo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LIMIAR_OPERACAO = 2.00     # mm/s, sobre o v-RMS suavizado
LIMIAR_PARADO = 0.30       # mm/s
JANELA_MEDIANA_S = 5.0     # mediana móvel temporal
DURACAO_MIN_S = 60.0       # trecho contíguo mínimo para valer como regime
HEAD_SKIP_S = 30.0         # subida inicial descartada
TAIL_SKIP_S = 90.0         # descida final descartada (rampa medida: 121-127 s)
MARGEM_PARADO_S = 10.0     # erosão nas fronteiras do PARADO

PARADO, TRANSIENTE, OPERACAO = "PARADO", "TRANSIENTE", "OPERACAO"


def _segundos(ts: np.ndarray) -> np.ndarray:
    """Segundos desde o início. Usa `.total_seconds()` sobre a diferença — nunca
    `.astype('int64')`, que assumiria nanossegundos num array datetime64[us]."""
    return (ts - ts[0]) / np.timedelta64(1, "s")


def mediana_movel_temporal(seg: np.ndarray, val: np.ndarray, janela_s: float) -> np.ndarray:
    """Mediana móvel por JANELA DE TEMPO, não por número de amostras.

    Com amostragem irregular (mediana 0,60 s, p99 31 s, lacunas de até 228 s),
    uma janela de N amostras teria duração física variável e misturaria escalas
    de tempo diferentes no mesmo vetor. Janela trailing [t-janela, t].
    """
    n = len(val)
    out = np.empty(n)
    ini = 0
    for i in range(n):
        while seg[i] - seg[ini] > janela_s:
            ini += 1
        out[i] = np.median(val[ini : i + 1])
    return out


def _blocos_contiguos(mascara: np.ndarray) -> list[tuple[int, int]]:
    """Índices [ini, fim) de cada bloco contíguo True."""
    if not mascara.any():
        return []
    d = np.diff(mascara.astype(np.int8))
    inis = list(np.flatnonzero(d == 1) + 1)
    fins = list(np.flatnonzero(d == -1) + 1)
    if mascara[0]:
        inis.insert(0, 0)
    if mascara[-1]:
        fins.append(len(mascara))
    return list(zip(inis, fins))


def segmentar(df_motor: pd.DataFrame) -> pd.DataFrame:
    """Adiciona as colunas `regime`, `v_smooth` e `bloco` a UM motor.

    `bloco` numera os trechos contíguos de OPERACAO — é o que permite o corte
    treino/teste ser feito DENTRO de cada bloco, e não globalmente.
    """
    out = df_motor.reset_index(drop=True).copy()
    ts = out["ts"].to_numpy()
    seg = _segundos(ts)
    v = out["v_rms"].to_numpy(dtype=float)

    v_smooth = mediana_movel_temporal(seg, np.nan_to_num(v, nan=0.0), JANELA_MEDIANA_S)
    out["v_smooth"] = v_smooth

    regime = np.full(len(out), TRANSIENTE, dtype=object)
    bloco = np.full(len(out), -1, dtype=int)

    # --- OPERACAO: as três condições simultâneas ---
    acima = v_smooth > LIMIAR_OPERACAO
    n_bloco = 0
    for ini, fim in _blocos_contiguos(acima):
        dur = seg[fim - 1] - seg[ini]
        if dur < DURACAO_MIN_S:          # (b) acionamento curto: é rampa, não regime
            continue
        t0, t1 = seg[ini] + HEAD_SKIP_S, seg[fim - 1] - TAIL_SKIP_S
        nucleo = (seg >= t0) & (seg <= t1)
        nucleo[:ini] = False
        nucleo[fim:] = False
        if nucleo.any():
            regime[nucleo] = OPERACAO
            bloco[nucleo] = n_bloco
            n_bloco += 1

    # --- PARADO: com erosão de MARGEM_PARADO_S nas fronteiras ---
    baixo = v_smooth < LIMIAR_PARADO
    for ini, fim in _blocos_contiguos(baixo):
        t0, t1 = seg[ini] + MARGEM_PARADO_S, seg[fim - 1] - MARGEM_PARADO_S
        if t1 <= t0:
            continue
        nucleo = (seg >= t0) & (seg <= t1)
        nucleo[:ini] = False
        nucleo[fim:] = False
        # não sobrescreve OPERACAO (não deveria haver interseção, mas é barato garantir)
        nucleo &= regime != OPERACAO
        regime[nucleo] = PARADO

    # Numera também os trechos contíguos de PARADO e TRANSIENTE. O corte
    # treino/teste tem que ser feito DENTRO de cada bloco em todos os regimes,
    # não só em OPERACAO: os transientes estão espalhados pelas 4 h e o estado
    # térmico muda ao longo da sessão (heat soak). Com bloco único, os últimos
    # 30% do transiente ficavam fora da distribuição de treino e o falso
    # positivo medido nesse regime subia para 62%.
    proximo = int(bloco.max()) + 1 if (bloco >= 0).any() else 0
    for alvo in (PARADO, TRANSIENTE):
        for ini, fim in _blocos_contiguos(regime == alvo):
            bloco[ini:fim] = proximo
            proximo += 1

    out["regime"] = regime
    out["bloco"] = bloco
    return out


def segmentar_todos(longo: pd.DataFrame) -> pd.DataFrame:
    """Aplica a segmentação por motor. NUNCA agrupar séries de origens temporais
    diferentes: as janelas móveis atravessariam o vazio entre elas."""
    return pd.concat(
        [segmentar(g) for _, g in longo.groupby("motor", sort=True)],
        ignore_index=True,
    )


def classificar_atual(df_motor: pd.DataFrame) -> str:
    """Regime da ÚLTIMA amostra, para uso em INFERÊNCIA.

    Não usa o descarte de cabeça e cauda. Aquele descarte é higiene do conjunto
    de TREINO: serve para o modelo não aprender a rampa de partida nem a
    desaceleração como se fossem regime. Em produção ele produziria o resultado
    errado, porque a última amostra de qualquer janela cai sempre dentro dos
    últimos 90 s dela — e o regime atual sairia como TRANSIENTE em 100% dos
    casos, independentemente do que o motor está fazendo.

    Aqui a classificação é física: nível de velocidade suavizado mais a exigência
    de que o trecho contíguo já dure o suficiente para não ser um acionamento
    curto.
    """
    if len(df_motor) == 0:
        return TRANSIENTE

    ts = df_motor["ts"].to_numpy()
    seg = _segundos(ts)
    v = np.nan_to_num(df_motor["v_rms"].to_numpy(dtype=float), nan=0.0)
    v_smooth = mediana_movel_temporal(seg, v, JANELA_MEDIANA_S)

    atual = v_smooth[-1]
    if atual < LIMIAR_PARADO:
        return PARADO
    if atual <= LIMIAR_OPERACAO:
        return TRANSIENTE

    # Há quanto tempo está continuamente acima do limiar de operação?
    i = len(v_smooth) - 1
    while i > 0 and v_smooth[i - 1] > LIMIAR_OPERACAO:
        i -= 1
    if (seg[-1] - seg[i]) >= DURACAO_MIN_S:
        return OPERACAO
    return TRANSIENTE


def resumo(df: pd.DataFrame) -> pd.DataFrame:
    """Contagem e estatística por motor e regime — para conferir contra os
    números de referência do documento de transferência."""
    g = df.groupby(["motor", "regime"])
    return pd.DataFrame({
        "n": g.size(),
        "v_rms_media": g["v_rms"].mean().round(3),
        "v_rms_std": g["v_rms"].std().round(3),
        "a_rms_media": g["a_rms"].mean().round(3),
        "temp_media": g["temp_c"].mean().round(2),
    })
