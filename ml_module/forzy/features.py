"""
As 16 features do detector de novidade.

Todas as janelas são TEMPORAIS, nunca por número de amostras. Com amostragem
irregular (mediana 0,60 s, p99 31 s, lacunas de até 228 s), uma janela de N
amostras teria duração física variável e misturaria escalas de tempo diferentes
no mesmo vetor de features.

A janela curta de 60 s captura tendência mecânica; a longa de 300 s captura
tendência térmica, porque a temperatura tem resolução de 1 °C e constante de
tempo de minutos (tau medido: 25,2 min; atraso de transporte: ~7 min).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

JANELA_CURTA_S = 60.0
JANELA_LONGA_S = 300.0
JANELA_DELTA_S = 5.0
GATE_CREST = 0.05        # a_rms abaixo disso: motor parado, crest seria ruído/ruído
MIN_JANELA_SLOPE_S = 60.0  # guarda do temp_slope

FEATURES = [
    "v_rms", "a_rms", "a_peak", "temp_c",           # canais brutos
    "crest", "razao_av",                            # razões físicas
    "v_roll_mean", "v_roll_std", "v_roll_max",      # janela 60 s
    "a_roll_mean", "a_roll_std", "crest_roll_mean",
    "temp_roll_mean", "temp_slope",                 # janela 300 s
    "v_delta",                                      # variação 5 s
    "frac_repetida",                                # instrumentação
]

# Grupo físico de cada feature — usado na atribuição por componente.
GRUPO_FEATURE = {
    "v_rms": "velocidade", "v_roll_mean": "velocidade", "v_roll_std": "velocidade",
    "v_roll_max": "velocidade", "v_delta": "velocidade",
    "a_rms": "aceleracao", "a_peak": "aceleracao", "a_roll_mean": "aceleracao",
    "a_roll_std": "aceleracao", "crest": "aceleracao", "crest_roll_mean": "aceleracao",
    "razao_av": "aceleracao",
    "temp_c": "temperatura", "temp_roll_mean": "temperatura", "temp_slope": "temperatura",
    "frac_repetida": "instrumentacao",
}


def _segundos(ts: np.ndarray) -> np.ndarray:
    return (ts - ts[0]) / np.timedelta64(1, "s")


def _janela_inicios(seg: np.ndarray, janela_s: float) -> np.ndarray:
    """Para cada i, o menor índice j tal que seg[i] - seg[j] <= janela_s.
    Dois ponteiros: O(n). Posicional, tolera timestamps duplicados."""
    n = len(seg)
    ini = np.empty(n, dtype=np.int64)
    j = 0
    for i in range(n):
        while seg[i] - seg[j] > janela_s:
            j += 1
        ini[i] = j
    return ini


def _roll(seg: np.ndarray, val: np.ndarray, janela_s: float, fn: str) -> np.ndarray:
    """Estatística móvel por janela temporal trailing."""
    ini = _janela_inicios(seg, janela_s)
    n = len(val)
    out = np.empty(n)
    for i in range(n):
        w = val[ini[i] : i + 1]
        w = w[~np.isnan(w)]
        if w.size == 0:
            out[i] = np.nan
        elif fn == "mean":
            out[i] = w.mean()
        elif fn == "std":
            out[i] = w.std(ddof=0)
        elif fn == "max":
            out[i] = w.max()
    return out


def _frac_repetida(seg: np.ndarray, val: np.ndarray, janela_s: float) -> np.ndarray:
    """Fração de leituras idênticas à anterior dentro da janela.

    É a ÚNICA feature capaz de detectar sensor travado, porque nesse modo os
    valores continuam dentro da faixa nominal e passam por todo o resto.
    Em regime saudável o v-RMS repete 57,7% (medido); travado repete 100%.
    Só é discriminante no canal v_rms: em a_rms/a_peak o valor saudável já é
    84–95%, sem margem.
    """
    ini = _janela_inicios(seg, janela_s)
    igual = np.zeros(len(val), dtype=np.float64)
    igual[1:] = (val[1:] == val[:-1]).astype(np.float64)
    csum = np.concatenate([[0.0], np.cumsum(igual)])
    idx = np.arange(len(val))
    n_janela = idx - ini + 1
    return (csum[idx + 1] - csum[ini]) / np.maximum(n_janela, 1)


def _temp_slope(seg: np.ndarray, temp: np.ndarray, janela_s: float) -> np.ndarray:
    """Taxa térmica em °C/min sobre a janela longa.

    GUARDA OBRIGATÓRIA: zera a taxa quando a janela efetiva tem menos de 1 min.
    Sem ela, a quantização de 1 °C dividida por um intervalo curto produz valores
    absurdos — foi o que fez o modelo térmico de 1ª ordem sair com R² de 0,0005.
    """
    ini = _janela_inicios(seg, janela_s)
    out = np.zeros(len(temp))
    for i in range(len(temp)):
        j = ini[i]
        dt = seg[i] - seg[j]
        if dt < MIN_JANELA_SLOPE_S:
            continue
        if np.isnan(temp[i]) or np.isnan(temp[j]):
            continue
        out[i] = (temp[i] - temp[j]) / (dt / 60.0)
    return out


def construir(df_motor: pd.DataFrame) -> pd.DataFrame:
    """Constrói as 16 features para UM motor, já segmentado por regime.

    Recebe o dataframe com os canais brutos e devolve o mesmo dataframe com as
    16 colunas acrescentadas. Sempre recalcular do zero depois de perturbar os
    canais (bancada de injeção) — alterar a feature diretamente cria incoerência
    entre feature e sinal que qualquer detector acha trivialmente.
    """
    out = df_motor.reset_index(drop=True).copy()
    seg = _segundos(out["ts"].to_numpy())

    v = out["v_rms"].to_numpy(dtype=float)
    a = out["a_rms"].to_numpy(dtype=float)
    p = out["a_peak"].to_numpy(dtype=float)
    t = out["temp_c"].to_numpy(dtype=float)

    # --- razões físicas ---
    # crest = a_peak / a_rms, fixado em 0 com o motor parado
    with np.errstate(divide="ignore", invalid="ignore"):
        crest = np.where(a > GATE_CREST, p / np.maximum(a, 1e-9), 0.0)
        # razao_av = proxy de conteúdo de alta frequência
        razao_av = np.where(v > 1e-6, a / np.maximum(v, 1e-9), 0.0)
    crest = np.nan_to_num(crest, nan=0.0, posinf=0.0, neginf=0.0)
    razao_av = np.nan_to_num(razao_av, nan=0.0, posinf=0.0, neginf=0.0)

    out["crest"] = crest
    out["razao_av"] = razao_av

    # --- janela curta 60 s: tendência mecânica ---
    out["v_roll_mean"] = _roll(seg, v, JANELA_CURTA_S, "mean")
    out["v_roll_std"] = _roll(seg, v, JANELA_CURTA_S, "std")
    out["v_roll_max"] = _roll(seg, v, JANELA_CURTA_S, "max")
    out["a_roll_mean"] = _roll(seg, a, JANELA_CURTA_S, "mean")
    out["a_roll_std"] = _roll(seg, a, JANELA_CURTA_S, "std")
    out["crest_roll_mean"] = _roll(seg, crest, JANELA_CURTA_S, "mean")

    # --- janela longa 300 s: tendência térmica ---
    out["temp_roll_mean"] = _roll(seg, t, JANELA_LONGA_S, "mean")
    out["temp_slope"] = _temp_slope(seg, t, JANELA_LONGA_S)

    # --- variação de 5 s ---
    ini5 = _janela_inicios(seg, JANELA_DELTA_S)
    out["v_delta"] = v - v[ini5]

    # --- instrumentação ---
    out["frac_repetida"] = _frac_repetida(seg, v, JANELA_CURTA_S)

    # a_peak pode ter NaN residual dos blackouts de partida deixados de propósito.
    # Preenche só para o vetor de features ficar denso; a informação de ausência
    # já foi consumida no ETL.
    for c in FEATURES:
        if out[c].isna().any():
            out[c] = out[c].ffill().bfill().fillna(0.0)

    return out


def construir_todos(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica por motor. Nunca atravessar a fronteira entre motores: as janelas
    móveis passariam pelo vazio entre as séries."""
    return pd.concat(
        [construir(g) for _, g in df.groupby("motor", sort=True)],
        ignore_index=True,
    )
