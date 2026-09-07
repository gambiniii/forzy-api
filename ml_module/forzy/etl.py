"""
ETL dos dados Forzy — decodificação do PDI e tratamento dos artefatos do sensor.

ARMADILHAS TRATADAS AQUI (todas verificadas nos dados reais):
  1. O CSV tem 3 LINHAS de cabeçalho (caminho da tag, nome amigável, tipo) e
     separador ';'. Sem skiprows=3 o parse sai errado.
  2. `pd.to_datetime` sem `format="ISO8601"` infere o formato da primeira linha
     e quebra na segunda quando a fração de segundo é irregular.
  3. O timestamp é `datetime64[us]`, não `[ns]`. Converter com `.astype("int64")`
     assume nanossegundos e produz valores 1000x menores — bug real que fez a
     taxa térmica sair 1000x menor. Aqui sempre usamos `.total_seconds()` sobre
     a diferença.
  4. Existem 7 TIMESTAMPS DUPLICADOS exatos. Isso quebra indexação por rótulo no
     pandas ("Reindexing only valid with uniquely valued Index objects"). Toda a
     lógica é POSICIONAL, com numpy.
  5. O canal a-Peak é LATCHED: o sensor só recalcula o pico a cada ~9,02 s
     (medido) enquanto v-RMS e a-RMS atualizam a cada ~0,6 s. Entre refreshes ele
     SEGURA o valor anterior; os zeros com eixo girando são blackouts de partida,
     concentrados no transiente (61,3% deles), em rajadas de até 52 s.
  6. A quantização é exata: v_rms, a_rms e a_peak em múltiplos de 0,01 e a
     temperatura em graus inteiros. `requantizar()` existe para a bancada de
     injeção — sem ela o detector acusaria as falhas pela perda de quantização em
     vez da física da degradação.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Mapa do Process Data In, obtido por engenharia reversa e validado contra as
# colunas já convertidas do próprio arquivo.
PDI_V_RMS = (0, 1)      # big-endian / 100 -> mm/s
PDI_A_RMS = (4, 5)      # big-endian / 100 -> g
PDI_A_PEAK = (8, 9)     # big-endian / 100 -> g   (NÃO exportado em coluna)
PDI_TEMP = 13           # byte único -> °C

# Gate do tratamento do a-Peak: abaixo disso o eixo está parado e zero é leitura
# correta, não ausência.
GATE_A_RMS = 0.05
# Três ciclos de refresh de ~9 s. Rajadas de blackout chegam a 52 s, então parte
# dos zeros fica como ausente de propósito — propagar valor obsoleto pela rampa
# inteira seria pior.
FFILL_MAX_S = 30.0

PASSO_QUANT = 0.01      # v_rms, a_rms, a_peak
PASSO_TEMP = 1.0        # temperatura


def _decode_pdi(serie: pd.Series) -> pd.DataFrame:
    """Decodifica a coluna de array de bytes do PDI nos 4 canais físicos."""
    n = len(serie)
    v = np.full(n, np.nan)
    a = np.full(n, np.nan)
    p = np.full(n, np.nan)
    t = np.full(n, np.nan)

    for i, bruto in enumerate(serie.to_numpy()):
        if not isinstance(bruto, str):
            continue
        try:
            b = [int(x) for x in bruto.split(",")]
        except ValueError:
            continue
        if len(b) < 16:
            continue
        v[i] = ((b[PDI_V_RMS[0]] << 8) | b[PDI_V_RMS[1]]) / 100.0
        a[i] = ((b[PDI_A_RMS[0]] << 8) | b[PDI_A_RMS[1]]) / 100.0
        p[i] = ((b[PDI_A_PEAK[0]] << 8) | b[PDI_A_PEAK[1]]) / 100.0
        t[i] = float(b[PDI_TEMP])

    return pd.DataFrame({"v_rms": v, "a_rms": a, "a_peak": p, "temp_c": t})


def corrigir_a_peak(
    ts: np.ndarray, a_rms: np.ndarray, a_peak: np.ndarray
) -> tuple[np.ndarray, int, int]:
    """Trata o latch do a-Peak. Retorna (a_peak_corrigido, n_sinalizadas, n_preenchidas).

    Com o eixo girando (a_rms > GATE_A_RMS), a_peak == 0 é ausência de leitura,
    não pico zero: preenchemos para frente por no máximo FFILL_MAX_S. Com o motor
    parado, zero é a leitura correta e fica como está.

    Trabalha posicionalmente com numpy justamente porque há timestamps duplicados.
    """
    out = a_peak.copy()
    girando = a_rms > GATE_A_RMS
    ausente = girando & (a_peak == 0.0)
    n_sinalizadas = int(ausente.sum())

    seg = (ts - ts[0]) / np.timedelta64(1, "s") if ts.dtype.kind == "M" else ts.astype(float)

    n_preenchidas = 0
    ultimo_val = np.nan
    ultimo_t = -np.inf
    for i in range(len(out)):
        if ausente[i]:
            if not np.isnan(ultimo_val) and (seg[i] - ultimo_t) <= FFILL_MAX_S:
                out[i] = ultimo_val
                n_preenchidas += 1
            else:
                out[i] = np.nan          # deixa ausente de propósito
        elif out[i] > 0.0:
            ultimo_val = out[i]
            ultimo_t = seg[i]

    return out, n_sinalizadas, n_preenchidas


def requantizar(df: pd.DataFrame) -> pd.DataFrame:
    """Devolve os canais à grade do sensor real.

    OBRIGATÓRIO depois de qualquer injeção de falha. Os injetores multiplicam por
    fatores contínuos e produzem infinitas casas decimais; o sensor real entrega
    múltiplos exatos de 0,01 (e graus inteiros na temperatura). Sem requantizar,
    o detector acusa a falha pela perda de quantização, não pela física — o que
    inflava a PR-AUC macro de 0,838 para 0,891 na medição original.
    """
    out = df.copy()
    for col in ("v_rms", "a_rms", "a_peak"):
        if col in out.columns:
            out[col] = np.round(out[col] / PASSO_QUANT) * PASSO_QUANT
    if "temp_c" in out.columns:
        out["temp_c"] = np.round(out["temp_c"] / PASSO_TEMP) * PASSO_TEMP
    return out


def carregar_csv(caminho: str | Path) -> pd.DataFrame:
    """Carrega o histórico do IO-Link Master em formato LONGO (uma linha por
    motor por instante), com os 4 canais decodificados do PDI.

    Usamos os canais do PDI e ignoramos as colunas já convertidas: o payload
    adianta as colunas em uma amostra, e usar só o payload garante consistência
    interna. As colunas convertidas são mantidas apenas como `*_conv` para
    auditoria da decodificação.
    """
    df = pd.read_csv(
        caminho,
        sep=";",
        skiprows=3,
        header=None,
        names=["ts", "pdi1", "pdi2", "v1", "a1", "t1", "v2", "a2", "t2"],
        dtype=str,
    )

    ts = pd.to_datetime(df["ts"], format="ISO8601").astype("datetime64[us]")

    partes = []
    for porta, motor in ((1, "MOTOR-01"), (2, "MOTOR-02")):
        canais = _decode_pdi(df[f"pdi{porta}"])
        canais["ts"] = ts.to_numpy()
        canais["motor"] = motor

        # Corrige o latch do a-Peak por motor, posicionalmente.
        corrigido, n_sinal, n_fill = corrigir_a_peak(
            canais["ts"].to_numpy(),
            canais["a_rms"].to_numpy(),
            canais["a_peak"].to_numpy(),
        )
        canais["a_peak"] = corrigido
        canais.attrs[f"a_peak_{motor}"] = {"sinalizadas": n_sinal, "preenchidas": n_fill}

        # Convertidas, só para auditoria.
        canais["v_conv"] = pd.to_numeric(df[f"v{porta}"], errors="coerce")
        canais["a_conv"] = pd.to_numeric(df[f"a{porta}"], errors="coerce")
        canais["t_conv"] = pd.to_numeric(df[f"t{porta}"], errors="coerce")

        partes.append(canais)

    longo = pd.concat(partes, ignore_index=True)
    # Ordenação estável por (motor, ts). `kind="stable"` preserva a ordem
    # original entre timestamps duplicados, o que mantém a série reproduzível.
    longo = longo.sort_values(["motor", "ts"], kind="stable").reset_index(drop=True)
    return longo


def carregar_xlsx(caminho: str | Path) -> pd.DataFrame:
    """Carrega as coletas da API REST — conjunto de VERIFICAÇÃO INDEPENDENTE.

    Nunca usar para treino: são 102 coletas válidas, todas com o motor parado,
    variância zero em aceleração e apenas 2 valores distintos de velocidade.

    A chave "Aceleração" pode chegar com encoding corrompido dependendo do
    caminho de leitura, então normalizamos por regex antes do json.loads. No
    arquivo atual ela vem em UTF-8 correto, mas a normalização é defensiva e
    inofensiva.
    """
    import json
    import re

    abas = {"motor_1_s1": "MOTOR-01", "motor_2_s2": "MOTOR-02"}
    linhas = []

    for aba, motor in abas.items():
        try:
            df = pd.read_excel(caminho, sheet_name=aba)
        except Exception:
            continue
        for _, r in df.iterrows():
            bruto = r.get("raw_payload")
            if not isinstance(bruto, str) or not bruto.strip():
                continue
            texto = re.sub(r'"[A-Za-z]cel[^"]*"', '"Aceleracao"', bruto)
            try:
                obj = json.loads(texto)
            except json.JSONDecodeError:
                continue
            dados = next((v for v in obj.values() if isinstance(v, dict)), None)
            if not dados:
                continue
            linhas.append({
                "ts": pd.to_datetime(r.get("captured_at")),
                "motor": motor,
                "v_rms": float(dados.get("Velocidade", 0)),
                "a_rms": float(dados.get("Aceleracao", 0)),
                "a_peak": np.nan,          # este arquivo não tem o canal de pico
                "temp_c": float(dados.get("Temperatura", 0)),
            })

    if not linhas:
        return pd.DataFrame(columns=["ts", "motor", "v_rms", "a_rms", "a_peak", "temp_c"])
    return pd.DataFrame(linhas).sort_values(["motor", "ts"], kind="stable").reset_index(drop=True)
