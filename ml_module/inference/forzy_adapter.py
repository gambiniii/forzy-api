"""
Adapter que converte leituras do banco Forzy para o DataFrame
que build_features espera (port1 + port2 com nomes de coluna do CSV original).

O sensor envia apenas uma porta por vez. Enquanto temos um único
componente_id para os dois sensores físicos, espelhamos port2 = port1
para que build_features produza as 61 features sem KeyError.
Quando port2 tiver componente_id próprio no futuro, basta passar
leituras separadas nos argumentos port1_rows / port2_rows.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd


# Nomes exatos das colunas que build_features resolve via _resolve_sensor_columns
COL_TIMESTAMP = "timestamp"
COL_P1_VEL   = "1.1. Velocidade"
COL_P1_ACC   = "1.2. Aceleração"
COL_P1_TEMP  = "1.3. Temperatura"
COL_P2_VEL   = "2.1. Velocidade"
COL_P2_ACC   = "2.2. Aceleração"
COL_P2_TEMP  = "2.3. Temperatura"


def leituras_to_raw_df(
    port1_rows: list[dict[str, Any]],
    port2_rows: list[dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """
    Converte listas de dicts (vindas do banco) em raw DataFrame
    compatível com load_raw_csv / build_features.

    Cada dict deve ter: timestamp, rpm (velocidade), vibracao (aceleração), temperatura.
    port2_rows é opcional — se None, espelha port1.
    """
    if not port1_rows:
        return pd.DataFrame()

    if port2_rows is None:
        port2_rows = port1_rows

    n = min(len(port1_rows), len(port2_rows))
    p1 = port1_rows[:n]
    p2 = port2_rows[:n]

    records = []
    for r1, r2 in zip(p1, p2):
        ts = r1.get("timestamp") or datetime.utcnow()
        if isinstance(ts, str):
            ts = pd.to_datetime(ts)
        records.append({
            COL_TIMESTAMP: ts,
            COL_P1_VEL:   r1.get("rpm", 0.0) or 0.0,
            COL_P1_ACC:   r1.get("vibracao", 0.0) or 0.0,
            COL_P1_TEMP:  r1.get("temperatura", 0.0) or 0.0,
            COL_P2_VEL:   r2.get("rpm", 0.0) or 0.0,
            COL_P2_ACC:   r2.get("vibracao", 0.0) or 0.0,
            COL_P2_TEMP:  r2.get("temperatura", 0.0) or 0.0,
        })

    return pd.DataFrame(records)
