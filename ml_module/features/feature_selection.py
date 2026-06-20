"""
Features excluídas do IF e LSTM (mas mantidas no RUL híbrido).

Motivos de exclusão:
- _CROSS_FEATURES: port2 espelhado de port1 cria correlação perfeita artificial,
  inflando a capacidade discriminativa sem informação real.
- _TEMP_FEATURES: temperatura fica estática (~27°C) no trecho normal de treino
  (motor frio no início da sessão), mas opera em 40-46°C em produção (motor quente).
  O modelo aprenderia "temperatura alta = anomalia", gerando 100% de falsos positivos.
"""

from __future__ import annotations

import pandas as pd

_CROSS_FEATURES: frozenset[str] = frozenset({
    "velocidade_media_global",
    "aceleracao_media_global",
    "temperatura_media_global",
    "delta_temperatura_entre_sensores",
})

_TEMP_FEATURES: frozenset[str] = frozenset({
    "port1_temperatura_rolling_mean_50",
    "port1_temperatura_rolling_std_50",
    "port1_temperatura_rolling_mean_200",
    "port1_temperatura_rolling_std_200",
    "port1_temperatura_rolling_mean_500",
    "port1_temperatura_rolling_std_500",
    "port2_temperatura_rolling_mean_50",
    "port2_temperatura_rolling_std_50",
    "port2_temperatura_rolling_mean_200",
    "port2_temperatura_rolling_std_200",
    "port2_temperatura_rolling_mean_500",
    "port2_temperatura_rolling_std_500",
    "port1_delta_temperatura",
    "port2_delta_temperatura",
})

EXCLUDED_FROM_ANOMALY_MODELS: frozenset[str] = _CROSS_FEATURES | _TEMP_FEATURES


def anomaly_feature_columns(features_df: pd.DataFrame) -> list[str]:
    """Retorna colunas válidas para IF e LSTM (exclui timestamp, cross e temperatura)."""
    return [
        col for col in features_df.columns
        if col != "timestamp" and col not in EXCLUDED_FROM_ANOMALY_MODELS
    ]
