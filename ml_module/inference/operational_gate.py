"""Gate operacional — motor desligado vs operando antes de anomalia/saúde."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml_module.features.feature_engineering import _resolve_sensor_columns

VIBRATION_FLOOR = 0.3


def compute_gate_velocity(raw_df: pd.DataFrame) -> pd.Series:
    """Média instantânea de velocidade (mm/s) das portas 1 e 2 — sinal bruto, sem rolling."""
    work = raw_df.copy()
    sensor_columns = _resolve_sensor_columns(work)
    v1 = pd.to_numeric(work[sensor_columns["port1"]["velocidade"]], errors="coerce")
    v2 = pd.to_numeric(work[sensor_columns["port2"]["velocidade"]], errors="coerce")
    return (v1 + v2) / 2.0


def gate_velocity_snapshot(
    features_df: pd.DataFrame, raw_df: pd.DataFrame | None = None
) -> float:
    """Velocidade média instantânea do último snapshot para o gate operacional."""
    if raw_df is not None:
        vel_series = compute_gate_velocity(raw_df)
        idx = min(len(features_df) - 1, len(vel_series) - 1)
        return float(vel_series.iloc[idx])
    r50 = [c for c in features_df.columns if c.endswith("velocidade_rolling_mean_50")]
    if r50:
        return float(features_df[r50].iloc[-1].mean())
    return 999.0


def gate_velocity_from_sensors(v1: float, v2: float) -> float:
    """Velocidade média a partir de leituras instantâneas das duas portas."""
    return (float(v1) + float(v2)) / 2.0


def motor_operando(velocidade_media):
    """Retorna True se o motor está operando (vibração acima do piso)."""
    return float(velocidade_media) > VIBRATION_FLOOR


def estado_operacional(velocidade_media):
    """Classifica o estado: 'desligado' ou 'operando'."""
    return "operando" if motor_operando(velocidade_media) else "desligado"


def avaliar_com_gate(velocidade_media, if_anomaly_func, ae_anomaly_func, health_func):
    """
    Wrapper: só avalia anomalia/saúde se o motor estiver operando.
    Se desligado, retorna estado neutro (não é anomalia).
    """
    if not motor_operando(velocidade_media):
        return {
            "estado": "desligado",
            "is_anomaly": False,
            "health_index": None,
            "mensagem": "Motor desligado — análise de anomalia não aplicável.",
        }
    return {
        "estado": "operando",
        "is_anomaly": if_anomaly_func() or ae_anomaly_func(),
        "health_index": health_func(),
        "mensagem": "Motor operando — análise ativa.",
    }
