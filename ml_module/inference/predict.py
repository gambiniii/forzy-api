"""
Ponto único de inferência — combina Isolation Forest, LSTM Autoencoder e XGBoost RUL.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib as _joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_module.features.feature_engineering import build_features, load_raw_csv
from ml_module.inference.operational_gate import (
    gate_velocity_from_sensors,
    gate_velocity_snapshot,
    motor_operando,
)
from ml_module.models.baseline.isolation_forest import load_model as load_if_model
from ml_module.models.anomaly.lstm_autoencoder import load_model as load_lstm_model
from ml_module.models.rul.xgboost_rul import (
    RUL_FEATURE_NAMES,
    load_model as load_rul_model,
    predict as predict_rul,
)
from ml_module.models.rul.xgboost_rul_hybrid import FORZY_RUL_FEATURES as HYBRID_BASE_FEATURES

IF_MODEL_DIR = PROJECT_ROOT / "ml_module/models/baseline/saved"
LSTM_MODEL_DIR = PROJECT_ROOT / "ml_module/models/anomaly/saved"
RUL_MODEL_DIR = PROJECT_ROOT / "ml_module/models/rul/saved"
IF_CWRU_DIR = PROJECT_ROOT / "ml_module/models/baseline/saved_cwru"
LSTM_CWRU_DIR = PROJECT_ROOT / "ml_module/models/anomaly/saved_cwru"
HYBRID_RUL_DIR = PROJECT_ROOT / "ml_module/models/rul/saved_hybrid"

WINDOW_SIZE = 60
FORZY_FEATURE_COUNT = 61
HYBRID_RUL_MIN = 80.0
HYBRID_RUL_MAX = 160.0

LSTM_SEVERITY_SCORES = {
    "normal": 1.0,
    "low": 0.5,
    "medium": 0.3,
    "high": 0.1,
    "critical": 0.0,
}


def _feature_columns(features_df: pd.DataFrame) -> list[str]:
    return [col for col in features_df.columns if col != "timestamp"]


def _lstm_reconstruction_error(model, sequence: np.ndarray) -> float:
    """Calcula MSE de reconstrução para uma sequência (1, window, features)."""
    reconstructed = model.predict(sequence, verbose=0)
    return float(np.mean(np.square(sequence - reconstructed)))


def _lstm_severity(error: float, threshold: float) -> str:
    """Mapeia erro de reconstrução para categoria de severidade."""
    if error < threshold * 1.5:
        return "normal"
    if error < threshold * 2.0:
        return "low"
    if error < threshold * 3.0:
        return "medium"
    if error < threshold * 4.0:
        return "high"
    return "critical"


def _overall_status(combined_health: float, risk_level: str) -> str:
    """Determina status geral com base na saúde combinada e risco RUL."""
    if combined_health < 0.4 or risk_level == "critical":
        return "critical"
    if combined_health >= 0.7 and risk_level == "low":
        return "healthy"
    if combined_health >= 0.4 or risk_level in ("medium", "high"):
        return "warning"
    return "warning"


def _recommendation(status: str, maintenance_window_days: float) -> str:
    """Gera mensagem de recomendação operacional."""
    if status == "healthy":
        return (
            f"Motor operando normalmente. Próxima revisão em "
            f"{maintenance_window_days:.0f} dias."
        )
    if status == "warning":
        return (
            "Atenção: padrão de operação irregular detectado. "
            "Agendar inspeção em breve."
        )
    return "ALERTA: anomalia crítica detectada. Intervenção imediata recomendada."


def _scaler_matches_forzy(scaler) -> bool:
    """Verifica se o scaler é compatível com features Forzy (61 colunas)."""
    return getattr(scaler, "n_features_in_", None) == FORZY_FEATURE_COUNT


def _resolve_model_dir(cwru_dir: Path, forzy_dir: Path, model_kind: str) -> tuple[Path, str]:
    """Prefere diretório CWRU se existir e for compatível com inferência Forzy."""
    if cwru_dir.exists():
        if model_kind == "if":
            _, scaler, _ = load_if_model(cwru_dir)
        else:
            _, scaler, _ = load_lstm_model(cwru_dir)

        if _scaler_matches_forzy(scaler):
            return cwru_dir, "CWRU"

    return forzy_dir, "Forzy (fallback)"


def load_all_models() -> dict:
    """Carrega os três modelos salvos e retorna em um único dicionário."""
    if_dir, if_version = _resolve_model_dir(IF_CWRU_DIR, IF_MODEL_DIR, "if")
    if_model, if_scaler, if_threshold = load_if_model(if_dir)
    print(f"[OK] Isolation Forest ({if_version}) carregado de {if_dir}")

    lstm_dir, lstm_version = _resolve_model_dir(LSTM_CWRU_DIR, LSTM_MODEL_DIR, "lstm")
    lstm_model, lstm_scaler, lstm_threshold = load_lstm_model(lstm_dir)
    print(f"[OK] LSTM Autoencoder ({lstm_version}) carregado de {lstm_dir}")

    hybrid_joblib = HYBRID_RUL_DIR / "hybrid_rul.joblib"
    if hybrid_joblib.exists():
        rul_model = _joblib.load(hybrid_joblib)
        rul_scaler = _joblib.load(HYBRID_RUL_DIR / "hybrid_rul_scaler.joblib")
        with open(HYBRID_RUL_DIR / "hybrid_rul_metrics.json", encoding="utf-8") as f:
            rul_metrics = json.load(f)
        rul_metrics["is_hybrid"] = True
        print("[OK] XGBoost RUL Híbrido carregado")
    else:
        rul_model, rul_scaler, rul_metrics = load_rul_model(RUL_MODEL_DIR)
        rul_metrics["is_hybrid"] = False
        print("[OK] XGBoost RUL Forzy carregado (fallback)")

    return {
        "isolation_forest": (if_model, if_scaler, if_threshold),
        "lstm": (lstm_model, lstm_scaler, lstm_threshold),
        "rul": (rul_model, rul_scaler, rul_metrics),
        "if_version": if_version,
        "lstm_version": lstm_version,
    }


def _hybrid_risk_level(rul_hours: float) -> str:
    if rul_hours > 5000:
        return "low"
    if rul_hours > 1000:
        return "medium"
    if rul_hours > 100:
        return "high"
    return "critical"


def predict_single(
    features_df: pd.DataFrame, models: dict, raw_df: pd.DataFrame | None = None
) -> dict:
    """Executa inferência combinada dos 3 modelos no snapshot mais recente."""
    last_timestamp = features_df["timestamp"].iloc[-1]
    vel_media = gate_velocity_snapshot(features_df, raw_df)

    if not motor_operando(vel_media):
        return {
            "timestamp": last_timestamp,
            "estado_operacional": "desligado",
            "isolation_forest": {
                "is_anomaly": False,
                "anomaly_score": None,
                "label": None,
                "model_version": models.get("if_version", "Forzy"),
            },
            "lstm": {
                "is_anomaly": False,
                "severity": "n/a",
                "reconstruction_error": None,
                "model_version": models.get("lstm_version", "Forzy"),
            },
            "rul": {"available": False, "reason": "motor desligado"},
            "health_index": None,
            "combined": {
                "health_score": None,
                "overall_status": "motor_desligado",
                "recommendation": (
                    "Motor fora de operação. Análise de anomalia não aplicável."
                ),
            },
        }

    if_model, if_scaler, if_threshold = models["isolation_forest"]
    lstm_model, lstm_scaler, lstm_threshold = models["lstm"]
    rul_model, rul_scaler, rul_metrics = models["rul"]

    feature_cols = _feature_columns(features_df)

    # --- Isolation Forest ---
    last_if_features = features_df[feature_cols].iloc[-1:].values
    last_if_scaled = if_scaler.transform(last_if_features)
    if_score_raw = float(if_model.decision_function(last_if_scaled)[0])
    if_label = int(-1 if if_score_raw <= if_threshold else 1)
    if_is_anomaly = if_label == -1
    if_normalized = 0.0 if if_is_anomaly else 1.0

    # --- LSTM Autoencoder ---
    window_data = features_df[feature_cols].values
    if len(window_data) < WINDOW_SIZE:
        padding = np.zeros((WINDOW_SIZE - len(window_data), window_data.shape[1]))
        window_data = np.vstack([padding, window_data])
    else:
        window_data = window_data[-WINDOW_SIZE:]

    window_scaled = lstm_scaler.transform(window_data)
    lstm_sequence = window_scaled.reshape(1, WINDOW_SIZE, window_scaled.shape[1])
    lstm_error = _lstm_reconstruction_error(lstm_model, lstm_sequence)
    lstm_severity = _lstm_severity(lstm_error, lstm_threshold)
    lstm_is_anomaly = lstm_error > lstm_threshold
    lstm_normalized = LSTM_SEVERITY_SCORES[lstm_severity]

    combined_health = 0.4 * if_normalized + 0.6 * lstm_normalized

    # --- XGBoost RUL (resiliente) ---
    rul_result = None
    rul_available = True
    rul_unavailable_reason = None

    try:
        use_hybrid = (HYBRID_RUL_DIR / "hybrid_rul.joblib").exists()
        required_rul_features = HYBRID_BASE_FEATURES if use_hybrid else RUL_FEATURE_NAMES

        missing = [f for f in required_rul_features if f not in features_df.columns]
        if missing:
            rul_available = False
            rul_unavailable_reason = (
                f"Modelo RUL aguarda retreino sem temperatura. "
                f"{len(missing)} features ausentes (ex: {missing[:3]})."
            )
        elif use_hybrid:
            base_features = features_df[HYBRID_BASE_FEATURES].iloc[-1].to_numpy()
            rul_trend = combined_health * HYBRID_RUL_MAX
            rul_trend_rolling = rul_trend
            hybrid_features = np.concatenate(
                [base_features, np.array([rul_trend, rul_trend_rolling])]
            ).reshape(1, -1)
            hybrid_scaled = rul_scaler.transform(hybrid_features)
            rul_hours = float(rul_model.predict(hybrid_scaled)[0])
            rul_result = {
                "rul_hours": rul_hours,
                "maintenance_window_days": rul_hours / 24.0,
                "risk_level": _hybrid_risk_level(rul_hours),
                "confidence": float(rul_metrics.get("r2", 0.0)),
                "model_version": "hybrid",
            }
        else:
            last_rul_features = features_df[RUL_FEATURE_NAMES].iloc[-1].to_numpy()
            rul_result = predict_rul(rul_model, last_rul_features, rul_scaler)
            rul_result["model_version"] = "forzy_original"
    except Exception as exc:
        rul_available = False
        rul_unavailable_reason = f"Erro ao executar RUL: {exc}"

    if not rul_available:
        rul_result = {
            "rul_hours": None,
            "maintenance_window_days": None,
            "risk_level": "unknown",
            "confidence": None,
            "model_version": "unavailable",
            "available": False,
            "reason": rul_unavailable_reason,
        }
    else:
        rul_result["available"] = True
        rul_result["reason"] = None

    if rul_available and rul_result.get("risk_level") not in (None, "unknown"):
        risk_level = rul_result["risk_level"]
    else:
        risk_level = "unknown"

    if risk_level == "unknown":
        if combined_health >= 0.7:
            overall_status = "healthy"
        elif combined_health >= 0.4:
            overall_status = "warning"
        else:
            overall_status = "critical"
    else:
        overall_status = _overall_status(combined_health, risk_level)

    if not rul_available:
        if overall_status == "healthy":
            recommendation = (
                "Motor operando normalmente (análise de vibração). "
                "Estimativa de vida útil temporariamente indisponível."
            )
        elif overall_status == "warning":
            recommendation = (
                "Padrão de operação irregular detectado na vibração. "
                "Agendar inspeção. Estimativa de vida útil indisponível."
            )
        else:
            recommendation = (
                "ALERTA: anomalia crítica de vibração detectada. "
                "Intervenção imediata recomendada."
            )
    else:
        recommendation = _recommendation(
            overall_status, rul_result["maintenance_window_days"]
        )

    return {
        "timestamp": last_timestamp,
        "estado_operacional": "operando",
        "isolation_forest": {
            "anomaly_score": if_score_raw,
            "is_anomaly": if_is_anomaly,
            "label": if_label,
            "model_version": models.get("if_version", "Forzy"),
        },
        "lstm": {
            "reconstruction_error": lstm_error,
            "severity": lstm_severity,
            "is_anomaly": lstm_is_anomaly,
            "model_version": models.get("lstm_version", "Forzy"),
        },
        "rul": rul_result,
        "combined": {
            "health_score": combined_health,
            "overall_status": overall_status,
            "recommendation": recommendation,
        },
        "health_index": round(combined_health * 100, 1),
    }


def predict_batch(features_df: pd.DataFrame, models: dict) -> pd.DataFrame:
    """Executa inferência em batch (IF + RUL por registro; LSTM apenas no último)."""
    if_model, if_scaler, if_threshold = models["isolation_forest"]
    rul_model, rul_scaler, _ = models["rul"]

    feature_cols = _feature_columns(features_df)
    X_if = if_scaler.transform(features_df[feature_cols].values)
    if_scores = if_model.decision_function(X_if)
    if_is_anomaly = if_scores <= if_threshold
    if_normalized = np.where(if_is_anomaly, 0.0, 1.0)

    missing_rul = [f for f in RUL_FEATURE_NAMES if f not in features_df.columns]
    if missing_rul:
        rul_hours = np.full(len(features_df), np.nan)
        risk_levels = pd.Series(["unknown"] * len(features_df))
    else:
        X_rul = features_df[RUL_FEATURE_NAMES].values
        X_rul_scaled = rul_scaler.transform(X_rul)
        rul_hours = rul_model.predict(X_rul_scaled)
        risk_levels = pd.Series(rul_hours).apply(
            lambda h: "low"
            if h > 5000
            else "medium"
            if h > 1000
            else "high"
            if h > 100
            else "critical"
        )

    return pd.DataFrame(
        {
            "timestamp": features_df["timestamp"].values,
            "if_score": if_scores,
            "if_is_anomaly": if_is_anomaly,
            "rul_hours": rul_hours,
            "risk_level": risk_levels.values,
            "health_score": if_normalized,
        }
    )


def _print_single_result(result: dict) -> None:
    """Imprime resultado formatado de predict_single."""
    print("=" * 60)
    print("RESULTADO DA INFERÊNCIA (predict_single)")
    print("=" * 60)
    print(f"Timestamp: {result['timestamp']}")
    print(f"Estado operacional: {result.get('estado_operacional', 'N/A')}")
    print()

    if result.get("estado_operacional") == "desligado":
        print("Motor desligado — inferência de anomalia não aplicável.")
        print(f"Recomendação: {result['combined']['recommendation']}")
        print("=" * 60)
        return

    print("Isolation Forest:")
    print(f"  anomaly_score: {result['isolation_forest']['anomaly_score']:.6f}")
    print(f"  is_anomaly:    {result['isolation_forest']['is_anomaly']}")
    print(f"  label:         {result['isolation_forest']['label']}")
    print()
    print("LSTM Autoencoder:")
    print(f"  reconstruction_error: {result['lstm']['reconstruction_error']:.6f}")
    print(f"  severity:             {result['lstm']['severity']}")
    print(f"  is_anomaly:           {result['lstm']['is_anomaly']}")
    print()
    print("XGBoost RUL:")
    print(f"  rul_hours:                {result['rul']['rul_hours']:.2f}")
    print(f"  maintenance_window_days:  {result['rul']['maintenance_window_days']:.2f}")
    print(f"  risk_level:               {result['rul']['risk_level']}")
    print(f"  confidence:               {result['rul']['confidence']:.4f}")
    print(f"  model_version:            {result['rul']['model_version']}")
    print()
    print("Combinado:")
    print(f"  health_score:    {result['combined']['health_score']:.4f}")
    print(f"  overall_status:  {result['combined']['overall_status']}")
    print(f"  recommendation:  {result['combined']['recommendation']}")
    print("=" * 60)


if __name__ == "__main__":
    CSV_PATH = PROJECT_ROOT / "History_32026-05-19T11-46-10-920.csv"

    print("Carregando dados e modelos...")
    raw_df = load_raw_csv(CSV_PATH)
    features_df = build_features(raw_df)
    models = load_all_models()

    print("\nExecutando predict_single...")
    single_result = predict_single(features_df, models, raw_df=raw_df)
    _print_single_result(single_result)

    print("\nExecutando predict_batch...")
    batch_result = predict_batch(features_df, models)

    print(f"\nShape do batch: {batch_result.shape[0]} linhas x {batch_result.shape[1]} colunas")
    print("\nPrimeiras 5 linhas:")
    pd.set_option("display.width", 200)
    print(batch_result.head(5).to_string(index=False))

    total_if_anomalies = int(batch_result["if_is_anomaly"].sum())
    print(f"\nTotal de anomalias IF no batch: {total_if_anomalies}")
    print("Distribuição de risk_level:")
    for level, count in batch_result["risk_level"].value_counts().items():
        print(f"  {level}: {count}")
