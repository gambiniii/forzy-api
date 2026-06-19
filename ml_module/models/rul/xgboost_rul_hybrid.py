"""
XGBoost RUL híbrido — labels inteligentes para Forzy
combinando scores de anomalia (IF + LSTM) e tendência temporal.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_module.features.feature_engineering import build_features, load_raw_csv
from ml_module.models.anomaly.lstm_autoencoder import (
    _reconstruction_errors,
    create_sequences,
    load_model as load_lstm_model,
)
from ml_module.models.baseline.isolation_forest import load_model as load_if_model
from ml_module.models.rul.xgboost_rul_cmapss import load_cmapss

FORZY_CSV = PROJECT_ROOT / "History_32026-05-19T11-46-10-920.csv"
# Modelos Forzy (saved/) — saved_cwru usa features CWRU incompatíveis com build_features
IF_MODEL_DIR = PROJECT_ROOT / "ml_module/models/baseline/saved"
LSTM_MODEL_DIR = PROJECT_ROOT / "ml_module/models/anomaly/saved"
CMAPSS_TRAIN = PROJECT_ROOT / "ml_module/data/processed/cmapss_train.parquet"
CMAPSS_TEST = PROJECT_ROOT / "ml_module/data/processed/cmapss_test.parquet"
OUTPUT_DIR = PROJECT_ROOT / "ml_module/models/rul/saved_hybrid"

VIDA_UTIL_MAX = 175_200.0
RUL_MIN_ESPERADO = 80.0
RUL_MAX_ESPERADO = 160.0
WINDOW_SIZE = 60

FORZY_RUL_FEATURES = [
    "elapsed_seconds",
    "port1_velocidade_rolling_mean_50",
    "port1_velocidade_rolling_mean_200",
    "port1_velocidade_rolling_std_50",
    "port1_aceleracao_rolling_mean_50",
    "port1_aceleracao_rolling_mean_200",
    "port1_temperatura_rolling_mean_50",
    "port1_temperatura_rolling_mean_200",
    "port1_velocidade_desvio_iso",
    "port1_delta_velocidade",
    "port1_delta_temperatura",
    "port2_velocidade_rolling_mean_50",
    "port2_velocidade_rolling_mean_200",
    "port2_velocidade_rolling_std_50",
    "port2_aceleracao_rolling_mean_50",
    "port2_aceleracao_rolling_mean_200",
    "port2_temperatura_rolling_mean_50",
    "port2_temperatura_rolling_mean_200",
    "port2_velocidade_desvio_iso",
    "port2_delta_velocidade",
    "port2_delta_temperatura",
    "velocidade_media_global",
    "aceleracao_media_global",
    "temperatura_media_global",
    "delta_temperatura_entre_sensores",
]

MODEL_FILENAME = "hybrid_rul.joblib"
SCALER_FILENAME = "hybrid_rul_scaler.joblib"
METRICS_FILENAME = "hybrid_rul_metrics.json"


def _forzy_feature_matrix(features_df: pd.DataFrame) -> np.ndarray:
    """Extrai matriz de features Forzy (sem timestamp) na ordem do treino."""
    feature_cols = [col for col in features_df.columns if col != "timestamp"]
    return features_df[feature_cols].values


def _min_max_normalize_array(values: np.ndarray) -> np.ndarray:
    """Normaliza array para [0, 1]."""
    min_val = values.min()
    max_val = values.max()
    if max_val == min_val:
        return np.zeros_like(values, dtype=np.float64)
    return (values - min_val) / (max_val - min_val)


def generate_hybrid_rul_labels(
    features_df: pd.DataFrame,
    if_model,
    if_scaler,
    if_threshold: float,
    lstm_model,
    lstm_scaler,
    lstm_threshold: float,
    window_size: int = WINDOW_SIZE,
) -> np.ndarray:
    """Gera labels de RUL híbridos a partir de IF, LSTM e tendência temporal."""
    del if_threshold, lstm_threshold  # usados na inferência externa; não necessários aqui

    n_samples = len(features_df)
    X_raw = _forzy_feature_matrix(features_df)

    # Fonte 1 — IF anomaly score (peso 0.3)
    X_if = if_scaler.transform(X_raw)
    if_scores = if_model.decision_function(X_if)
    if_health = _min_max_normalize_array(if_scores)

    # Fonte 2 — LSTM reconstruction error (peso 0.5)
    X_lstm = lstm_scaler.transform(X_raw)
    X_seq = create_sequences(X_lstm, window_size=window_size)
    lstm_errors = _reconstruction_errors(lstm_model, X_seq)
    error_max = lstm_errors.max() if lstm_errors.max() > 0 else 1.0
    lstm_health_seq = 1.0 - (lstm_errors / error_max)

    lstm_health = np.full(n_samples, lstm_health_seq[0], dtype=np.float64)
    lstm_health[window_size - 1 :] = lstm_health_seq

    # Fonte 3 — Tendência temporal física (peso 0.2)
    elapsed = features_df["elapsed_seconds"].to_numpy()
    elapsed_max = elapsed.max() if elapsed.max() > 0 else 1.0
    elapsed_norm = elapsed / elapsed_max
    time_health = 1.0 - elapsed_norm

    combined_health = 0.3 * if_health + 0.5 * lstm_health + 0.2 * time_health
    rul_hours = RUL_MIN_ESPERADO + combined_health * (RUL_MAX_ESPERADO - RUL_MIN_ESPERADO)

    return rul_hours.astype(np.float64)


def select_forzy_rul_features(
    features_df: pd.DataFrame,
    extra_features: list[str] | None = None,
) -> tuple[np.ndarray, StandardScaler, list[str]]:
    """Seleciona e escala features Forzy relevantes para RUL."""
    extra_features = extra_features or []
    feature_names = FORZY_RUL_FEATURES + extra_features

    missing = [col for col in feature_names if col not in features_df.columns]
    if missing:
        raise KeyError(f"Features ausentes no dataframe: {missing}")

    X = features_df[feature_names].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, scaler, feature_names.copy()


def train_hybrid(
    features_df: pd.DataFrame,
    rul_labels: np.ndarray,
    output_dir: Path,
) -> dict:
    """Treina XGBoost híbrido com split estratificado e salva artefatos."""
    features_df = features_df.copy()
    features_df["rul_trend"] = rul_labels
    features_df["rul_trend_rolling"] = (
        pd.Series(rul_labels).rolling(50, min_periods=1).mean().values
    )

    extra_features = ["rul_trend", "rul_trend_rolling"]
    X, scaler, feature_names = select_forzy_rul_features(
        features_df, extra_features=extra_features
    )

    quartiles = pd.qcut(rul_labels, q=4, labels=False, duplicates="drop")
    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=42,
        stratify=quartiles,
    )

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = rul_labels[train_idx], rul_labels[test_idx]
    elapsed_test = features_df["elapsed_seconds"].values[test_idx]

    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=10,
        gamma=0.3,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    if len(elapsed_test) > 1 and np.std(elapsed_test) > 0 and np.std(y_pred) > 0:
        temporal_correlation = float(np.corrcoef(elapsed_test, y_pred)[0, 1])
    else:
        temporal_correlation = 0.0

    metrics = {
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "r2": float(r2_score(y_test, y_pred)),
        "temporal_correlation": temporal_correlation,
        "feature_names": feature_names,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / MODEL_FILENAME)
    joblib.dump(scaler, output_dir / SCALER_FILENAME)
    with open(output_dir / METRICS_FILENAME, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return {
        **metrics,
        "model": model,
        "scaler": scaler,
        "y_test": y_test,
        "y_pred": y_pred,
        "test_timestamps": features_df["timestamp"].iloc[test_idx].reset_index(drop=True),
        "features_df_augmented": features_df,
        "extra_features": extra_features,
    }


if __name__ == "__main__":
    raw_df = load_raw_csv(FORZY_CSV)
    features_df = build_features(raw_df)

    if_model, if_scaler, if_threshold = load_if_model(IF_MODEL_DIR)
    lstm_model, lstm_scaler, lstm_threshold = load_lstm_model(LSTM_MODEL_DIR)

    rul_labels = generate_hybrid_rul_labels(
        features_df,
        if_model,
        if_scaler,
        if_threshold,
        lstm_model,
        lstm_scaler,
        lstm_threshold,
        window_size=WINDOW_SIZE,
    )

    elapsed = features_df["elapsed_seconds"].to_numpy()
    if len(elapsed) > 1 and np.std(elapsed) > 0 and np.std(rul_labels) > 0:
        labels_correlation = float(np.corrcoef(elapsed, rul_labels)[0, 1])
    else:
        labels_correlation = 0.0

    print("Distribuição dos labels gerados (horas):")
    print(f"  min:  {rul_labels.min():.2f}")
    print(f"  max:  {rul_labels.max():.2f}")
    print(f"  mean: {rul_labels.mean():.2f}")
    print(f"  std:  {rul_labels.std():.2f}")
    print(f"\nCorrelação labels vs elapsed_seconds: {labels_correlation:.4f}")

    result = train_hybrid(features_df, rul_labels, OUTPUT_DIR)

    print("\nMétricas do modelo (teste 20%):")
    print(f"  MAE:  {result['mae']:.2f} horas")
    print(f"  RMSE: {result['rmse']:.2f} horas")
    print(f"  R²:   {result['r2']:.4f}")
    print(f"\nCorrelação temporal das predições: {result['temporal_correlation']:.4f}")

    aug_df = result["features_df_augmented"]
    full_X, _full_scaler, _ = select_forzy_rul_features(
        aug_df, extra_features=result["extra_features"]
    )
    full_pred = result["model"].predict(full_X)
    pred_df = pd.DataFrame(
        {
            "timestamp": features_df["timestamp"],
            "rul_hours": full_pred,
        }
    )

    print("\nTop 5 registros com menor RUL predito:")
    for _, row in pred_df.nsmallest(5, "rul_hours").iterrows():
        print(f"  {row['timestamp']} | RUL: {row['rul_hours']:.2f}h")

    if result["r2"] > 0.70 and result["temporal_correlation"] < -0.30:
        print("\nConclusão: RUL híbrido funcional")
    else:
        print("\nConclusão: Ajuste necessário")

    _ = load_cmapss(CMAPSS_TRAIN, CMAPSS_TEST)
