"""
Modelo XGBoost para estimativa de RUL (Remaining Useful Life) por degradação relativa.
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

VIDA_UTIL_MAX = 175_200  # 20 anos em horas

RUL_FEATURE_NAMES = [
    "elapsed_seconds",
    "port1_velocidade_rolling_mean_50",
    "port1_velocidade_rolling_mean_200",
    "port1_velocidade_rolling_std_50",
    "port1_velocidade_rolling_std_200",
    "port1_aceleracao_rolling_mean_50",
    "port1_aceleracao_rolling_mean_200",
    "port1_temperatura_rolling_mean_50",
    "port1_temperatura_rolling_mean_200",
    "port1_delta_velocidade",
    "port1_delta_aceleracao",
    "port1_delta_temperatura",
    "port1_velocidade_desvio_iso",
    "port2_velocidade_rolling_mean_50",
    "port2_velocidade_rolling_mean_200",
    "port2_velocidade_rolling_std_50",
    "port2_velocidade_rolling_std_200",
    "port2_aceleracao_rolling_mean_50",
    "port2_aceleracao_rolling_mean_200",
    "port2_temperatura_rolling_mean_50",
    "port2_temperatura_rolling_mean_200",
    "port2_delta_velocidade",
    "port2_delta_aceleracao",
    "port2_delta_temperatura",
    "port2_velocidade_desvio_iso",
    "velocidade_media_global",
    "aceleracao_media_global",
    "temperatura_media_global",
    "delta_temperatura_entre_sensores",
]

MODEL_FILENAME = "xgboost_rul.joblib"
SCALER_FILENAME = "xgboost_rul_scaler.joblib"
METRICS_FILENAME = "xgboost_rul_metrics.json"


def _min_max_normalize(series: pd.Series) -> pd.Series:
    """Normaliza série para o intervalo [0, 1]."""
    min_val = series.min()
    max_val = series.max()
    if max_val == min_val:
        return pd.Series(0.0, index=series.index)
    return (series - min_val) / (max_val - min_val)


def create_rul_labels(features_df: pd.DataFrame) -> np.ndarray:
    """Estima RUL em horas com base em índice de saúde derivado da degradação relativa."""
    vel_media = (
        features_df["port1_velocidade_desvio_iso"] + features_df["port2_velocidade_desvio_iso"]
    ) / 2
    temp_media = features_df["temperatura_media_global"]
    acc_media = features_df["aceleracao_media_global"]

    vel_norm = _min_max_normalize(vel_media)
    temp_norm = _min_max_normalize(temp_media)
    acc_norm = _min_max_normalize(acc_media)

    health_score = 0.5 * vel_norm + 0.3 * temp_norm + 0.2 * acc_norm
    rul_hours = VIDA_UTIL_MAX * (1.0 - health_score)
    rul_hours = rul_hours.clip(0, VIDA_UTIL_MAX)

    return rul_hours.to_numpy()


def prepare_data(
    features_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, StandardScaler, list[str]]:
    """Seleciona features relevantes, escala com StandardScaler e retorna X, y e scaler."""
    missing = [col for col in RUL_FEATURE_NAMES if col not in features_df.columns]
    if missing:
        raise KeyError(f"Features ausentes no dataframe: {missing}")

    X = features_df[RUL_FEATURE_NAMES].values
    y_rul = create_rul_labels(features_df)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y_rul, scaler, RUL_FEATURE_NAMES.copy()


def train(
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[xgb.XGBRegressor, np.ndarray, np.ndarray, dict]:
    """Treina XGBRegressor com split temporal 80/20 e retorna métricas de avaliação."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "r2": float(r2_score(y_test, y_pred)),
    }

    model.r2_score_ = metrics["r2"]
    model.feature_names_ = getattr(model, "feature_names_", None)

    return model, X_test, y_test, metrics


def _risk_level(rul_hours: float) -> str:
    """Classifica nível de risco com base nas horas de RUL restantes."""
    if rul_hours > 5000:
        return "low"
    if rul_hours > 1000:
        return "medium"
    if rul_hours > 100:
        return "high"
    return "critical"


def predict(
    model: xgb.XGBRegressor,
    X: np.ndarray,
    scaler: StandardScaler,
) -> dict:
    """Prediz RUL a partir de features brutas, aplicando scaler antes da inferência."""
    if X.ndim == 1:
        X = X.reshape(1, -1)

    X_scaled = scaler.transform(X)
    rul_hours = float(model.predict(X_scaled)[0])
    maintenance_window_days = rul_hours / 24
    confidence = float(getattr(model, "r2_score_", 0.0))

    return {
        "rul_hours": rul_hours,
        "maintenance_window_days": maintenance_window_days,
        "confidence": confidence,
        "risk_level": _risk_level(rul_hours),
    }


def save_model(
    model: xgb.XGBRegressor,
    scaler: StandardScaler | None,
    metrics: dict,
    output_dir: Path,
) -> None:
    """Persiste modelo, scaler e métricas de avaliação."""
    output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, output_dir / MODEL_FILENAME)
    if scaler is not None:
        joblib.dump(scaler, output_dir / SCALER_FILENAME)

    with open(output_dir / METRICS_FILENAME, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def load_model(model_dir: Path) -> tuple[xgb.XGBRegressor, StandardScaler | None, dict]:
    """Carrega modelo, scaler e métricas salvos."""
    model = joblib.load(model_dir / MODEL_FILENAME)
    scaler_path = model_dir / SCALER_FILENAME
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None

    with open(model_dir / METRICS_FILENAME, encoding="utf-8") as f:
        metrics = json.load(f)

    if "r2" in metrics and not hasattr(model, "r2_score_"):
        model.r2_score_ = metrics["r2"]

    return model, scaler, metrics


if __name__ == "__main__":
    CSV_PATH = PROJECT_ROOT / "History_32026-05-19T11-46-10-920.csv"
    OUTPUT_DIR = Path(__file__).resolve().parent / "saved"

    raw_df = load_raw_csv(CSV_PATH)
    features_df = build_features(raw_df)

    y_rul = create_rul_labels(features_df)
    X_scaled, y_rul, scaler, feature_names = prepare_data(features_df)
    model, X_test, y_test, metrics = train(X_scaled, y_rul)

    print(f"Shape de X: {X_scaled.shape}")
    print(
        f"Distribuição de y_rul (horas) — "
        f"min: {y_rul.min():.2f}, max: {y_rul.max():.2f}, "
        f"mean: {y_rul.mean():.2f}, std: {y_rul.std():.2f}"
    )

    print("\nMétricas de avaliação (teste):")
    print(f"  MAE:  {metrics['mae']:.2f} horas")
    print(f"  RMSE: {metrics['rmse']:.2f} horas")
    print(f"  R²:   {metrics['r2']:.4f}")

    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(
        ascending=False
    )
    print("\nTop 10 feature importances:")
    for rank, (name, importance) in enumerate(importances.head(10).items(), start=1):
        print(f"  {rank}. {name}: {importance:.4f}")

    last_features = features_df[RUL_FEATURE_NAMES].iloc[-1].to_numpy()
    last_prediction = predict(model, last_features, scaler)

    print("\nPredição para o último registro do dataset:")
    print(f"  RUL (horas): {last_prediction['rul_hours']:.2f}")
    print(f"  Janela de manutenção (dias): {last_prediction['maintenance_window_days']:.2f}")
    print(f"  Nível de risco: {last_prediction['risk_level']}")
    print(f"  Confiança (R²): {last_prediction['confidence']:.4f}")

    percentiles = np.percentile(y_rul, [25, 50, 75])
    print("\nEstatísticas gerais de RUL do dataset (percentis em horas):")
    print(f"  P25: {percentiles[0]:.2f}")
    print(f"  P50: {percentiles[1]:.2f}")
    print(f"  P75: {percentiles[2]:.2f}")

    save_model(model, scaler, metrics, OUTPUT_DIR)
    print(f"\nModelo salvo em: {OUTPUT_DIR.resolve()}")
    print(f"  - {MODEL_FILENAME}")
    print(f"  - {SCALER_FILENAME}")
    print(f"  - {METRICS_FILENAME}")
