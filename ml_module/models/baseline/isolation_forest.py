"""
Modelo baseline de detecção de anomalias com Isolation Forest.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_module.features.feature_engineering import build_features, load_raw_csv

MODEL_FILENAME = "isolation_forest.joblib"
SCALER_FILENAME = "isolation_forest_scaler.joblib"
THRESHOLD_FILENAME = "isolation_forest_threshold.json"


def prepare_data(features_df: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    """Remove timestamp, normaliza com StandardScaler e retorna array + scaler."""
    feature_cols = [col for col in features_df.columns if col != "timestamp"]
    X = features_df[feature_cols].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, scaler


def train(X: np.ndarray) -> IsolationForest:
    """Treina IsolationForest em modo não supervisionado."""
    model = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X)
    return model


def compute_threshold(model: IsolationForest, X: np.ndarray) -> float:
    """Define threshold no percentil 5 dos anomaly scores (mais negativos = anomalia)."""
    scores = model.decision_function(X)
    return float(np.percentile(scores, 5))


def predict(model: IsolationForest, X: np.ndarray, threshold: float) -> dict:
    """Gera scores, labels e métricas de anomalia com base no threshold."""
    scores = model.decision_function(X)
    labels = np.where(scores <= threshold, -1, 1).astype(int)
    n_anomalies = int((labels == -1).sum())
    anomaly_rate = float(n_anomalies / len(labels) * 100)

    return {
        "scores": scores,
        "labels": labels,
        "anomaly_rate": anomaly_rate,
        "n_anomalies": n_anomalies,
    }


def save_model(
    model: IsolationForest,
    scaler: StandardScaler,
    threshold: float,
    output_dir: Path,
) -> None:
    """Persiste modelo, scaler e threshold em disco."""
    output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, output_dir / MODEL_FILENAME)
    joblib.dump(scaler, output_dir / SCALER_FILENAME)

    with open(output_dir / THRESHOLD_FILENAME, "w", encoding="utf-8") as f:
        json.dump({"threshold": threshold}, f, indent=2)


def load_model(model_dir: Path) -> tuple[IsolationForest, StandardScaler, float]:
    """Carrega modelo, scaler e threshold salvos."""
    model = joblib.load(model_dir / MODEL_FILENAME)
    scaler = joblib.load(model_dir / SCALER_FILENAME)

    with open(model_dir / THRESHOLD_FILENAME, encoding="utf-8") as f:
        threshold = float(json.load(f)["threshold"])

    return model, scaler, threshold


if __name__ == "__main__":
    csv_path = PROJECT_ROOT / "History_32026-05-19T11-46-10-920.csv"
    output_dir = Path(__file__).resolve().parent / "saved"

    raw_df = load_raw_csv(csv_path)
    features_df = build_features(raw_df)
    timestamps = features_df["timestamp"].copy()

    X, scaler = prepare_data(features_df)
    model = train(X)
    threshold = compute_threshold(model, X)
    results = predict(model, X, threshold)

    scores = results["scores"]

    print(f"Shape dos dados de treino: {X.shape}")
    print(f"Threshold calculado: {threshold:.6f}")
    print(f"Taxa de anomalias detectadas: {results['anomaly_rate']:.2f}%")
    print(f"Score mínimo: {scores.min():.6f}")
    print(f"Score máximo: {scores.max():.6f}")
    print(f"Score médio: {scores.mean():.6f}")

    print("\nTop 5 registros mais anômalos:")
    top5_indices = np.argsort(scores)[:5]
    for rank, idx in enumerate(top5_indices, start=1):
        print(
            f"  {rank}. índice={idx} | "
            f"timestamp={timestamps.iloc[idx]} | "
            f"score={scores[idx]:.6f}"
        )

    save_model(model, scaler, threshold, output_dir)
    print(f"\nModelo salvo em: {output_dir.resolve()}")
    print(f"  - {MODEL_FILENAME}")
    print(f"  - {SCALER_FILENAME}")
    print(f"  - {THRESHOLD_FILENAME}")
