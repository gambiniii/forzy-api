"""
Isolation Forest treinado no CWRU Bearing Dataset (apenas amostras normais).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_module.models.baseline.isolation_forest import load_model, save_model

WINDOW_SIZE = 512
FEATURE_COLS = [f"feat_{i}" for i in range(WINDOW_SIZE)]


def load_cwru(parquet_path: Path) -> tuple[np.ndarray, np.ndarray, StandardScaler, pd.DataFrame]:
    """Carrega CWRU processado, separa normal/falha e escala com fit apenas no normal."""
    df = pd.read_parquet(parquet_path)

    X_normal = df.loc[df["label"] == 0, FEATURE_COLS].to_numpy()
    X_fault = df.loc[df["label"] == 1, FEATURE_COLS].to_numpy()

    scaler = StandardScaler()
    X_normal_scaled = scaler.fit_transform(X_normal)
    X_fault_scaled = scaler.transform(X_fault)

    return X_normal_scaled, X_fault_scaled, scaler, df


def extract_statistical_features(X: np.ndarray) -> np.ndarray:
    """Extrai 9 features estatísticas por janela de sinal (n_samples, 512)."""
    mean = X.mean(axis=1)
    std = X.std(axis=1)
    min_val = X.min(axis=1)
    max_val = X.max(axis=1)
    rms = np.sqrt(np.mean(X ** 2, axis=1))
    peak_to_peak = max_val - min_val
    skewness = skew(X, axis=1)
    kurt = kurtosis(X, axis=1)
    max_abs = np.max(np.abs(X), axis=1)
    crest_factor = np.divide(
        max_abs,
        rms,
        out=np.zeros_like(max_abs),
        where=rms != 0,
    )

    return np.column_stack(
        [mean, std, min_val, max_val, rms, peak_to_peak, skewness, kurt, crest_factor]
    )


def train_on_cwru(parquet_path: Path, output_dir: Path) -> dict:
    """Treina IF apenas em normais e avalia separação nas amostras com falha."""
    X_normal_scaled, X_fault_scaled, scaler, df = load_cwru(parquet_path)

    X_normal_features = extract_statistical_features(X_normal_scaled)
    X_fault_features = extract_statistical_features(X_fault_scaled)

    model = IsolationForest(
        n_estimators=300,
        contamination=0.05,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_normal_features)

    scores_normal = model.decision_function(X_normal_features)
    scores_fault = model.decision_function(X_fault_features)
    threshold = float(np.percentile(scores_normal, 5))

    normal_predicted_anomaly = scores_normal <= threshold
    fault_predicted_anomaly = scores_fault <= threshold

    true_positives = int(fault_predicted_anomaly.sum())
    true_negatives = int((~normal_predicted_anomaly).sum())
    false_positives = int(normal_predicted_anomaly.sum())
    false_negatives = int((~fault_predicted_anomaly).sum())

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    save_model(model, scaler, threshold, output_dir)

    return {
        "X_normal_shape": X_normal_scaled.shape,
        "X_fault_shape": X_fault_scaled.shape,
        "features_shape": X_normal_features.shape,
        "scores_normal_mean": float(scores_normal.mean()),
        "scores_fault_mean": float(scores_fault.mean()),
        "threshold": threshold,
        "true_positives": true_positives,
        "true_negatives": true_negatives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "df_completo": df,
    }


if __name__ == "__main__":
    PARQUET_PATH = PROJECT_ROOT / "ml_module/data/processed/cwru_processed.parquet"
    OUTPUT_DIR = Path(__file__).resolve().parent / "saved_cwru"

    print("=" * 60)
    print("Isolation Forest — CWRU Bearing Dataset")
    print("=" * 60)

    metrics = train_on_cwru(PARQUET_PATH, OUTPUT_DIR)

    print(f"\nShape de X_normal: {metrics['X_normal_shape']}")
    print(f"Shape de X_fault:  {metrics['X_fault_shape']}")
    print(f"Features estatísticas extraídas: {metrics['features_shape']}")

    print(f"\nScore médio normais:    {metrics['scores_normal_mean']:.6f}")
    print(f"Score médio anomalias:  {metrics['scores_fault_mean']:.6f}")
    print(f"Threshold calculado:    {metrics['threshold']:.6f}")

    print("\nMétricas de separação:")
    print(f"  True Positives:  {metrics['true_positives']}")
    print(f"  True Negatives:  {metrics['true_negatives']}")
    print(f"  False Positives: {metrics['false_positives']}")
    print(f"  False Negatives: {metrics['false_negatives']}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")

    conclusion = "Modelo apto para produção" if metrics["f1"] > 0.80 else "Requer ajuste"
    print(f"\nConclusão: {conclusion}")
    print(f"\nModelo salvo em: {OUTPUT_DIR.resolve()}")
    print("=" * 60)
