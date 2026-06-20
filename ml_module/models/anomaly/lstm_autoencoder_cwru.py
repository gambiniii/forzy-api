"""
LSTM Autoencoder treinado no CWRU Bearing Dataset (apenas amostras normais).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_module.models.anomaly.lstm_autoencoder import (
    _reconstruction_errors,
    build_model,
    compute_threshold,
    save_model,
    train,
)

WINDOW_SIZE = 512
SEQ_LEN = 64
N_FEATURES = WINDOW_SIZE // SEQ_LEN  # 8
FEATURE_COLS = [f"feat_{i}" for i in range(WINDOW_SIZE)]


def _reshape_to_sequences(X: np.ndarray, seq_len: int = SEQ_LEN) -> np.ndarray:
    """Reshape (n_samples, 512) para (n_samples, seq_len, n_features)."""
    n_samples = X.shape[0]
    return X.reshape(n_samples, seq_len, WINDOW_SIZE // seq_len)


def load_cwru_sequences(
    parquet_path: Path,
    seq_len: int = SEQ_LEN,
) -> tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    """Carrega CWRU, separa normal/falha, escala e reshape em sequências temporais."""
    df = pd.read_parquet(parquet_path)

    X_normal = df.loc[df["label"] == 0, FEATURE_COLS].to_numpy()
    X_fault = df.loc[df["label"] == 1, FEATURE_COLS].to_numpy()

    scaler = MinMaxScaler()
    X_normal_scaled = scaler.fit_transform(X_normal)
    X_fault_scaled = scaler.transform(X_fault)

    X_normal_seq = _reshape_to_sequences(X_normal_scaled, seq_len)
    X_fault_seq = _reshape_to_sequences(X_fault_scaled, seq_len)

    return X_normal_seq, X_fault_seq, scaler


def evaluate_on_cwru(
    model,
    X_normal_seq: np.ndarray,
    X_fault_seq: np.ndarray,
    threshold: float,
) -> dict:
    """Avalia separação entre normais e falhas via erros de reconstrução."""
    errors_normal = _reconstruction_errors(model, X_normal_seq)
    errors_fault = _reconstruction_errors(model, X_fault_seq)

    normal_predicted_anomaly = errors_normal > threshold
    fault_predicted_anomaly = errors_fault > threshold

    true_positives = int(fault_predicted_anomaly.sum())
    true_negatives = int((~normal_predicted_anomaly).sum())
    false_positives = int(normal_predicted_anomaly.sum())
    false_negatives = int((~fault_predicted_anomaly).sum())

    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "errors_normal": errors_normal,
        "errors_fault": errors_fault,
        "errors_normal_mean": float(errors_normal.mean()),
        "errors_fault_mean": float(errors_fault.mean()),
        "true_positives": true_positives,
        "true_negatives": true_negatives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def train_on_cwru(parquet_path: Path, output_dir: Path) -> dict:
    """Pipeline completo: carregar, treinar, avaliar e salvar LSTM Autoencoder CWRU."""
    X_normal_seq, X_fault_seq, scaler = load_cwru_sequences(parquet_path)

    model = build_model(window_size=SEQ_LEN, n_features=N_FEATURES)
    history = train(model, X_normal_seq, epochs=50, batch_size=64)
    threshold, _ = compute_threshold(model, X_normal_seq)
    eval_metrics = evaluate_on_cwru(model, X_normal_seq, X_fault_seq, threshold)

    save_model(model, scaler, threshold, output_dir)

    train_loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    return {
        "X_normal_seq_shape": X_normal_seq.shape,
        "X_fault_seq_shape": X_fault_seq.shape,
        "epochs_trained": len(train_loss),
        "final_train_loss": float(train_loss[-1]),
        "final_val_loss": float(val_loss[-1]),
        "threshold": threshold,
        **eval_metrics,
    }


if __name__ == "__main__":
    PARQUET_PATH = PROJECT_ROOT / "ml_module/data/processed/cwru_processed.parquet"
    OUTPUT_DIR = Path(__file__).resolve().parent / "saved_cwru"

    print("=" * 60)
    print("LSTM Autoencoder — CWRU Bearing Dataset")
    print("=" * 60)

    metrics = train_on_cwru(PARQUET_PATH, OUTPUT_DIR)

    print(f"\nShape de X_normal_seq: {metrics['X_normal_seq_shape']}")
    print(f"Shape de X_fault_seq:  {metrics['X_fault_seq_shape']}")
    print(f"Épocas treinadas: {metrics['epochs_trained']}")
    print(f"Loss final de treino: {metrics['final_train_loss']:.6f}")
    print(f"Loss final de validação: {metrics['final_val_loss']:.6f}")
    print(f"Threshold calculado: {metrics['threshold']:.6f}")

    print(f"\nErro médio reconstrução normais: {metrics['errors_normal_mean']:.6f}")
    print(f"Erro médio reconstrução falhas:  {metrics['errors_fault_mean']:.6f}")

    print("\nMétricas de separação:")
    print(f"  True Positives:  {metrics['true_positives']}")
    print(f"  True Negatives:  {metrics['true_negatives']}")
    print(f"  False Positives: {metrics['false_positives']}")
    print(f"  False Negatives: {metrics['false_negatives']}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")

    conclusion = "Modelo apto para produção" if metrics["f1"] > 0.75 else "Requer ajuste"
    print(f"\nConclusão: {conclusion}")
    print(f"\nModelo salvo em: {OUTPUT_DIR.resolve()}")
    print("=" * 60)
