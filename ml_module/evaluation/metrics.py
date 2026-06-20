"""
Avaliação e relatório dos modelos treinados — CWRU, CMAPSS e RUL híbrido.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_module.models.anomaly.lstm_autoencoder import load_model as load_lstm_model
from ml_module.models.anomaly.lstm_autoencoder_cwru import (
    evaluate_on_cwru,
    load_cwru_sequences,
)
from ml_module.models.baseline.isolation_forest import load_model as load_if_model
from ml_module.models.baseline.isolation_forest_cwru import (
    extract_statistical_features,
    load_cwru,
)
from ml_module.models.rul.xgboost_rul import load_model as load_rul_model
from ml_module.models.rul.xgboost_rul_cmapss import _eval_mask, _nasa_score, load_cmapss

CWRU_PARQUET = PROJECT_ROOT / "ml_module/data/processed/cwru_processed.parquet"
CMAPSS_TRAIN = PROJECT_ROOT / "ml_module/data/processed/cmapss_train.parquet"
CMAPSS_TEST = PROJECT_ROOT / "ml_module/data/processed/cmapss_test.parquet"
IF_CWRU_DIR = PROJECT_ROOT / "ml_module/models/baseline/saved_cwru"
LSTM_CWRU_DIR = PROJECT_ROOT / "ml_module/models/anomaly/saved_cwru"
RUL_CMAPSS_DIR = PROJECT_ROOT / "ml_module/models/rul/saved_cmapss"
HYBRID_RUL_DIR = PROJECT_ROOT / "ml_module/models/rul/saved_hybrid"


def evaluate_isolation_forest(model_dir: Path, parquet_path: Path) -> dict:
    """Avalia Isolation Forest CWRU em dados normais e com falha."""
    model, _scaler, threshold = load_if_model(model_dir)
    X_normal_scaled, X_fault_scaled, _, _ = load_cwru(parquet_path)

    X_normal_features = extract_statistical_features(X_normal_scaled)
    X_fault_features = extract_statistical_features(X_fault_scaled)

    scores_normal = model.decision_function(X_normal_features)
    scores_fault = model.decision_function(X_fault_features)

    normal_predicted_anomaly = scores_normal <= threshold
    fault_predicted_anomaly = scores_fault <= threshold

    true_positives = int(fault_predicted_anomaly.sum())
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
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "threshold": float(threshold),
    }


def evaluate_lstm(model_dir: Path, parquet_path: Path) -> dict:
    """Avalia LSTM Autoencoder CWRU em sequências normais e com falha."""
    model, _scaler, threshold = load_lstm_model(model_dir)
    X_normal_seq, X_fault_seq, _ = load_cwru_sequences(parquet_path)

    metrics = evaluate_on_cwru(model, X_normal_seq, X_fault_seq, threshold)
    return {
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "threshold": float(threshold),
        "mean_error_normal": metrics["errors_normal_mean"],
        "mean_error_fault": metrics["errors_fault_mean"],
    }


def evaluate_xgboost_rul(model_dir: Path, test_path: Path) -> dict:
    """Avalia XGBoost RUL CMAPSS no conjunto de teste."""
    model, _scaler, _metrics = load_rul_model(model_dir)
    _X_train, _y_train, X_test, y_test, _feature_names = load_cmapss(
        CMAPSS_TRAIN, test_path
    )

    eval_mask = _eval_mask(y_test)
    X_test_eval = X_test[eval_mask]
    y_test_eval = y_test[eval_mask]

    y_pred = model.predict(X_test_eval)

    return {
        "mae": float(mean_absolute_error(y_test_eval, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test_eval, y_pred))),
        "r2": float(r2_score(y_test_eval, y_pred)),
        "nasa_score": _nasa_score(y_test_eval, y_pred),
    }


def evaluate_hybrid_rul(model_dir: Path | None = None) -> dict:
    """Carrega modelo e métricas salvas do XGBoost RUL híbrido."""
    model_dir = model_dir or HYBRID_RUL_DIR
    joblib_path = model_dir / "hybrid_rul.joblib"
    metrics_path = model_dir / "hybrid_rul_metrics.json"

    if not joblib_path.exists() or not metrics_path.exists():
        raise FileNotFoundError(
            f"Modelo híbrido não encontrado em {model_dir}. "
            "Execute: python ml_module/models/rul/xgboost_rul_hybrid.py"
        )

    _model = joblib.load(joblib_path)
    with open(metrics_path, encoding="utf-8") as f:
        saved = json.load(f)

    return {
        "r2": float(saved["r2"]),
        "mae": float(saved["mae"]),
        "rmse": float(saved["rmse"]),
        "temporal_correlation": float(saved["temporal_correlation"]),
        "model_loaded": True,
    }


def generate_report() -> None:
    """Gera relatório formatado com métricas dos modelos CWRU, CMAPSS e híbrido."""
    if_metrics = evaluate_isolation_forest(IF_CWRU_DIR, CWRU_PARQUET)
    lstm_metrics = evaluate_lstm(LSTM_CWRU_DIR, CWRU_PARQUET)
    rul_metrics = evaluate_xgboost_rul(RUL_CMAPSS_DIR, CMAPSS_TEST)
    hybrid_metrics = evaluate_hybrid_rul(HYBRID_RUL_DIR)

    approved = (
        if_metrics["f1"] > 0.80
        and lstm_metrics["f1"] > 0.80
        and rul_metrics["r2"] > 0.85
        and hybrid_metrics["r2"] > 0.70
        and hybrid_metrics["temporal_correlation"] < -0.30
    )
    status = "APROVADO" if approved else "REPROVADO"

    hybrid_corr = hybrid_metrics["temporal_correlation"]

    print("+" + "=" * 42 + "+")
    print("|   FORZY DIGITAL TWIN - MODEL REPORT     |")
    print("+" + "=" * 42 + "+")
    print("| Isolation Forest (CWRU)                 |")
    print(
        f"|   F1: {if_metrics['f1']:.4f} | Recall: {if_metrics['recall']:.4f}          |"
    )
    print("+" + "=" * 42 + "+")
    print("| LSTM Autoencoder (CWRU)                 |")
    print(
        f"|   F1: {lstm_metrics['f1']:.4f} | Recall: {lstm_metrics['recall']:.4f}          |"
    )
    print(
        f"|   Erro normal: {lstm_metrics['mean_error_normal']:.4f} | "
        f"Falha: {lstm_metrics['mean_error_fault']:.4f}  |"
    )
    print("+" + "=" * 42 + "+")
    print("| XGBoost RUL (CMAPSS)                    |")
    print(
        f"|   R2: {rul_metrics['r2']:.4f} | MAE: {rul_metrics['mae']:.2f} ciclos       |"
    )
    print("+" + "=" * 42 + "+")
    print("| XGBoost RUL Hibrido (Forzy+CWRU)        |")
    print(
        f"|   R2: {hybrid_metrics['r2']:.4f} | MAE: {hybrid_metrics['mae']:.2f} h             |"
    )
    print(f"|   Correlacao temporal: {hybrid_corr:+.4f}         |")
    print("+" + "=" * 42 + "+")
    print(f"| STATUS GERAL: {status:<26}|")
    print("| (todos F1>0.80, R2CMAPSS>0.85,         |")
    print("|  R2hibrido>0.70, corr<-0.30)           |")
    print("+" + "=" * 42 + "+")


if __name__ == "__main__":
    generate_report()
