"""
Pipeline único de retreinamento — modelos CWRU, CMAPSS e Forzy.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_module.features.feature_engineering import build_features, load_raw_csv
from ml_module.models.anomaly.lstm_autoencoder import (
    build_model,
    compute_threshold as lstm_compute_threshold,
    create_sequences,
    prepare_data as lstm_prepare_data,
    predict as lstm_predict,
    save_model as lstm_save_model,
    train as lstm_train,
)
from ml_module.models.anomaly.lstm_autoencoder_cwru import train_on_cwru as train_lstm_cwru
from ml_module.models.baseline.isolation_forest import (
    compute_threshold as if_compute_threshold,
    prepare_data as if_prepare_data,
    predict as if_predict,
    save_model as if_save_model,
    train as if_train,
)
from ml_module.models.baseline.isolation_forest_cwru import train_on_cwru
from ml_module.models.rul.xgboost_rul import (
    prepare_data as rul_prepare_data,
    save_model as rul_save_model,
    train as rul_train,
)
from ml_module.models.rul.xgboost_rul_cmapss import load_cmapss, train_on_cmapss

CWRU_PARQUET = PROJECT_ROOT / "ml_module/data/processed/cwru_processed.parquet"
CMAPSS_TRAIN = PROJECT_ROOT / "ml_module/data/processed/cmapss_train.parquet"
CMAPSS_TEST = PROJECT_ROOT / "ml_module/data/processed/cmapss_test.parquet"
FORZY_CSV = PROJECT_ROOT / "History_32026-05-19T11-46-10-920.csv"

IF_CWRU_DIR = PROJECT_ROOT / "ml_module/models/baseline/saved_cwru"
LSTM_CWRU_DIR = PROJECT_ROOT / "ml_module/models/anomaly/saved_cwru"
RUL_CMAPSS_DIR = PROJECT_ROOT / "ml_module/models/rul/saved_cmapss"
IF_FORZY_DIR = PROJECT_ROOT / "ml_module/models/baseline/saved"
LSTM_FORZY_DIR = PROJECT_ROOT / "ml_module/models/anomaly/saved"
RUL_FORZY_DIR = PROJECT_ROOT / "ml_module/models/rul/saved"

WINDOW_SIZE = 60


def _train_forzy_models() -> dict:
    """Retreina IF, LSTM e XGBoost RUL com dados da Forzy."""
    raw_df = load_raw_csv(FORZY_CSV)
    features_df = build_features(raw_df)

    # --- Isolation Forest Forzy ---
    X_if, if_scaler = if_prepare_data(features_df)
    if_model = if_train(X_if)
    if_threshold = if_compute_threshold(if_model, X_if)
    if_results = if_predict(if_model, X_if, if_threshold)
    if_save_model(if_model, if_scaler, if_threshold, IF_FORZY_DIR)

    # --- LSTM Forzy ---
    X_lstm, lstm_scaler = lstm_prepare_data(features_df)
    X_seq = create_sequences(X_lstm, window_size=WINDOW_SIZE)
    lstm_model = build_model(WINDOW_SIZE, X_seq.shape[2])
    lstm_history = lstm_train(lstm_model, X_seq)
    lstm_threshold, _ = lstm_compute_threshold(lstm_model, X_seq)
    lstm_results = lstm_predict(
        lstm_model,
        X_seq,
        lstm_threshold,
        features_df["timestamp"].iloc[WINDOW_SIZE - 1 :].reset_index(drop=True),
    )
    lstm_save_model(lstm_model, lstm_scaler, lstm_threshold, LSTM_FORZY_DIR)

    # --- XGBoost RUL Forzy ---
    X_rul, y_rul, rul_scaler, feature_names = rul_prepare_data(features_df)
    rul_model, _X_test, _y_test, rul_metrics = rul_train(X_rul, y_rul)
    rul_save_model(rul_model, rul_scaler, rul_metrics, RUL_FORZY_DIR)

    return {
        "isolation_forest": {
            "anomaly_rate": if_results["anomaly_rate"],
            "n_anomalies": if_results["n_anomalies"],
            "threshold": if_threshold,
        },
        "lstm": {
            "anomaly_rate": lstm_results["anomaly_rate"],
            "n_anomalies": lstm_results["n_anomalies"],
            "epochs": len(lstm_history.history["loss"]),
            "threshold": lstm_threshold,
        },
        "xgboost_rul": rul_metrics,
    }


def run_pipeline(skip_forzy: bool = False) -> dict:
    """Executa retreinamento sequencial de todos os modelos."""
    results: dict = {}

    print("\n[Etapa 1/4] Isolation Forest — CWRU")
    if_cwru = train_on_cwru(CWRU_PARQUET, IF_CWRU_DIR)
    print(f"  F1 obtido: {if_cwru['f1']:.4f}")
    results["isolation_forest_cwru"] = {
        "precision": if_cwru["precision"],
        "recall": if_cwru["recall"],
        "f1": if_cwru["f1"],
        "threshold": if_cwru["threshold"],
    }

    print("\n[Etapa 2/4] LSTM Autoencoder — CWRU")
    lstm_cwru = train_lstm_cwru(CWRU_PARQUET, LSTM_CWRU_DIR)
    print(f"  F1 obtido: {lstm_cwru['f1']:.4f}")
    results["lstm_cwru"] = {
        "precision": lstm_cwru["precision"],
        "recall": lstm_cwru["recall"],
        "f1": lstm_cwru["f1"],
        "threshold": lstm_cwru["threshold"],
        "epochs": lstm_cwru["epochs_trained"],
    }

    print("\n[Etapa 3/4] XGBoost RUL — CMAPSS")
    X_train, y_train, X_test, y_test, feature_names = load_cmapss(
        CMAPSS_TRAIN, CMAPSS_TEST
    )
    cmapss = train_on_cmapss(
        X_train, y_train, X_test, y_test, feature_names, RUL_CMAPSS_DIR
    )
    print(f"  R² obtido: {cmapss['r2']:.4f}")
    results["xgboost_cmapss"] = {
        "mae": cmapss["mae"],
        "rmse": cmapss["rmse"],
        "r2": cmapss["r2"],
        "nasa_score": cmapss["nasa_score"],
        "n_estimators_used": cmapss["n_estimators_used"],
    }

    if not skip_forzy:
        print("\n[Etapa 4/4] Retreino Forzy (IF + LSTM + XGBoost)")
        results["forzy"] = _train_forzy_models()
        print(f"  IF anomalias: {results['forzy']['isolation_forest']['n_anomalies']}")
        print(f"  LSTM anomalias: {results['forzy']['lstm']['n_anomalies']}")
        print(f"  RUL R²: {results['forzy']['xgboost_rul']['r2']:.4f}")
    else:
        print("\n[Etapa 4/4] Retreino Forzy — PULADO (--skip-forzy)")
        results["forzy"] = {"skipped": True}

    return results


def _print_summary(results: dict) -> None:
    """Imprime resumo final de todas as métricas."""
    print("\n" + "=" * 60)
    print("RESUMO FINAL DO PIPELINE")
    print("=" * 60)

    if_cwru = results["isolation_forest_cwru"]
    print(
        f"IF CWRU      — F1: {if_cwru['f1']:.4f} | "
        f"Precision: {if_cwru['precision']:.4f} | Recall: {if_cwru['recall']:.4f}"
    )

    lstm = results["lstm_cwru"]
    print(
        f"LSTM CWRU    — F1: {lstm['f1']:.4f} | "
        f"Precision: {lstm['precision']:.4f} | Recall: {lstm['recall']:.4f}"
    )

    cmapss = results["xgboost_cmapss"]
    print(
        f"RUL CMAPSS   — R²: {cmapss['r2']:.4f} | "
        f"MAE: {cmapss['mae']:.2f} | NASA: {cmapss['nasa_score']:.2f}"
    )

    forzy = results.get("forzy", {})
    if forzy.get("skipped"):
        print("Forzy        — pulado")
    elif forzy:
        print(
            f"Forzy IF     — anomalias: {forzy['isolation_forest']['n_anomalies']} | "
            f"taxa: {forzy['isolation_forest']['anomaly_rate']:.2f}%"
        )
        print(
            f"Forzy LSTM   — anomalias: {forzy['lstm']['n_anomalies']} | "
            f"épocas: {forzy['lstm']['epochs']}"
        )
        print(
            f"Forzy RUL    — R²: {forzy['xgboost_rul']['r2']:.4f} | "
            f"MAE: {forzy['xgboost_rul']['mae']:.2f}"
        )

    print("=" * 60)


if __name__ == "__main__":
    skip_forzy = "--skip-forzy" in sys.argv

    print("=" * 60)
    print("FORZY DIGITAL TWIN — PIPELINE DE TREINAMENTO")
    print("=" * 60)
    if skip_forzy:
        print("Modo: --skip-forzy (etapa Forzy desabilitada)")

    start = time.time()
    pipeline_results = run_pipeline(skip_forzy=skip_forzy)
    elapsed = time.time() - start

    _print_summary(pipeline_results)
    print(f"\nTempo total de execução: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
