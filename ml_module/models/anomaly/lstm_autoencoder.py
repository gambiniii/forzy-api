"""
Modelo LSTM Autoencoder para detecção de anomalias em séries temporais.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

try:
    import tensorflow as tf
except ImportError:
    import keras

    keras.utils.set_random_seed(42)

    class _KerasAPI:
        Model = keras.Model
        models = keras.models
        callbacks = keras.callbacks

        class layers:
            Input = keras.layers.Input
            LSTM = keras.layers.LSTM
            Dropout = keras.layers.Dropout
            Dense = keras.layers.Dense
            RepeatVector = keras.layers.RepeatVector
            TimeDistributed = keras.layers.TimeDistributed

    class _TF:
        keras = _KerasAPI()

        class random:
            @staticmethod
            def set_seed(seed: int) -> None:
                keras.utils.set_random_seed(seed)

    tf = _TF()

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_module.features.feature_engineering import build_features, load_raw_csv

tf.random.set_seed(42)
np.random.seed(42)

MODEL_FILENAME = "lstm_autoencoder.keras"
SCALER_FILENAME = "lstm_scaler.joblib"
THRESHOLD_FILENAME = "lstm_threshold.json"


def create_sequences(X: np.ndarray, window_size: int = 60) -> np.ndarray:
    """Converte array 2D em janelas temporais 3D."""
    n_samples, n_features = X.shape
    n_sequences = n_samples - window_size + 1

    sequences = np.zeros((n_sequences, window_size, n_features), dtype=X.dtype)
    for i in range(n_sequences):
        sequences[i] = X[i : i + window_size]

    return sequences


def build_model(window_size: int, n_features: int) -> tf.keras.Model:
    """Constrói e compila o LSTM Autoencoder."""
    inputs = tf.keras.layers.Input(shape=(window_size, n_features))

    encoded = tf.keras.layers.LSTM(64, return_sequences=True, activation="tanh")(inputs)
    encoded = tf.keras.layers.Dropout(0.2)(encoded)
    encoded = tf.keras.layers.LSTM(32, return_sequences=False, activation="tanh")(encoded)
    encoded = tf.keras.layers.Dense(16, activation="relu")(encoded)

    decoded = tf.keras.layers.RepeatVector(window_size)(encoded)
    decoded = tf.keras.layers.LSTM(32, return_sequences=True, activation="tanh")(decoded)
    decoded = tf.keras.layers.Dropout(0.2)(decoded)
    decoded = tf.keras.layers.LSTM(64, return_sequences=True, activation="tanh")(decoded)
    outputs = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(n_features)
    )(decoded)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="mse")
    return model


def prepare_data(features_df: pd.DataFrame) -> tuple[np.ndarray, MinMaxScaler]:
    """Remove timestamp, aplica MinMaxScaler (0-1) e retorna array + scaler."""
    feature_cols = [col for col in features_df.columns if col != "timestamp"]
    X = features_df[feature_cols].values

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, scaler


def _reconstruction_errors(model: tf.keras.Model, X_seq: np.ndarray) -> np.ndarray:
    """Calcula MSE de reconstrução por amostra."""
    reconstructed = model.predict(X_seq, verbose=0)
    return np.mean(np.square(X_seq - reconstructed), axis=(1, 2))


def train(
    model: tf.keras.Model,
    X_seq: np.ndarray,
    epochs: int = 50,
    batch_size: int = 32,
) -> tf.keras.callbacks.History:
    """Treina o autoencoder (input = target) com early stopping."""
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )

    history = model.fit(
        X_seq,
        X_seq,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=[early_stopping],
        verbose=1,
    )
    return history


def compute_threshold(
    model: tf.keras.Model,
    X_seq: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Define threshold como mean + 2*std dos erros de reconstrução."""
    reconstruction_errors = _reconstruction_errors(model, X_seq)
    threshold = float(reconstruction_errors.mean() + 2 * reconstruction_errors.std())
    return threshold, reconstruction_errors


def _map_severity(errors: np.ndarray, threshold: float) -> np.ndarray:
    """Mapeia erros de reconstrução para categorias de severidade."""
    severity = np.full(len(errors), "critical", dtype=object)
    severity[errors < threshold * 4.0] = "high"
    severity[errors < threshold * 3.0] = "medium"
    severity[errors < threshold * 2.0] = "low"
    severity[errors < threshold * 1.5] = "normal"
    return severity


def predict(
    model: tf.keras.Model,
    X_seq: np.ndarray,
    threshold: float,
    timestamps: pd.Series,
) -> dict:
    """Gera erros, labels, severidade e métricas de anomalia."""
    errors = _reconstruction_errors(model, X_seq)
    labels = np.where(errors > threshold, -1, 1).astype(int)
    n_anomalies = int((labels == -1).sum())
    anomaly_rate = float(n_anomalies / len(labels) * 100)
    severity = _map_severity(errors, threshold)

    return {
        "errors": errors,
        "labels": labels,
        "anomaly_rate": anomaly_rate,
        "n_anomalies": n_anomalies,
        "threshold": threshold,
        "severity": severity,
        "timestamps": timestamps.reset_index(drop=True),
    }


def save_model(
    model: tf.keras.Model,
    scaler: MinMaxScaler,
    threshold: float,
    output_dir: Path,
) -> None:
    """Persiste modelo, scaler e threshold."""
    output_dir.mkdir(parents=True, exist_ok=True)

    model.save(output_dir / MODEL_FILENAME)
    joblib.dump(scaler, output_dir / SCALER_FILENAME)

    with open(output_dir / THRESHOLD_FILENAME, "w", encoding="utf-8") as f:
        json.dump({"threshold": threshold}, f, indent=2)


def load_model(model_dir: Path) -> tuple[tf.keras.Model, MinMaxScaler, float]:
    """Carrega modelo, scaler e threshold."""
    model = tf.keras.models.load_model(model_dir / MODEL_FILENAME)
    scaler = joblib.load(model_dir / SCALER_FILENAME)

    with open(model_dir / THRESHOLD_FILENAME, encoding="utf-8") as f:
        threshold = float(json.load(f)["threshold"])

    return model, scaler, threshold


if __name__ == "__main__":
    CSV_PATH = PROJECT_ROOT / "History_32026-05-19T11-46-10-920.csv"
    OUTPUT_DIR = Path(__file__).resolve().parent / "saved"
    WINDOW_SIZE = 60

    raw_df = load_raw_csv(CSV_PATH)
    features_df = build_features(raw_df)
    timestamps = features_df["timestamp"].iloc[WINDOW_SIZE - 1 :].reset_index(drop=True)

    X_scaled, scaler = prepare_data(features_df)
    X_seq = create_sequences(X_scaled, window_size=WINDOW_SIZE)

    n_features = X_seq.shape[2]
    model = build_model(WINDOW_SIZE, n_features)
    history = train(model, X_seq)
    threshold, _ = compute_threshold(model, X_seq)
    results = predict(model, X_seq, threshold, timestamps)

    train_loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    print(f"Shape das sequências criadas: {X_seq.shape}")
    print(f"Épocas treinadas: {len(train_loss)}")
    print(f"Loss final de treino: {train_loss[-1]:.6f}")
    print(f"Loss final de validação: {val_loss[-1]:.6f}")
    print(f"Threshold calculado: {threshold:.6f}")
    print(f"Taxa de anomalias detectadas: {results['anomaly_rate']:.2f}%")
    print(f"Contagem de anomalias: {results['n_anomalies']}")

    print("\nDistribuição de severidade:")
    severity_counts = pd.Series(results["severity"]).value_counts()
    for category in ["normal", "low", "medium", "high", "critical"]:
        count = int(severity_counts.get(category, 0))
        print(f"  {category}: {count}")

    print("\nTop 5 anomalias mais críticas:")
    top5_indices = np.argsort(results["errors"])[-5:][::-1]
    for rank, idx in enumerate(top5_indices, start=1):
        print(
            f"  {rank}. índice={idx} | "
            f"timestamp={results['timestamps'].iloc[idx]} | "
            f"erro={results['errors'][idx]:.6f} | "
            f"severidade={results['severity'][idx]}"
        )

    save_model(model, scaler, threshold, OUTPUT_DIR)
    print(f"\nModelo salvo em: {OUTPUT_DIR.resolve()}")
    print(f"  - {MODEL_FILENAME}")
    print(f"  - {SCALER_FILENAME}")
    print(f"  - {THRESHOLD_FILENAME}")
