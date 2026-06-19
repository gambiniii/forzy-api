"""
Adaptador de transferência: mapeia features Forzy para o espaço CMAPSS
e prediz RUL usando o modelo XGBoost treinado no NASA CMAPSS FD001.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_module.features.feature_engineering import build_features, load_raw_csv
from ml_module.models.rul.xgboost_rul import VIDA_UTIL_MAX, load_model as load_rul_model
from ml_module.models.rul.xgboost_rul_cmapss import load_cmapss

SENSOR_ANALOG_SPECS: dict[str, tuple[str, bool]] = {
    "sensor_4_analog": ("port1_velocidade_rolling_mean_50", False),
    "sensor_11_analog": ("temperatura_media_global", True),
    "sensor_12_analog": ("port2_velocidade_rolling_mean_50", False),
    "sensor_9_analog": ("aceleracao_media_global", True),
    "sensor_7_analog": ("port1_velocidade_desvio_iso", True),
    "sensor_14_analog": ("port2_velocidade_desvio_iso", True),
    "sensor_15_analog": ("delta_temperatura_entre_sensores", True),
    "sensor_2_analog": ("velocidade_media_global", True),
    "sensor_3_analog": ("port1_aceleracao_rolling_mean_50", True),
}

NEUTRAL_FILL = 0.5
ROLLING_WINDOW = 5


def _min_max_normalize(series: pd.Series) -> pd.Series:
    """Normaliza série para [0, 1]; constante retorna 0.5."""
    min_val = series.min()
    max_val = series.max()
    if max_val == min_val:
        return pd.Series(NEUTRAL_FILL, index=series.index)
    return (series - min_val) / (max_val - min_val)


def _risk_level(rul_hours: float) -> str:
    """Classifica nível de risco com base nas horas de RUL restantes."""
    if rul_hours > 5000:
        return "low"
    if rul_hours > 1000:
        return "medium"
    if rul_hours > 100:
        return "high"
    return "critical"


def extract_forzy_health_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """Extrai indicadores de saúde Forzy análogos aos sensores CMAPSS."""
    health = pd.DataFrame(index=features_df.index)

    for analog_name, (source_col, normalize) in SENSOR_ANALOG_SPECS.items():
        values = features_df[source_col]
        health[analog_name] = _min_max_normalize(values) if normalize else values

    elapsed_max = features_df["elapsed_seconds"].max()
    if elapsed_max > 0:
        health["cycle_analog"] = features_df["elapsed_seconds"] / elapsed_max
    else:
        health["cycle_analog"] = 0.0

    analog_cols = list(SENSOR_ANALOG_SPECS.keys())
    for col in analog_cols:
        roll = health[col].rolling(window=ROLLING_WINDOW, min_periods=1)
        health[f"{col}_roll_mean_{ROLLING_WINDOW}"] = roll.mean()
        health[f"{col}_roll_std_{ROLLING_WINDOW}"] = roll.std().fillna(0)
        health[f"{col}_delta"] = health[col].diff().fillna(0)

    return health


def _forzy_column_for_cmapss(cmapss_feature: str, forzy_columns: list[str]) -> str | None:
    """Resolve nome da coluna Forzy correspondente a uma feature CMAPSS."""
    if cmapss_feature == "cycle_norm":
        return "cycle_analog" if "cycle_analog" in forzy_columns else None

    if cmapss_feature.startswith("op_setting"):
        return None

    for base_analog in SENSOR_ANALOG_SPECS:
        cmapss_base = base_analog.replace("_analog", "")
        if cmapss_feature == cmapss_base:
            return base_analog if base_analog in forzy_columns else None
        prefix = f"{cmapss_base}_"
        if cmapss_feature.startswith(prefix):
            suffix = cmapss_feature[len(prefix) :]
            candidate = f"{base_analog}_{suffix}"
            if candidate in forzy_columns:
                return candidate

    for col in forzy_columns:
        if cmapss_feature in col:
            return col

    return None


def align_to_cmapss_features(
    forzy_health_df: pd.DataFrame,
    cmapss_feature_names: list[str],
) -> tuple[np.ndarray, dict]:
    """
    Alinha features Forzy ao espaço CMAPSS.

    Retorna array (n_samples, n_features) e estatísticas de mapeamento.
    """
    n_samples = len(forzy_health_df)
    n_features = len(cmapss_feature_names)
    aligned = np.full((n_samples, n_features), NEUTRAL_FILL, dtype=np.float64)

    forzy_columns = list(forzy_health_df.columns)
    mapped = 0
    filled_neutral = 0
    mapping_detail: dict[str, str | None] = {}

    for idx, cmapss_feat in enumerate(cmapss_feature_names):
        forzy_col = _forzy_column_for_cmapss(cmapss_feat, forzy_columns)
        mapping_detail[cmapss_feat] = forzy_col

        if forzy_col is not None:
            aligned[:, idx] = forzy_health_df[forzy_col].to_numpy()
            mapped += 1
        else:
            filled_neutral += 1

    stats = {
        "mapped": mapped,
        "filled_neutral": filled_neutral,
        "total": n_features,
        "mapping_detail": mapping_detail,
    }
    return aligned, stats


def predict_rul_forzy(
    features_df: pd.DataFrame,
    cmapss_model_dir: Path,
) -> pd.DataFrame:
    """Prediz RUL Forzy usando modelo CMAPSS via adaptador de features."""
    health_df = extract_forzy_health_features(features_df)

    cmapss_train = PROJECT_ROOT / "ml_module/data/processed/cmapss_train.parquet"
    cmapss_test = PROJECT_ROOT / "ml_module/data/processed/cmapss_test.parquet"
    _, _, _, _, cmapss_feature_names = load_cmapss(cmapss_train, cmapss_test)

    model, scaler, _metrics = load_rul_model(cmapss_model_dir)
    X_aligned, _ = align_to_cmapss_features(health_df, cmapss_feature_names)

    if scaler is not None:
        X_input = scaler.transform(X_aligned)
    else:
        X_input = X_aligned

    rul_cycles = model.predict(X_input)
    rul_hours = np.clip(rul_cycles * 1.0, 0, VIDA_UTIL_MAX)
    maintenance_window_days = rul_hours / 24.0
    health_score = np.clip(rul_hours / VIDA_UTIL_MAX, 0, 1)

    result = pd.DataFrame(
        {
            "timestamp": features_df["timestamp"].values,
            "rul_cycles": rul_cycles,
            "rul_hours": rul_hours,
            "maintenance_window_days": maintenance_window_days,
            "risk_level": [_risk_level(float(h)) for h in rul_hours],
            "health_score": health_score,
        }
    )
    return result


def evaluate_adapter(
    features_df: pd.DataFrame,
    cmapss_model_dir: Path,
) -> dict:
    """Avalia predições do adaptador e consistência temporal."""
    predictions = predict_rul_forzy(features_df, cmapss_model_dir)
    rul_hours = predictions["rul_hours"].to_numpy()

    elapsed = features_df["elapsed_seconds"].to_numpy()
    if len(elapsed) > 1 and np.std(elapsed) > 0 and np.std(rul_hours) > 0:
        temporal_correlation = float(np.corrcoef(elapsed, rul_hours)[0, 1])
    else:
        temporal_correlation = 0.0

    return {
        "min": float(np.min(rul_hours)),
        "max": float(np.max(rul_hours)),
        "mean": float(np.mean(rul_hours)),
        "std": float(np.std(rul_hours)),
        "p25": float(np.percentile(rul_hours, 25)),
        "p50": float(np.percentile(rul_hours, 50)),
        "p75": float(np.percentile(rul_hours, 75)),
        "temporal_correlation": temporal_correlation,
        "predictions": predictions,
    }


if __name__ == "__main__":
    FORZY_CSV = PROJECT_ROOT / "History_32026-05-19T11-46-10-920.csv"
    CMAPSS_MODEL = PROJECT_ROOT / "ml_module/models/rul/saved_cmapss"
    CMAPSS_TRAIN = PROJECT_ROOT / "ml_module/data/processed/cmapss_train.parquet"
    CMAPSS_TEST = PROJECT_ROOT / "ml_module/data/processed/cmapss_test.parquet"

    raw_df = load_raw_csv(FORZY_CSV)
    features_df = build_features(raw_df)

    health_df = extract_forzy_health_features(features_df)
    print(f"Shape features análogas: {health_df.shape}")

    _, _, _, _, cmapss_feature_names = load_cmapss(CMAPSS_TRAIN, CMAPSS_TEST)
    _, align_stats = align_to_cmapss_features(health_df, cmapss_feature_names)
    print(
        f"Features CMAPSS mapeadas: {align_stats['mapped']}/{align_stats['total']} "
        f"| preenchidas com {NEUTRAL_FILL}: {align_stats['filled_neutral']}"
    )

    predictions = predict_rul_forzy(features_df, CMAPSS_MODEL)
    eval_result = evaluate_adapter(features_df, CMAPSS_MODEL)

    print("\nDistribuição RUL predito (horas):")
    print(f"  min:  {eval_result['min']:.2f}")
    print(f"  max:  {eval_result['max']:.2f}")
    print(f"  mean: {eval_result['mean']:.2f}")
    print(f"  std:  {eval_result['std']:.2f}")
    print(f"  P25:  {eval_result['p25']:.2f}")
    print(f"  P50:  {eval_result['p50']:.2f}")
    print(f"  P75:  {eval_result['p75']:.2f}")

    print(f"\nCorrelação temporal (elapsed_seconds vs rul_hours): {eval_result['temporal_correlation']:.4f}")

    print("\nTop 5 registros mais críticos (menor RUL):")
    critical = predictions.nsmallest(5, "rul_hours")
    for _, row in critical.iterrows():
        print(
            f"  {row['timestamp']} | RUL: {row['rul_hours']:.2f}h | "
            f"risco: {row['risk_level']} | health: {row['health_score']:.4f}"
        )

    if eval_result["temporal_correlation"] < -0.3:
        print("\nConclusão: Adaptador funcionando")
    else:
        print("\nConclusão: Ajuste necessário")
