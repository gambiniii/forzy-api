"""
XGBoost RUL supervisionado no NASA CMAPSS FD001.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_module.models.rul.xgboost_rul import load_model, save_model

RUL_CLIP_MAX = 125
META_COLS = ["unit_id", "cycle", "RUL"]


def _engineer_features(
    df: pd.DataFrame,
    base_features: list[str],
    max_cycle_ref: float,
) -> pd.DataFrame:
    """Adiciona rolling stats por unidade e ciclo normalizado."""
    engineered = df.copy()

    # Referência global do treino evita cycle_norm=1 em todos os últimos ciclos do teste
    engineered["cycle_norm"] = engineered["cycle"] / max_cycle_ref

    for feature in base_features:
        grouped = engineered.groupby("unit_id")[feature]
        engineered[f"{feature}_roll_mean_5"] = grouped.transform(
            lambda x: x.rolling(5, min_periods=1).mean()
        )
        engineered[f"{feature}_roll_std_5"] = grouped.transform(
            lambda x: x.rolling(5, min_periods=1).std().fillna(0)
        )
        engineered[f"{feature}_delta"] = engineered[feature] - grouped.transform("first")

    return engineered


def load_cmapss(
    train_path: Path,
    test_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Carrega CMAPSS, aplica feature engineering e clipa RUL em 125."""
    df_train = pd.read_parquet(train_path)
    df_test = pd.read_parquet(test_path)

    base_features = [col for col in df_train.columns if col not in META_COLS]

    max_cycle_ref = float(df_train["cycle"].max())
    df_train = _engineer_features(df_train, base_features, max_cycle_ref)
    df_test = _engineer_features(df_test, base_features, max_cycle_ref)

    feature_names = [col for col in df_train.columns if col not in META_COLS]

    X_train = df_train[feature_names].to_numpy()
    y_train = np.clip(df_train["RUL"].to_numpy(), 0, RUL_CLIP_MAX)

    X_test = df_test[feature_names].to_numpy()
    y_test = np.clip(df_test["RUL"].to_numpy(), 0, RUL_CLIP_MAX)

    return X_train, y_train, X_test, y_test, feature_names


def _nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcula NASA Score (menor = melhor), penalizando predições otimistas."""
    diff = y_pred - y_true
    score = 0.0
    for d in diff:
        if d < 0:
            score += np.exp(-d / 13.0) - 1.0
        else:
            score += np.exp(d / 10.0) - 1.0
    return float(score)


def _eval_mask(y: np.ndarray) -> np.ndarray:
    """Máscara de amostras com RUL válido (último ciclo no conjunto de teste)."""
    return ~np.isnan(y)


def train_on_cmapss(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    output_dir: Path,
) -> dict:
    """Treina XGBRegressor com early stopping e avalia no conjunto de teste."""
    eval_mask = _eval_mask(y_test)
    X_test_eval = X_test[eval_mask]
    y_test_eval = y_test[eval_mask]

    model = xgb.XGBRegressor(
        n_estimators=2000,
        max_depth=8,
        learning_rate=0.005,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=0.1,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
        early_stopping_rounds=75,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test_eval, y_test_eval)],
        verbose=100,
    )

    y_pred = model.predict(X_test_eval)

    metrics = {
        "mae": float(mean_absolute_error(y_test_eval, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test_eval, y_pred))),
        "r2": float(r2_score(y_test_eval, y_pred)),
        "nasa_score": _nasa_score(y_test_eval, y_pred),
        "n_estimators_used": int(model.best_iteration + 1) if hasattr(model, "best_iteration") else model.n_estimators,
    }

    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(
        ascending=False
    )
    metrics["top_features"] = importances.head(10).to_dict()

    model.r2_score_ = metrics["r2"]
    save_model(model, None, metrics, output_dir)

    return {
        **metrics,
        "model": model,
        "importances": importances,
        "y_test_eval": y_test_eval,
        "y_pred": y_pred,
    }


if __name__ == "__main__":
    TRAIN_PATH = PROJECT_ROOT / "ml_module/data/processed/cmapss_train.parquet"
    TEST_PATH = PROJECT_ROOT / "ml_module/data/processed/cmapss_test.parquet"
    OUTPUT_DIR = Path(__file__).resolve().parent / "saved_cmapss"

    print("=" * 60)
    print("XGBoost RUL — NASA CMAPSS FD001")
    print("=" * 60)

    X_train, y_train, X_test, y_test, feature_names = load_cmapss(TRAIN_PATH, TEST_PATH)

    print(f"\nShape de X_train: {X_train.shape}")
    print(f"Shape de X_test:  {X_test.shape}")
    print(
        f"Distribuição de y_train (após clip 0-{RUL_CLIP_MAX}): "
        f"min={y_train.min():.1f}, max={y_train.max():.1f}, mean={y_train.mean():.1f}"
    )

    print("\n--- Treinamento (early stopping) ---")
    results = train_on_cmapss(
        X_train, y_train, X_test, y_test, feature_names, OUTPUT_DIR
    )

    print("\n" + "=" * 60)
    print("MÉTRICAS FINAIS (teste — 100 últimos ciclos)")
    print("=" * 60)
    print(f"  MAE:         {results['mae']:.4f}")
    print(f"  RMSE:        {results['rmse']:.4f}")
    print(f"  R²:          {results['r2']:.4f}")
    print(f"  NASA Score:  {results['nasa_score']:.4f}")
    print(f"  Estimadores: {results['n_estimators_used']}")

    print("\nTop 10 features por importância:")
    for rank, (name, importance) in enumerate(results["importances"].head(10).items(), start=1):
        print(f"  {rank}. {name}: {importance:.4f}")

    conclusion = "Modelo apto para produção" if results["r2"] > 0.85 else "Requer ajuste"
    print(f"\nConclusão: {conclusion}")
    print(f"\nModelo salvo em: {OUTPUT_DIR.resolve()}")
    print("=" * 60)
