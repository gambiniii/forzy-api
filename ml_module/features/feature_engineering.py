"""
Engenharia de features para dados de vibração — motor WEG W22 3cv.

V2: apenas velocidade e aceleração das 2 portas. Temperatura e features cruzadas
entre portas foram removidas (temperatura introduz deriva por aquecimento; a porta 2
tem correlação ~0.998 com a porta 1, tornando médias/diferenças cruzadas redundantes).
"""

from pathlib import Path

import pandas as pd

ISO_VELOCIDADE_LIMITE_BOM = 2.8

ROLLING_WINDOWS = {
    50: "30s",
    200: "2min",
    500: "10min",
}

SENSOR_PREFIXES = {
    "port1": {
        "velocidade": "1.1",
        "aceleracao": "1.2",
    },
    "port2": {
        "velocidade": "2.1",
        "aceleracao": "2.2",
    },
}

SIGNAL_KEYWORDS = {
    "velocidade": "Velocidade",
    "aceleracao": "Acelera",
}


def _resolve_sensor_columns(df: pd.DataFrame) -> dict[str, dict[str, str]]:
    """Mapeia portas para nomes reais das colunas no dataframe."""
    columns = list(df.columns)
    resolved: dict[str, dict[str, str]] = {}

    for port_name, signals in SENSOR_PREFIXES.items():
        resolved[port_name] = {}
        for signal, prefix in signals.items():
            keyword = SIGNAL_KEYWORDS[signal]
            match = next(
                (col for col in columns if col.startswith(prefix) and keyword in col),
                None,
            )
            if match is None:
                raise KeyError(
                    f"Coluna não encontrada para {port_name}.{signal} "
                    f"(prefixo '{prefix}', palavra-chave '{keyword}')"
                )
            resolved[port_name][signal] = match

    return resolved


def _add_rolling_features(
    features: pd.DataFrame,
    source: pd.Series,
    prefix: str,
    signal: str,
    window: int,
    include_max: bool,
) -> None:
    """Adiciona estatísticas rolling para um sinal de um sensor."""
    roll = source.rolling(window=window, min_periods=1)
    features[f"{prefix}_{signal}_rolling_mean_{window}"] = roll.mean()
    features[f"{prefix}_{signal}_rolling_std_{window}"] = roll.std()
    if include_max:
        features[f"{prefix}_{signal}_rolling_max_{window}"] = roll.max()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recebe o dataframe bruto do CSV e retorna apenas timestamp + features calculadas.
    """
    work = df.copy()
    sensor_columns = _resolve_sensor_columns(work)

    work["timestamp"] = pd.to_datetime(work["timestamp"], errors="coerce", format="ISO8601")
    work = work.sort_values("timestamp").reset_index(drop=True)

    for port_cols in sensor_columns.values():
        for col in port_cols.values():
            work[col] = pd.to_numeric(work[col], errors="coerce")

    features = pd.DataFrame({"timestamp": work["timestamp"]})

    timestamps = work["timestamp"]
    features["elapsed_seconds"] = (timestamps - timestamps.min()).dt.total_seconds()

    for port_name, cols in sensor_columns.items():
        vel = work[cols["velocidade"]]
        acc = work[cols["aceleracao"]]

        for window in ROLLING_WINDOWS:
            _add_rolling_features(features, vel, port_name, "velocidade", window, include_max=True)
            _add_rolling_features(features, acc, port_name, "aceleracao", window, include_max=True)

        features[f"{port_name}_delta_velocidade"] = vel.diff()
        features[f"{port_name}_delta_aceleracao"] = acc.diff()

        features[f"{port_name}_velocidade_desvio_iso"] = (vel - ISO_VELOCIDADE_LIMITE_BOM) / ISO_VELOCIDADE_LIMITE_BOM

    feature_cols = [col for col in features.columns if col != "timestamp"]
    features[feature_cols] = features[feature_cols].ffill().bfill()

    return features


def load_raw_csv(filepath: Path) -> pd.DataFrame:
    """Carrega o CSV bruto com as colunas necessárias para feature engineering."""
    df = pd.read_csv(
        filepath,
        sep=";",
        skiprows=[0, 2],
        header=0,
        dtype={0: str},
        encoding="cp1252",
        low_memory=False,
    )

    first_col = df.columns[0]
    if pd.isna(first_col) or str(first_col).strip() == "" or str(first_col).startswith("Unnamed"):
        df = df.rename(columns={first_col: "timestamp"})

    sensor_columns = _resolve_sensor_columns(df)
    keep_cols = ["timestamp"] + [cols[signal] for cols in sensor_columns.values() for signal in cols]
    return df[keep_cols]


if __name__ == "__main__":
    csv_path = Path(__file__).resolve().parents[2] / "History_32026-05-19T11-46-10-920.csv"

    raw_df = load_raw_csv(csv_path)
    features_df = build_features(raw_df)

    print(f"Shape: {features_df.shape[0]} linhas x {features_df.shape[1]} colunas")
    print("\nPrimeiras 3 linhas:")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(features_df.head(3).to_string())

    feature_names = [col for col in features_df.columns if col != "timestamp"]
    print(f"\nTotal de features geradas: {len(feature_names)}")
    print("\nFeatures geradas:")
    for i, name in enumerate(feature_names, start=1):
        print(f"  {i}. {name}")
