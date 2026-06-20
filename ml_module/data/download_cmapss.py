"""
Download e processamento do NASA CMAPSS FD001 (Turbofan Engine Degradation).
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import MinMaxScaler

NASA_ZIP_URL = "https://data.nasa.gov/download/ffnz-7ejw/application%2Fzip"

ALTERNATIVE_URLS = {
    "train_FD001.txt": "https://raw.githubusercontent.com/schwenker/cmapss-data/master/CMAPSSData/train_FD001.txt",
    "test_FD001.txt": "https://raw.githubusercontent.com/schwenker/cmapss-data/master/CMAPSSData/test_FD001.txt",
    "RUL_FD001.txt": "https://raw.githubusercontent.com/schwenker/cmapss-data/master/CMAPSSData/RUL_FD001.txt",
}

FALLBACK_URLS = {
    "train_FD001.txt": "https://raw.githubusercontent.com/LahiruJayasinghe/RUL-Net/master/CMAPSSData/train_FD001.txt",
    "test_FD001.txt": "https://raw.githubusercontent.com/LahiruJayasinghe/RUL-Net/master/CMAPSSData/test_FD001.txt",
    "RUL_FD001.txt": "https://raw.githubusercontent.com/LahiruJayasinghe/RUL-Net/master/CMAPSSData/RUL_FD001.txt",
}

CMAPSS_COLUMNS = (
    ["unit_id", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)

FD001_FILES = ["train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt"]
VARIANCE_THRESHOLD = 0.01


def _download_file(url: str, dest_path: Path) -> None:
    """Baixa um arquivo via HTTP."""
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    dest_path.write_bytes(response.content)


def _try_download_individual_files(
    save_dir: Path,
    url_map: dict[str, str],
    source_name: str,
) -> bool:
    """Tenta baixar os 3 arquivos FD001 de um mapa de URLs."""
    try:
        for filename, url in url_map.items():
            dest_path = save_dir / filename
            if dest_path.exists():
                print(f"[SKIP] {filename} já existe")
                continue
            print(f"[DOWNLOAD] {filename} ({source_name}) ...")
            _download_file(url, dest_path)
            print(f"[OK] {filename} salvo ({dest_path.stat().st_size / 1024:.1f} KB)")
        return True
    except requests.RequestException as exc:
        print(f"[FALHA] Download individual ({source_name}): {exc}")
        return False


def download_cmapss(output_dir: Path) -> Path:
    """Baixa e extrai CMAPSS FD001. Retorna path da pasta com os arquivos."""
    save_dir = output_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    if all((save_dir / f).exists() for f in FD001_FILES):
        print("[SKIP] Arquivos FD001 já presentes, pulando download.")
        return save_dir

    zip_path = save_dir / "cmapss.zip"
    if not zip_path.exists():
        print(f"[DOWNLOAD] Tentando NASA zip: {NASA_ZIP_URL}")
        try:
            response = requests.get(NASA_ZIP_URL, timeout=180)
            response.raise_for_status()
            if response.headers.get("content-type", "").startswith("text/html"):
                raise requests.RequestException("Resposta HTML — URL indisponível")
            zip_path.write_bytes(response.content)
            print(f"[OK] cmapss.zip salvo ({len(response.content) / 1024:.1f} KB)")
        except requests.RequestException as exc:
            print(f"[FALHA] NASA zip: {exc}")

    if zip_path.exists():
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(save_dir)
            print(f"[OK] Zip extraído em {save_dir}")
            if all((save_dir / f).exists() for f in FD001_FILES):
                return save_dir
        except zipfile.BadZipFile as exc:
            print(f"[FALHA] Zip inválido: {exc}")
            zip_path.unlink(missing_ok=True)

    if _try_download_individual_files(save_dir, ALTERNATIVE_URLS, "schwenker/cmapss-data"):
        return save_dir

    if _try_download_individual_files(save_dir, FALLBACK_URLS, "LahiruJayasinghe/RUL-Net"):
        return save_dir

    missing = [f for f in FD001_FILES if not (save_dir / f).exists()]
    raise FileNotFoundError(f"Não foi possível obter os arquivos: {missing}")


def _read_cmapss_txt(filepath: Path) -> pd.DataFrame:
    """Lê arquivo .txt do CMAPSS sem header."""
    return pd.read_csv(
        filepath,
        sep=r"\s+",
        header=None,
        names=CMAPSS_COLUMNS,
        engine="python",
    )


def load_cmapss_fd001(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carrega treino e teste FD001 com coluna RUL calculada/atribuída."""
    train_path = raw_dir / "train_FD001.txt"
    test_path = raw_dir / "test_FD001.txt"
    rul_path = raw_dir / "RUL_FD001.txt"

    df_train = _read_cmapss_txt(train_path)
    df_test = _read_cmapss_txt(test_path)
    rul_values = np.loadtxt(rul_path).astype(float)

    max_cycles = df_train.groupby("unit_id")["cycle"].transform("max")
    df_train = df_train.copy()
    df_train["RUL"] = max_cycles - df_train["cycle"]

    df_test = df_test.copy()
    df_test["RUL"] = np.nan
    unit_ids = sorted(df_test["unit_id"].unique())
    for idx, unit_id in enumerate(unit_ids):
        last_index = df_test[df_test["unit_id"] == unit_id].index[-1]
        df_test.loc[last_index, "RUL"] = rul_values[idx]

    return df_train, df_test


def select_features_cmapss(
    df: pd.DataFrame,
    reference_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Remove sensores com variância baixa e mantém metadados + features relevantes."""
    ref = reference_df if reference_df is not None else df
    meta_cols = ["unit_id", "cycle", "RUL"]
    op_cols = ["op_setting_1", "op_setting_2", "op_setting_3"]
    sensor_cols = [col for col in df.columns if col.startswith("sensor_")]

    keep_sensors = [
        col for col in sensor_cols if ref[col].std() >= VARIANCE_THRESHOLD
    ]
    selected_cols = meta_cols + op_cols + keep_sensors

    return df[selected_cols].copy()


def normalize_cmapss(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    """Aplica MinMaxScaler (fit no treino) nas colunas de features."""
    meta_cols = {"unit_id", "cycle", "RUL"}
    feature_cols = [col for col in df_train.columns if col not in meta_cols]

    scaler = MinMaxScaler()
    df_train_norm = df_train.copy()
    df_test_norm = df_test.copy()

    df_train_norm[feature_cols] = scaler.fit_transform(df_train[feature_cols])
    df_test_norm[feature_cols] = scaler.transform(df_test[feature_cols])

    return df_train_norm, df_test_norm, scaler


def prepare_cmapss_dataset(
    raw_dir: Path,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pipeline completo: load → select_features → normalize → salvar parquet."""
    df_train, df_test = load_cmapss_fd001(raw_dir)

    df_train_sel = select_features_cmapss(df_train, reference_df=df_train)
    df_test_sel = select_features_cmapss(df_test, reference_df=df_train)

    df_train_norm, df_test_norm, _ = normalize_cmapss(df_train_sel, df_test_sel)

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "cmapss_train.parquet"
    test_path = output_dir / "cmapss_test.parquet"

    df_train_norm.to_parquet(train_path, index=False)
    df_test_norm.to_parquet(test_path, index=False)

    print(f"[OK] Treino salvo em {train_path}")
    print(f"[OK] Teste salvo em {test_path}")

    return df_train_norm, df_test_norm


if __name__ == "__main__":
    RAW_DIR = Path(__file__).resolve().parent / "raw" / "cmapss"
    OUTPUT_DIR = Path(__file__).resolve().parent / "processed"

    print("=" * 60)
    print("NASA CMAPSS FD001 — Download e Processamento")
    print("=" * 60)

    print("\n--- Download ---")
    extracted_dir = download_cmapss(RAW_DIR)

    print(f"\nArquivos em {extracted_dir}:")
    for filename in FD001_FILES:
        filepath = extracted_dir / filename
        if filepath.exists():
            print(f"  - {filename} ({filepath.stat().st_size / 1024:.1f} KB)")

    print("\n--- Processamento ---")
    df_train, df_test = prepare_cmapss_dataset(RAW_DIR, OUTPUT_DIR)

    meta_cols = {"unit_id", "cycle", "RUL"}
    feature_cols = [col for col in df_train.columns if col not in meta_cols]

    print("\n" + "=" * 60)
    print("RESUMO")
    print("=" * 60)
    print(f"Shape de train: {df_train.shape}")
    print(f"Shape de test:  {df_test.shape}")

    rul_train = df_train["RUL"].dropna()
    percentiles = np.percentile(rul_train, [25, 50, 75])
    print("\nDistribuição de RUL no treino:")
    print(f"  min:  {rul_train.min():.1f}")
    print(f"  max:  {rul_train.max():.1f}")
    print(f"  mean: {rul_train.mean():.1f}")
    print(f"  std:  {rul_train.std():.1f}")
    print(f"  P25:  {percentiles[0]:.1f}")
    print(f"  P50:  {percentiles[1]:.1f}")
    print(f"  P75:  {percentiles[2]:.1f}")

    print(f"\nFeatures selecionadas ({len(feature_cols)}):")
    for col in feature_cols:
        print(f"  - {col}")

    print(f"\nUnidades únicas — treino: {df_train['unit_id'].nunique()}")
    print(f"Unidades únicas — teste:  {df_test['unit_id'].nunique()}")
    print("=" * 60)
