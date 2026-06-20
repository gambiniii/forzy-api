"""
Download e processamento do CWRU Bearing Dataset (Case Western Reserve University).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.io import loadmat

BASE_URL = "https://engineering.case.edu/sites/default/files/"

CWRU_FILES = {
    "97.mat": {"fault_type": "normal", "label": 0, "description": "Normal 1797 RPM"},
    "98.mat": {"fault_type": "normal", "label": 0, "description": "Normal 1772 RPM"},
    "105.mat": {"fault_type": "inner_race", "label": 1, "description": "Inner Race 1797 RPM"},
    "106.mat": {"fault_type": "inner_race", "label": 1, "description": "Inner Race 1772 RPM"},
    "118.mat": {"fault_type": "ball", "label": 1, "description": "Ball fault 1797 RPM"},
    "119.mat": {"fault_type": "ball", "label": 1, "description": "Ball fault 1772 RPM"},
    "130.mat": {"fault_type": "outer_race", "label": 1, "description": "Outer Race 1797 RPM"},
    "131.mat": {"fault_type": "outer_race", "label": 1, "description": "Outer Race 1772 RPM"},
}


def download_cwru(output_dir: Path) -> None:
    """Baixa arquivos .mat do repositório CWRU para output_dir/cwru_raw/."""
    save_dir = output_dir / "cwru_raw"
    save_dir.mkdir(parents=True, exist_ok=True)

    for filename in CWRU_FILES:
        dest_path = save_dir / filename
        if dest_path.exists():
            print(f"[SKIP] {filename} já existe em {dest_path}")
            continue

        url = f"{BASE_URL}{filename}"
        print(f"[DOWNLOAD] Baixando {filename} de {url} ...")

        response = requests.get(url, timeout=120)
        response.raise_for_status()

        dest_path.write_bytes(response.content)
        print(f"[OK] {filename} salvo ({len(response.content) / 1024:.1f} KB)")


def load_mat_file(filepath: Path) -> dict:
    """Carrega arquivo .mat e extrai sinal DE_time (drive end)."""
    mat = loadmat(filepath)

    de_keys = [key for key in mat.keys() if "DE_time" in key and not key.startswith("__")]
    if not de_keys:
        raise KeyError(f"Nenhuma chave DE_time encontrada em {filepath.name}")

    signal = mat[de_keys[0]].flatten().astype(np.float64)

    return {
        "signal": signal,
        "filename": filepath.name,
    }


def mat_to_dataframe(mat_data: dict, label: int, fault_type: str) -> pd.DataFrame:
    """Converte sinal de vibração 1D em DataFrame com metadados."""
    return pd.DataFrame(
        {
            "signal": mat_data["signal"],
            "label": label,
            "fault_type": fault_type,
            "sample_idx": np.arange(len(mat_data["signal"])),
        }
    )


def create_windows(
    df: pd.DataFrame,
    window_size: int = 512,
    step: int = 256,
) -> pd.DataFrame:
    """Cria janelas deslizantes do sinal de vibração."""
    signal = df["signal"].to_numpy()
    label = int(df["label"].iloc[0])
    fault_type = str(df["fault_type"].iloc[0])

    windows = []
    for start in range(0, len(signal) - window_size + 1, step):
        window = signal[start : start + window_size]
        row = {f"feat_{i}": window[i] for i in range(window_size)}
        row["label"] = label
        row["fault_type"] = fault_type
        windows.append(row)

    return pd.DataFrame(windows)


def _resolve_mat_dir(raw_dir: Path) -> Path:
    """Resolve diretório onde os arquivos .mat foram salvos."""
    cwru_raw = raw_dir / "cwru_raw"
    if cwru_raw.exists():
        return cwru_raw
    return raw_dir


def prepare_cwru_dataset(
    raw_dir: Path,
    output_dir: Path,
    window_size: int = 512,
    step: int = 256,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Processa arquivos .mat, cria janelas e salva dataset processado."""
    mat_dir = _resolve_mat_dir(raw_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_windows: list[pd.DataFrame] = []

    for filename, meta in CWRU_FILES.items():
        filepath = mat_dir / filename
        if not filepath.exists():
            print(f"[AVISO] Arquivo não encontrado, pulando: {filepath}")
            continue

        print(f"[PROCESS] {filename} ({meta['description']})")
        mat_data = load_mat_file(filepath)
        signal_df = mat_to_dataframe(mat_data, meta["label"], meta["fault_type"])
        windows_df = create_windows(signal_df, window_size=window_size, step=step)
        windows_df["source_file"] = filename
        all_windows.append(windows_df)
        print(f"         -> {len(windows_df)} janelas criadas")

    if not all_windows:
        raise FileNotFoundError(f"Nenhum arquivo .mat encontrado em {mat_dir}")

    df_completo = pd.concat(all_windows, ignore_index=True)

    feature_cols = [f"feat_{i}" for i in range(window_size)]
    X = df_completo[feature_cols].to_numpy()
    y = df_completo["label"].to_numpy()

    parquet_path = output_dir / "cwru_processed.parquet"
    df_completo.to_parquet(parquet_path, index=False)
    print(f"[OK] Dataset salvo em {parquet_path}")

    return X, y, df_completo


if __name__ == "__main__":
    RAW_DIR = Path(__file__).resolve().parent / "raw" / "cwru"
    PROCESSED_DIR = Path(__file__).resolve().parent / "processed"

    print("=" * 60)
    print("CWRU Bearing Dataset — Download e Processamento")
    print("=" * 60)

    print("\n--- Download ---")
    download_cwru(RAW_DIR)

    mat_dir = _resolve_mat_dir(RAW_DIR)
    downloaded = sorted(mat_dir.glob("*.mat"))
    print(f"\nArquivos em {mat_dir}:")
    for f in downloaded:
        print(f"  - {f.name} ({f.stat().st_size / 1024:.1f} KB)")

    print("\n--- Processamento ---")
    X, y, df_completo = prepare_cwru_dataset(RAW_DIR, PROCESSED_DIR)

    print("\n" + "=" * 60)
    print("RESUMO")
    print("=" * 60)
    print(f"Shape do dataset processado: {df_completo.shape}")
    print(f"Shape de X: {X.shape}")
    print(f"Shape de y: {y.shape}")

    print("\nDistribuição de classes:")
    class_dist = df_completo["label"].value_counts().sort_index()
    for label, count in class_dist.items():
        name = "normal" if label == 0 else "anomalia"
        pct = count / len(df_completo) * 100
        print(f"  label={label} ({name}): {count} ({pct:.1f}%)")

    print("\nDistribuição por fault_type:")
    fault_dist = df_completo["fault_type"].value_counts()
    for fault_type, count in fault_dist.items():
        pct = count / len(df_completo) * 100
        print(f"  {fault_type}: {count} ({pct:.1f}%)")

    print("=" * 60)
