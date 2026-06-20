"""
Preprocessador do Ottawa Electric Motor Dataset para o formato Forzy.

Converte sinal bruto (42 kHz, m/s^2) para trend (1 s/amostra):
  - Velocidade RMS  → integração numérica aceleração → mm/s
  - Aceleração RMS  → RMS da aceleração bruta → g (÷9.81)
  - Temperatura     → média da janela → °C

Saída: DataFrame com as mesmas colunas do CSV Forzy + coluna 'label' (0=normal, 1=anomalia)
e colunas port2 espelhadas de port1 (igual ao forzy_adapter em produção).

Convenção de nomes: {letra1}_{letra2}_{numero1}_{numero2}

Letra 1 (estado do motor):
  H = healthy        S = stator         V = voltage
  R = rotor          B = bowed          K = broken         F = faulty

Letra 2 (tipo de falha):
  H = healthy        U = unbalance      M = misalignment   W = winding
  R = rotor          A = rotor bars     B = bearing

Normal: letra1=H E letra2=H  (HH = motor completamente saudável)
Anomalia: qualquer outra combinação

Número 1 (velocidade):  1-4=constante (15/30/45/60 Hz), 5-8=variável
Número 2 (carga):       0=sem carga, 1=com carga
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

SAMPLE_RATE = 42_000          # Hz — 420000 amostras / 10 segundos
WINDOW_SIZE = SAMPLE_RATE     # 1 janela = 1 segundo = 42000 amostras
G = 9.81                      # m/s² por g


def _is_normal(fname: str) -> bool | None:
    """
    Retorna True se o arquivo for motor saudável (H_H_*),
    False se for qualquer tipo de falha, None se não reconhecido.
    """
    parts = fname.split("_")
    if len(parts) < 2:
        return None
    letter1, letter2 = parts[0], parts[1]
    if letter1 == "H" and letter2 == "H":
        return True
    # qualquer combinação com falha
    valid_l1 = {"H", "R", "S", "V", "B", "K", "F"}
    valid_l2 = {"H", "U", "M", "W", "R", "A", "B"}
    if letter1 in valid_l1 and letter2 in valid_l2:
        return False
    return None


def _acc_to_velocity_rms(acc_ms2: np.ndarray, fs: int = SAMPLE_RATE) -> float:
    """
    Integra aceleração (m/s²) numericamente para obter velocidade (m/s),
    depois calcula RMS e converte para mm/s.
    Usa integração trapezoidal com remoção de tendência (detrend) antes.
    """
    dt = 1.0 / fs
    acc_detrended = acc_ms2 - acc_ms2.mean()
    velocity = np.cumsum(acc_detrended) * dt          # m/s
    velocity -= velocity.mean()                        # remove drift DC
    rms_ms = np.sqrt(np.mean(velocity ** 2))
    return rms_ms * 1000.0                             # → mm/s


def _acc_to_rms_g(acc_ms2: np.ndarray) -> float:
    """RMS da aceleração bruta em g."""
    rms = np.sqrt(np.mean(acc_ms2 ** 2))
    return rms / G


def _process_file(
    raw: pd.DataFrame,
    label: int,
    window_size: int = WINDOW_SIZE,
) -> pd.DataFrame:
    """
    Processa um arquivo Ottawa (420000 linhas) em janelas de 1 segundo.
    Retorna DataFrame com colunas Forzy + label.
    """
    acc1 = raw["Accelerometer 1 (m/s^2)"].values
    temp = raw["Temperature (Celsius)"].values

    n_windows = len(acc1) // window_size
    rows = []
    for i in range(n_windows):
        start = i * window_size
        end   = start + window_size
        chunk_acc  = acc1[start:end]
        chunk_temp = temp[start:end]

        vel_rms  = _acc_to_velocity_rms(chunk_acc)
        acc_rms  = _acc_to_rms_g(chunk_acc)
        temp_avg = float(chunk_temp.mean())

        rows.append({
            "1.1. Velocidade":  round(vel_rms,  4),
            "1.2. Aceleração":  round(acc_rms,  4),
            "1.3. Temperatura": round(temp_avg,  2),
            # port2 espelhada = port1 (igual ao forzy_adapter em produção)
            "2.1. Velocidade":  round(vel_rms,  4),
            "2.2. Aceleração":  round(acc_rms,  4),
            "2.3. Temperatura": round(temp_avg,  2),
            "label": label,
        })

    return pd.DataFrame(rows)


def prepare_ottawa(
    zip_path: str | Path,
    output_path: str | Path,
    condition: str = "both",   # "unloaded", "loaded", ou "both"
    max_files_per_class: int = 20,
) -> pd.DataFrame:
    """
    Extrai e processa arquivos CSV do zip Ottawa.

    Args:
        zip_path: caminho para o .zip do dataset Ottawa
        output_path: onde salvar o parquet processado
        condition: quais condições de carga incluir
        max_files_per_class: limite de arquivos por classe (normal/anomalia)
                             para não explodir a memória

    Returns:
        DataFrame processado com colunas Forzy + label
    """
    zip_path    = Path(zip_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    condition_folders = []
    if condition in ("unloaded", "both"):
        condition_folders.append("1_Unloaded_Condition")
    if condition in ("loaded", "both"):
        condition_folders.append("2_Loaded_Condition")

    all_frames: list[pd.DataFrame] = []
    normal_count  = 0
    anomaly_count = 0

    print(f"Abrindo {zip_path.name}...")
    with zipfile.ZipFile(zip_path, "r") as z:
        csv_files = [
            n for n in z.namelist()
            if n.endswith(".csv") and "2_CSV_Data_Files" in n
            and any(cf in n for cf in condition_folders)
        ]
        print(f"Total de arquivos CSV encontrados: {len(csv_files)}")

        for fpath in sorted(csv_files):
            fname   = Path(fpath).stem          # ex: H_H_1_0  ou  B_R_3_1
            is_norm = _is_normal(fname)

            if is_norm is None:
                continue
            if is_norm:
                if normal_count >= max_files_per_class:
                    continue
                label = 0
                normal_count += 1
            else:
                if anomaly_count >= max_files_per_class:
                    continue
                label = 1
                anomaly_count += 1

            raw = pd.read_csv(io.BytesIO(z.read(fpath)))
            df  = _process_file(raw, label)
            all_frames.append(df)
            print(f"  {'NORMAL' if label==0 else 'ANOMALIA'} | {fname} -> {len(df)} amostras de 1s")

    if not all_frames:
        raise RuntimeError("Nenhum arquivo processado — verifique o zip e os filtros.")

    result = pd.concat(all_frames, ignore_index=True)

    n_normal  = (result["label"] == 0).sum()
    n_anomaly = (result["label"] == 1).sum()
    print(f"\nDataset Ottawa processado:")
    print(f"  Normal:   {n_normal} amostras")
    print(f"  Anomalia: {n_anomaly} amostras")
    print(f"  Total:    {len(result)} amostras")
    print(f"\nEstatísticas:")
    print(result[["1.1. Velocidade","1.2. Aceleração","1.3. Temperatura"]].describe().round(4))

    try:
        result.to_parquet(output_path, index=False)
    except ImportError:
        # fallback para CSV se pyarrow não estiver instalado
        output_path = output_path.with_suffix(".csv")
        result.to_csv(output_path, index=False)
    print(f"\nSalvo em: {str(output_path)}")
    return result


if __name__ == "__main__":
    import sys
    ROOT = Path(__file__).resolve().parents[3]   # raiz do projeto FIAP/_FORZY
    ZIP  = ROOT / "University of Ottawa Electric Motor Dataset – Vibr.zip"
    OUT  = Path(__file__).resolve().parents[1] / "data" / "processed" / "ottawa_processed.parquet"

    if not ZIP.exists():
        print(f"ERRO: zip não encontrado em {ZIP}")
        sys.exit(1)

    prepare_ottawa(ZIP, OUT, condition="both", max_files_per_class=20)
