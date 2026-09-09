"""
Carrega o histórico real (sensor/History_*.csv) direto em leitura_sensor para
os motores S1 (componente_id=2) e S2 (componente_id=3) — dado GENUÍNO, não
sintético, marcado como origem='real'.

Sem isso, um banco novo + seed_forzy_motors.py cria as máquinas mas sem
NENHUMA leitura: gráfico vazio, sem tendência de saúde, sem diagnóstico até o
poller/modo demonstração acumular dado do zero.

Os timestamps originais são preservados nos deltas (mesma cadência real, com
as transições parado→transiente→operação do dataset) — só deslocados no
tempo pra a leitura mais recente terminar em "agora", pra parecer histórico
recém-capturado.

Uso: python scripts/seed_leituras_historico.py
Idempotente — não duplica se já houver leitura para os componentes 2/3.
"""
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import text

from src.db.postgres import SessionLocal, create_tables

CSV_PATH = Path(__file__).resolve().parents[1] / "sensor" / "History_32026-05-19T11-46-10-920.csv"
COMPONENTE_ID_S1 = 2
COMPONENTE_ID_S2 = 3
TAMANHO_LOTE = 500


def _carregar_csv() -> list[dict]:
    linhas: list[dict] = []
    with open(CSV_PATH, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=";")
        for _ in range(3):
            next(reader, None)  # 3 linhas de header
        for row in reader:
            if len(row) < 9:
                continue
            try:
                linhas.append({
                    "timestamp": datetime.fromisoformat(row[0]),
                    "s1": {"rpm": float(row[3]), "vibracao": float(row[4]), "temperatura": float(row[5])},
                    "s2": {"rpm": float(row[6]), "vibracao": float(row[7]), "temperatura": float(row[8])},
                })
            except ValueError:
                continue
    return linhas


def run() -> None:
    create_tables()
    db = SessionLocal()
    try:
        ja_existe = db.execute(text(
            "SELECT 1 FROM leitura_sensor WHERE componente_id IN (:s1, :s2) LIMIT 1"
        ), {"s1": COMPONENTE_ID_S1, "s2": COMPONENTE_ID_S2}).first()
        if ja_existe:
            print("Já existem leituras para os componentes 2/3 — pulando (evita duplicar histórico).")
            return

        linhas = _carregar_csv()
        if not linhas:
            print(f"CSV vazio ou não encontrado em {CSV_PATH} — nada a semear.")
            return

        agora = datetime.now(timezone.utc)
        deslocamento = agora - linhas[-1]["timestamp"].replace(tzinfo=timezone.utc)
        print(f"{len(linhas)} leituras no histórico — inserindo para os componentes 2 e 3...")

        params: list[dict] = []
        for linha in linhas:
            ts = linha["timestamp"].replace(tzinfo=timezone.utc) + deslocamento
            for componente_id, chave in ((COMPONENTE_ID_S1, "s1"), (COMPONENTE_ID_S2, "s2")):
                d = linha[chave]
                params.append({
                    "cid": componente_id, "ts": ts,
                    "temp": d["temperatura"], "rpm": d["rpm"], "vib": d["vibracao"],
                })

        sql = text("""
            INSERT INTO leitura_sensor (componente_id, timestamp, temperatura, rpm, vibracao, origem)
            VALUES (:cid, :ts, :temp, :rpm, :vib, 'real')
        """)
        for i in range(0, len(params), TAMANHO_LOTE):
            db.execute(sql, params[i:i + TAMANHO_LOTE])
            db.commit()
            print(f"  {min(i + TAMANHO_LOTE, len(params))}/{len(params)}...")

        print(f"Concluído: {len(params)} leituras inseridas (componentes 2 e 3).")
    finally:
        db.close()


if __name__ == "__main__":
    run()
