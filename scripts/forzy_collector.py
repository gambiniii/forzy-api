"""
Coletor autônomo de dados Forzy — roda INDEPENDENTE do servidor FastAPI.

Inicia pelo Windows Task Scheduler nos dias e horários:
  Segunda, Terça, Quarta — 11h55 BRT (encerra automaticamente após a janela 14h)

Coleta dados dos 2 sensores a cada POLL_INTERVAL_SECONDS e armazena
na tabela forzy_sensor_readings do PostgreSQL RDS (AWS).

Uso manual:
  cd forzy-api-gambiniii
  python scripts/forzy_collector.py

Variáveis obrigatórias (carregadas do .env ou ambiente):
  POSTGRES_HOST, POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB, POSTGRES_PORT
"""

import logging
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

# ─── Carrega .env do diretório pai (forzy-api-gambiniii) ─────────────────────
_root = Path(__file__).resolve().parent.parent
if (_root / ".env").exists():
    from dotenv import load_dotenv
    load_dotenv(_root / ".env")

# ─── Lock file: impede duas instâncias simultâneas ───────────────────────────
_lock_file = _root / "logs" / "collector.lock"
(_root / "logs").mkdir(exist_ok=True)
if _lock_file.exists():
    try:
        _pid = int(_lock_file.read_text().strip())
        import psutil
        if psutil.pid_exists(_pid):
            print(f"Outra instancia ja esta rodando (PID {_pid}). Encerrando.")
            sys.exit(0)
    except Exception:
        pass  # lock file corrompido ou psutil ausente — continua
_lock_file.write_text(str(os.getpid()))

import atexit
atexit.register(lambda: _lock_file.unlink(missing_ok=True))

# ─── Logging para arquivo + console ──────────────────────────────────────────
_log_dir = _root / "logs"
_log_dir.mkdir(exist_ok=True)
_log_file = _log_dir / f"collector_{datetime.now().strftime('%Y%m%d')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(_log_file, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("forzy.collector")

# ─── Constantes (URL lida do .env para facilitar atualização) ────────────────
FORZY_BASE_URL = os.environ.get(
    "FORZY_SENSOR_URL",
    "https://reseller-prescribed-facing-dept.trycloudflare.com",
)
POLL_INTERVAL_SECONDS = 10
HTTP_TIMEOUT = 8.0
MOTOR_ID = 1

# ─── Timezone BRT ─────────────────────────────────────────────────────────────
try:
    from zoneinfo import ZoneInfo
    BRT = ZoneInfo("America/Sao_Paulo")
except ImportError:
    import pytz
    BRT = pytz.timezone("America/Sao_Paulo")


def now_brt() -> datetime:
    return datetime.now(BRT)


def _is_collection_window() -> bool:
    """Seg-Qua (0,1,2) das 12h às 14h BRT."""
    t = now_brt()
    return t.weekday() in (0, 1, 2) and 12 <= t.hour < 14


# ─── Conexão PostgreSQL ───────────────────────────────────────────────────────
def _get_pg_conn():
    import psycopg2
    return psycopg2.connect(
        host=os.environ["POSTGRES_HOST"],
        port=int(os.environ.get("POSTGRES_PORT", 5432)),
        dbname=os.environ["POSTGRES_DB"],
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
        connect_timeout=10,
        sslmode="require",
    )


def _ensure_table(conn) -> None:
    """Garante que a tabela existe (idempotente)."""
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS forzy_sensor_readings (
                id          BIGSERIAL PRIMARY KEY,
                collected_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                motor_id    INTEGER NOT NULL,
                sensor_port SMALLINT NOT NULL,
                velocidade  FLOAT NOT NULL,
                aceleracao  FLOAT NOT NULL,
                temperatura FLOAT NOT NULL,
                api_url     TEXT,
                session_id  TEXT
            )
        """)
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_fsr_collected_at "
            "ON forzy_sensor_readings(collected_at DESC)"
        )
        conn.commit()


def _save_reading(conn, port: int, data: dict, session_id: str, ts: datetime) -> None:
    key = f"dados{port}"
    d = data.get(key, {})
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO forzy_sensor_readings
                (collected_at, motor_id, sensor_port, velocidade, aceleracao, temperatura, api_url, session_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                ts,
                MOTOR_ID,
                port,
                float(d.get("Velocidade", 0)),
                float(d.get("Aceleração", d.get("Aceleracao", 0))),
                float(d.get("Temperatura", 0)),
                FORZY_BASE_URL,
                session_id,
            ),
        )
    conn.commit()


# ─── HTTP ─────────────────────────────────────────────────────────────────────
def _fetch(url: str) -> dict | None:
    import urllib.request
    import urllib.error
    import json
    import ssl

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request(url, headers={"accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT, context=ctx) as resp:
            return json.loads(resp.read().decode())
    except Exception as exc:
        log.warning("Falha em %s: %s", url, exc)
        return None


# ─── Loop principal ───────────────────────────────────────────────────────────
def run() -> None:
    session_id = str(uuid.uuid4())[:8]
    log.info("=" * 60)
    log.info("Forzy Collector iniciado | session=%s", session_id)
    log.info("Motor ID: %d | Intervalo: %ds", MOTOR_ID, POLL_INTERVAL_SECONDS)
    log.info("Janela: Seg-Qua 12h-14h BRT")
    log.info("Log: %s", _log_file)
    log.info("=" * 60)

    # ── Conecta ao PostgreSQL ─────────────────────────────────────────────────
    try:
        conn = _get_pg_conn()
        _ensure_table(conn)
        log.info("PostgreSQL RDS conectado.")
    except Exception as e:
        log.critical("Não foi possível conectar ao PostgreSQL: %s", e)
        sys.exit(1)

    total_collected = 0
    consecutive_failures = 0
    MAX_FAILURES = 12  # ~2 minutos de falhas antes de logar aviso

    while True:
        t = now_brt()

        # Fora da janela → aguarda ou encerra
        if not _is_collection_window():
            # Se passou das 14h em dia válido → encerra (Task Scheduler já agendará o próximo)
            if t.weekday() in (0, 1, 2) and t.hour >= 14:
                log.info("Janela encerrada às %s BRT. Total coletado: %d leituras.", t.strftime("%H:%M"), total_collected)
                log.info("Encerrando collector.")
                break
            log.info("Aguardando janela de operação... Agora: %s BRT (dia %d)", t.strftime("%H:%M"), t.weekday())
            time.sleep(60)
            continue

        # ── Dentro da janela → coleta ─────────────────────────────────────────
        ts = datetime.now(timezone.utc)
        s1 = _fetch(f"{FORZY_BASE_URL}/get_s1")
        s2 = _fetch(f"{FORZY_BASE_URL}/get_s2")

        if s1 is None and s2 is None:
            consecutive_failures += 1
            if consecutive_failures == MAX_FAILURES:
                log.warning(
                    "API inacessível por %d tentativas consecutivas (~%dm). "
                    "Motor pode estar desligado.",
                    consecutive_failures,
                    consecutive_failures * POLL_INTERVAL_SECONDS // 60,
                )
            time.sleep(POLL_INTERVAL_SECONDS)
            continue

        consecutive_failures = 0

        try:
            if s1:
                _save_reading(conn, 1, s1, session_id, ts)
                total_collected += 1
            if s2:
                _save_reading(conn, 2, s2, session_id, ts)
                total_collected += 1
        except Exception as e:
            log.error("Erro ao salvar leitura: %s. Reconectando...", e)
            try:
                conn.close()
            except Exception:
                pass
            try:
                conn = _get_pg_conn()
            except Exception as e2:
                log.critical("Falha ao reconectar: %s", e2)
                time.sleep(30)
                continue

        v1 = s1.get("dados1", {}).get("Velocidade", "?") if s1 else "?"
        v2 = s2.get("dados2", {}).get("Velocidade", "?") if s2 else "?"
        log.info("[%s] S1: %.3f mm/s | S2: %.3f mm/s | total: %d",
                 t.strftime("%H:%M:%S"), v1 if isinstance(v1, float) else 0.0,
                 v2 if isinstance(v2, float) else 0.0, total_collected)

        time.sleep(POLL_INTERVAL_SECONDS)

    try:
        conn.close()
    except Exception:
        pass
    log.info("Collector encerrado. Total de leituras salvas: %d", total_collected)


if __name__ == "__main__":
    run()
