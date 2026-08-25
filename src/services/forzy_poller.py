"""
Serviço de coleta de dados em tempo real da API Forzy (ngrok estável).

Endpoints:
  GET /get_s1 → {"dados1": {"Velocidade": float, "Aceleração": float, "Temperatura": float}}
  GET /get_s2 → {"dados2": {"Velocidade": float, "Aceleração": float, "Temperatura": float}}

S1 e S2 são motores físicos DISTINTOS (não dois sensores do mesmo motor) —
cada um é gravado em leitura_sensor sob seu próprio componente_id, sem média
entre os dois.

Grava em PostgreSQL RDS (leitura_sensor) a cada POLL_INTERVAL_SECONDS.
"""

import asyncio
import logging
from datetime import datetime, timezone

import httpx

from src.config import settings

logger = logging.getLogger("forzy.poller")

FORZY_BASE_URL = settings.FORZY_SENSOR_URL
POLL_INTERVAL_SECONDS = 10
HTTP_TIMEOUT = 8.0

COMPONENTE_ID_S1 = 1  # Motor WEG W22
COMPONENTE_ID_S2 = 2  # Motor 2 (S2)
COMPONENTE_IDS = [COMPONENTE_ID_S1, COMPONENTE_ID_S2]


async def _fetch_sensor(client: httpx.AsyncClient, endpoint: str) -> dict | None:
    try:
        resp = await client.get(endpoint, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.warning("Falha ao consultar %s: %s", endpoint, exc)
        return None


def _write_reading_to_postgres(componente_id: int, sensor_port: int, dados: dict) -> None:
    """Grava a leitura de UM motor em leitura_sensor (principal) e forzy_sensor_readings (raw)."""
    try:
        from src.db.postgres import SessionLocal
        import sqlalchemy as sa

        now = datetime.now(timezone.utc)
        vel = float(dados.get("Velocidade", 0))
        acc = float(dados.get("Aceleração", dados.get("Aceleracao", 0)))
        tmp = float(dados.get("Temperatura", 0))

        db = SessionLocal()
        try:
            # Tabela principal — lida pelo frontend via /sensors/
            db.execute(sa.text("""
                INSERT INTO leitura_sensor (componente_id, timestamp, temperatura, rpm, vibracao)
                VALUES (:cid, :ts, :temp, :rpm, :vib)
            """), {"cid": componente_id, "ts": now, "temp": tmp, "rpm": vel, "vib": acc})

            # Tabela raw do sensor físico (cria se não existir)
            db.execute(sa.text("""
                CREATE TABLE IF NOT EXISTS forzy_sensor_readings (
                    id SERIAL PRIMARY KEY,
                    collected_at TIMESTAMPTZ NOT NULL,
                    motor_id INTEGER,
                    sensor_port SMALLINT,
                    velocidade FLOAT,
                    aceleracao FLOAT,
                    temperatura FLOAT,
                    api_url TEXT,
                    session_id TEXT
                )
            """))
            db.execute(sa.text("""
                INSERT INTO forzy_sensor_readings
                    (collected_at, motor_id, sensor_port, velocidade, aceleracao, temperatura, api_url, session_id)
                VALUES (:ts, :mid, :port, :vel, :acc, :tmp, :url, :sid)
            """), {
                "ts": now, "mid": componente_id, "port": sensor_port,
                "vel": vel, "acc": acc, "tmp": tmp,
                "url": FORZY_BASE_URL, "sid": "poller",
            })
            db.commit()
        finally:
            db.close()
    except Exception as e:
        logger.error("componente=%d — erro ao gravar no PostgreSQL: %s", componente_id, e)


def _write_reading_to_influx(componente_id: int, dados: dict) -> None:
    try:
        from influxdb_client import Point
        from src.db.influx import get_write_api

        write_api = get_write_api()
        point = (
            Point("sensor_readings")
            .tag("motor_id", str(componente_id))
            .tag("source", "forzy_api")
            .field("vibration_velocity", float(dados.get("Velocidade", 0)))
            .field("acceleration", float(dados.get("Aceleração", dados.get("Aceleracao", 0))))
            .field("temperature", float(dados.get("Temperatura", 0)))
        )
        write_api.write(bucket=settings.INFLUX_BUCKET, org=settings.INFLUX_ORG, record=point)
    except Exception as e:
        logger.debug("componente=%d — InfluxDB indisponível (non-fatal): %s", componente_id, e)


async def poll_loop() -> None:
    """Loop principal — coleta contínua sem restrição de janela horária."""
    logger.info(
        "Poller Forzy iniciado — URL: %s | intervalo: %ds | S1→componente=%d | S2→componente=%d",
        FORZY_BASE_URL, POLL_INTERVAL_SECONDS, COMPONENTE_ID_S1, COMPONENTE_ID_S2,
    )

    async with httpx.AsyncClient(verify=False) as client:
        while True:
            s1 = await _fetch_sensor(client, f"{FORZY_BASE_URL}/get_s1")
            s2 = await _fetch_sensor(client, f"{FORZY_BASE_URL}/get_s2")

            if s1:
                d1 = s1.get("dados1", {})
                _write_reading_to_postgres(COMPONENTE_ID_S1, 1, d1)
                _write_reading_to_influx(COMPONENTE_ID_S1, d1)
                logger.info(
                    "[poller] S1 (componente=%d): Vel=%.3f Acc=%.3f Tmp=%.1f",
                    COMPONENTE_ID_S1, d1.get("Velocidade", 0),
                    d1.get("Aceleração", d1.get("Aceleracao", 0)), d1.get("Temperatura", 0),
                )

            if s2:
                d2 = s2.get("dados2", {})
                _write_reading_to_postgres(COMPONENTE_ID_S2, 2, d2)
                _write_reading_to_influx(COMPONENTE_ID_S2, d2)
                logger.info(
                    "[poller] S2 (componente=%d): Vel=%.3f Acc=%.3f Tmp=%.1f",
                    COMPONENTE_ID_S2, d2.get("Velocidade", 0),
                    d2.get("Aceleração", d2.get("Aceleracao", 0)), d2.get("Temperatura", 0),
                )

            if not s1 and not s2:
                logger.debug("[poller] Sensores indisponíveis. Aguardando...")

            await asyncio.sleep(POLL_INTERVAL_SECONDS)
