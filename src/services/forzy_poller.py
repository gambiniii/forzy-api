"""
Serviço de coleta de dados em tempo real da API Forzy (ngrok estável).

Endpoints:
  GET /get_s1 → {"dados1": {"Velocidade": float, "Aceleração": float, "Temperatura": float}}
  GET /get_s2 → {"dados2": {"Velocidade": float, "Aceleração": float, "Temperatura": float}}

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
COMPONENTE_ID = 1  # componente único no banco (motor WEG W22)


async def _fetch_sensor(client: httpx.AsyncClient, endpoint: str) -> dict | None:
    try:
        resp = await client.get(endpoint, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.warning("Falha ao consultar %s: %s", endpoint, exc)
        return None


def _write_to_postgres(s1: dict, s2: dict | None) -> None:
    """Grava leitura combinada em leitura_sensor (principal) e forzy_sensor_readings (raw)."""
    try:
        from src.db.postgres import SessionLocal
        import sqlalchemy as sa

        now = datetime.now(timezone.utc)
        d1 = s1.get("dados1", {})
        d2 = s2.get("dados2", {}) if s2 else {}

        vel1 = float(d1.get("Velocidade", 0))
        acc1 = float(d1.get("Aceleração", d1.get("Aceleracao", 0)))
        tmp1 = float(d1.get("Temperatura", 0))
        vel2 = float(d2.get("Velocidade", vel1)) if d2 else vel1
        tmp2 = float(d2.get("Temperatura", tmp1)) if d2 else tmp1
        acc2 = float(d2.get("Aceleração", d2.get("Aceleracao", acc1))) if d2 else acc1

        rpm = round((vel1 + vel2) / 2, 4)
        temperatura = round((tmp1 + tmp2) / 2, 2)
        vibracao = round((acc1 + acc2) / 2, 4)

        db = SessionLocal()
        try:
            # Tabela principal — lida pelo frontend via /sensors/
            db.execute(sa.text("""
                INSERT INTO leitura_sensor (componente_id, timestamp, temperatura, rpm, vibracao)
                VALUES (:cid, :ts, :temp, :rpm, :vib)
            """), {"cid": COMPONENTE_ID, "ts": now, "temp": temperatura, "rpm": rpm, "vib": vibracao})

            # Tabela raw dos sensores físicos (cria se não existir)
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
                VALUES
                    (:ts, 1, 1, :vel1, :acc1, :tmp1, :url, :sid),
                    (:ts, 1, 2, :vel2, :acc2, :tmp2, :url, :sid)
            """), {
                "ts": now, "vel1": vel1, "acc1": acc1, "tmp1": tmp1,
                "vel2": vel2, "acc2": acc2, "tmp2": tmp2,
                "url": FORZY_BASE_URL, "sid": "poller",
            })
            db.commit()
        finally:
            db.close()
    except Exception as e:
        logger.error("Erro ao gravar no PostgreSQL: %s", e)


def _write_to_influx(s1: dict, s2: dict | None) -> None:
    try:
        from influxdb_client import Point
        from influxdb_client.client.exceptions import InfluxDBError
        from src.db.influx import get_write_api

        write_api = get_write_api()
        d1 = s1.get("dados1", {})
        point = (
            Point("sensor_readings")
            .tag("motor_id", "1")
            .tag("source", "forzy_api")
            .field("vibration_velocity_port1", float(d1.get("Velocidade", 0)))
            .field("acceleration_port1", float(d1.get("Aceleração", d1.get("Aceleracao", 0))))
            .field("temperature_port1", float(d1.get("Temperatura", 0)))
        )
        if s2:
            d2 = s2.get("dados2", {})
            point = (
                point
                .field("vibration_velocity_port2", float(d2.get("Velocidade", 0)))
                .field("acceleration_port2", float(d2.get("Aceleração", d2.get("Aceleracao", 0))))
                .field("temperature_port2", float(d2.get("Temperatura", 0)))
            )
        write_api.write(bucket=settings.INFLUX_BUCKET, org=settings.INFLUX_ORG, record=point)
    except Exception as e:
        logger.debug("InfluxDB indisponível (non-fatal): %s", e)


async def poll_loop() -> None:
    """Loop principal — coleta contínua sem restrição de janela horária."""
    import uuid
    session_id = str(uuid.uuid4())[:8]
    logger.info("Poller Forzy iniciado — URL: %s | intervalo: %ds | session: %s",
                FORZY_BASE_URL, POLL_INTERVAL_SECONDS, session_id)

    async with httpx.AsyncClient(verify=False) as client:
        while True:
            s1 = await _fetch_sensor(client, f"{FORZY_BASE_URL}/get_s1")
            s2 = await _fetch_sensor(client, f"{FORZY_BASE_URL}/get_s2")
            if s1:
                _write_to_postgres(s1, s2)
                _write_to_influx(s1, s2)
                d1 = s1.get("dados1", {})
                d2 = s2.get("dados2", {}) if s2 else {}
                logger.info(
                    "[poller] S1: Vel=%.3f Acc=%.3f Tmp=%.1f | S2: Vel=%.3f Acc=%.3f Tmp=%.1f",
                    d1.get("Velocidade", 0), d1.get("Aceleração", d1.get("Aceleracao", 0)), d1.get("Temperatura", 0),
                    d2.get("Velocidade", 0), d2.get("Aceleração", d2.get("Aceleracao", 0)), d2.get("Temperatura", 0),
                )
            else:
                logger.debug("[poller] Sensores indisponíveis. Aguardando...")

            await asyncio.sleep(POLL_INTERVAL_SECONDS)
