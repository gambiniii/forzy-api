"""
Serviço de coleta de dados em tempo real da API Forzy.

Endpoints monitorados:
  GET /get_s1 → {"dados1": {"Velocidade": float, "Aceleração": float, "Temperatura": float}}
  GET /get_s2 → {"dados2": {"Velocidade": float, "Aceleração": float, "Temperatura": float}}

Janela de operação do motor: Seg-Qua, 12h-14h (horário de Brasília).
Grava no InfluxDB (se disponível) E no PostgreSQL RDS a cada POLL_INTERVAL_SECONDS.
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
MOTOR_ID_FORZY = 1  # int para PostgreSQL


def _is_motor_window() -> bool:
    """Verifica se estamos na janela de operação (Seg-Qua 12h-14h BRT = UTC-3)."""
    from zoneinfo import ZoneInfo
    now_brt = datetime.now(ZoneInfo("America/Sao_Paulo"))
    if now_brt.weekday() not in (0, 1, 2):
        return False
    return 12 <= now_brt.hour < 14


async def _fetch_sensor(client: httpx.AsyncClient, endpoint: str) -> dict | None:
    try:
        resp = await client.get(endpoint, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.warning("Falha ao consultar %s: %s", endpoint, exc)
        return None


def _write_to_influx(s1: dict, s2: dict | None) -> None:
    """Grava no InfluxDB se disponível."""
    try:
        from influxdb_client import Point
        from influxdb_client.client.exceptions import InfluxDBError
        from src.db.influx import get_write_api

        write_api = get_write_api()
        point = (
            Point("sensor_readings")
            .tag("motor_id", str(MOTOR_ID_FORZY))
            .tag("source", "forzy_api")
        )
        d1 = s1.get("dados1", {})
        point = (
            point
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
    except InfluxDBError as e:
        logger.warning("InfluxDB write error (non-fatal): %s", e)
    except Exception as e:
        logger.debug("InfluxDB indisponível (non-fatal): %s", e)


def _write_to_postgres(s1: dict, s2: dict | None, session_id: str) -> None:
    """Grava no PostgreSQL RDS — armazenamento principal AWS."""
    try:
        from src.db.postgres import SessionLocal
        import sqlalchemy as sa

        now = datetime.now(timezone.utc)
        rows = []

        d1 = s1.get("dados1", {})
        rows.append({
            "collected_at": now,
            "motor_id": MOTOR_ID_FORZY,
            "sensor_port": 1,
            "velocidade": float(d1.get("Velocidade", 0)),
            "aceleracao": float(d1.get("Aceleração", d1.get("Aceleracao", 0))),
            "temperatura": float(d1.get("Temperatura", 0)),
            "api_url": FORZY_BASE_URL,
            "session_id": session_id,
        })

        if s2:
            d2 = s2.get("dados2", {})
            rows.append({
                "collected_at": now,
                "motor_id": MOTOR_ID_FORZY,
                "sensor_port": 2,
                "velocidade": float(d2.get("Velocidade", 0)),
                "aceleracao": float(d2.get("Aceleração", d2.get("Aceleracao", 0))),
                "temperatura": float(d2.get("Temperatura", 0)),
                "api_url": FORZY_BASE_URL,
                "session_id": session_id,
            })

        db = SessionLocal()
        try:
            db.execute(
                sa.text("""
                    INSERT INTO forzy_sensor_readings
                        (collected_at, motor_id, sensor_port, velocidade, aceleracao, temperatura, api_url, session_id)
                    VALUES
                        (:collected_at, :motor_id, :sensor_port, :velocidade, :aceleracao, :temperatura, :api_url, :session_id)
                """),
                rows,
            )
            db.commit()
        finally:
            db.close()
    except Exception as e:
        logger.error("Erro ao gravar no PostgreSQL: %s", e)


async def poll_loop() -> None:
    """Loop principal de coleta. Roda como background task no startup da API."""
    import uuid
    session_id = str(uuid.uuid4())[:8]
    logger.info("Poller Forzy iniciado — intervalo: %ds | session: %s", POLL_INTERVAL_SECONDS, session_id)

    async with httpx.AsyncClient(verify=False) as client:
        while True:
            if _is_motor_window():
                s1 = await _fetch_sensor(client, f"{FORZY_BASE_URL}/get_s1")
                s2 = await _fetch_sensor(client, f"{FORZY_BASE_URL}/get_s2")
                if s1:
                    _write_to_postgres(s1, s2, session_id)
                    _write_to_influx(s1, s2)
                    v1 = s1.get("dados1", {}).get("Velocidade", "?")
                    v2 = s2.get("dados2", {}).get("Velocidade", "?") if s2 else "?"
                    logger.info("[poller] V_port1: %s mm/s | V_port2: %s mm/s", v1, v2)
            else:
                logger.debug("Fora da janela de operação. Aguardando...")

            await asyncio.sleep(POLL_INTERVAL_SECONDS)
