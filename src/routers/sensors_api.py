import re
from datetime import datetime, timezone, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.db.postgres import get_db
from src.routers.auth import get_current_user

router = APIRouter()


class SensorReading(BaseModel):
    id: int
    componente_id: int
    timestamp: str
    temperatura: Optional[float] = None
    umidade: Optional[float] = None
    corrente: Optional[float] = None
    voltagem: Optional[float] = None
    rpm: Optional[float] = None
    vibracao: Optional[float] = None
    inclinacao: Optional[float] = None


def _parse_start(start: Optional[str]) -> Optional[datetime]:
    if not start:
        return None
    m = re.match(r"^-(\d+)([hdm])$", start.strip())
    if m:
        n, unit = int(m.group(1)), m.group(2)
        now = datetime.now(timezone.utc)
        if unit == "h":
            return now - timedelta(hours=n)
        if unit == "d":
            return now - timedelta(days=n)
        if unit == "m":
            return now - timedelta(minutes=n)
    try:
        return datetime.fromisoformat(start.replace("Z", "+00:00"))
    except ValueError:
        return None


def _row_to_reading(r) -> SensorReading:
    return SensorReading(
        id=r[0], componente_id=r[1],
        timestamp=r[2].isoformat() if r[2] else "",
        temperatura=r[3], umidade=r[4], corrente=r[5],
        voltagem=r[6], rpm=r[7], vibracao=r[8], inclinacao=r[9],
    )


@router.get("/component/{component_id}/latest", response_model=Optional[SensorReading])
def get_latest_sensor(
    component_id: int,
    db: Session = Depends(get_db),
    _=Depends(get_current_user),
):
    row = db.execute(text("""
        SELECT id, componente_id, timestamp, temperatura, umidade,
               corrente, voltagem, rpm, vibracao, inclinacao
        FROM leitura_sensor
        WHERE componente_id = :cid
        ORDER BY timestamp DESC LIMIT 1
    """), {"cid": component_id}).fetchone()
    return _row_to_reading(row) if row else None


@router.get("/component/{component_id}", response_model=list[SensorReading])
def get_sensor_readings(
    component_id: int,
    start: Optional[str] = Query(default=None),
    limit: int = Query(default=100, le=2000),
    db: Session = Depends(get_db),
    _=Depends(get_current_user),
):
    since = _parse_start(start)
    params: dict = {"cid": component_id, "lim": limit}
    if since:
        params["since"] = since
        sql = """
            SELECT id, componente_id, timestamp, temperatura, umidade,
                   corrente, voltagem, rpm, vibracao, inclinacao
            FROM leitura_sensor
            WHERE componente_id = :cid AND timestamp >= :since
            ORDER BY timestamp DESC LIMIT :lim
        """
    else:
        sql = """
            SELECT id, componente_id, timestamp, temperatura, umidade,
                   corrente, voltagem, rpm, vibracao, inclinacao
            FROM leitura_sensor
            WHERE componente_id = :cid
            ORDER BY timestamp DESC LIMIT :lim
        """
    rows = db.execute(text(sql), params).fetchall()
    return [_row_to_reading(r) for r in rows]
