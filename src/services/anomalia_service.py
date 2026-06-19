from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session

from src.models.anomalia import Anomalia


def save_anomalia(
    db: Session,
    componente_id: int,
    overall_status: str,
    lstm_severity: Optional[str] = None,
    risk_level: Optional[str] = None,
    rul_hours: Optional[float] = None,
    maintenance_window_days: Optional[float] = None,
    health_score: Optional[float] = None,
    recommendation: Optional[str] = None,
) -> Anomalia:
    anomalia = Anomalia(
        componente_id=componente_id,
        overall_status=overall_status,
        lstm_severity=lstm_severity,
        risk_level=risk_level,
        rul_hours=rul_hours,
        maintenance_window_days=maintenance_window_days,
        health_score=health_score,
        recommendation=recommendation,
    )
    db.add(anomalia)
    db.commit()
    db.refresh(anomalia)
    return anomalia


def list_anomalias(
    db: Session,
    componente_id: int,
    limit: int = 50,
    inicio: Optional[datetime] = None,
    fim: Optional[datetime] = None,
) -> list[Anomalia]:
    q = db.query(Anomalia).filter(Anomalia.componente_id == componente_id)
    if inicio:
        q = q.filter(Anomalia.timestamp >= inicio)
    if fim:
        q = q.filter(Anomalia.timestamp <= fim)
    return q.order_by(Anomalia.timestamp.desc()).limit(limit).all()
