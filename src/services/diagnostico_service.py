from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session

from src.models.diagnostico import Diagnostico


def save_diagnostico(
    db: Session,
    componente_id: int,
    overall_status: str,
    is_anomaly: bool = False,
    lstm_severity: Optional[str] = None,
    risk_level: Optional[str] = None,
    rul_hours: Optional[float] = None,
    maintenance_window_days: Optional[float] = None,
    health_score: Optional[float] = None,
    health_index: Optional[float] = None,
    recommendation: Optional[str] = None,
) -> Diagnostico:
    diagnostico = Diagnostico(
        componente_id=componente_id,
        overall_status=overall_status,
        is_anomaly=is_anomaly,
        lstm_severity=lstm_severity,
        risk_level=risk_level,
        rul_hours=rul_hours,
        maintenance_window_days=maintenance_window_days,
        health_score=health_score,
        health_index=health_index,
        recommendation=recommendation,
    )
    db.add(diagnostico)
    db.commit()
    db.refresh(diagnostico)
    return diagnostico


def list_diagnosticos(
    db: Session,
    componente_id: int,
    limit: int = 50,
    inicio: Optional[datetime] = None,
    fim: Optional[datetime] = None,
    apenas_anomalias: bool = False,
) -> list[Diagnostico]:
    q = db.query(Diagnostico).filter(Diagnostico.componente_id == componente_id)
    if inicio:
        q = q.filter(Diagnostico.timestamp >= inicio)
    if fim:
        q = q.filter(Diagnostico.timestamp <= fim)
    if apenas_anomalias:
        q = q.filter(Diagnostico.is_anomaly == True)
    return q.order_by(Diagnostico.timestamp.desc()).limit(limit).all()
