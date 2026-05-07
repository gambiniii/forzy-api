from datetime import datetime, timezone
from typing import Optional

from fastapi import HTTPException
from sqlalchemy.orm import Session

from src.models.alert import Alert
from src.models.enums import SeverityEnum


def list_alerts(db: Session, machine_id: Optional[int], resolved: Optional[bool]) -> list[Alert]:
    query = db.query(Alert)
    if machine_id:
        query = query.filter(Alert.machine_id == machine_id)
    if resolved is True:
        query = query.filter(Alert.resolved_at.isnot(None))
    elif resolved is False:
        query = query.filter(Alert.resolved_at.is_(None))
    return query.order_by(Alert.created_at.desc()).all()


def create_alert(db: Session, machine_id: int, severity: SeverityEnum, message: str,
                 anomaly_score: Optional[float], rul_estimated: Optional[float]) -> Alert:
    alert = Alert(
        machine_id=machine_id,
        severity=severity,
        message=message,
        anomaly_score=anomaly_score,
        rul_estimated=rul_estimated,
    )
    db.add(alert)
    db.commit()
    db.refresh(alert)
    return alert


def resolve_alert(db: Session, alert_id: int) -> Alert:
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=404, detail="Alerta não encontrado")
    if alert.resolved_at:
        raise HTTPException(status_code=409, detail="Alerta já foi resolvido")
    alert.resolved_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(alert)
    return alert
