from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.db.postgres import get_db
from src.models.enums import SeverityEnum, UserRoleEnum
from src.routers.auth import get_current_user, require_role

router = APIRouter()


class AlertOut(BaseModel):
    id: int
    motor_id: int
    severity: str
    message: str
    anomaly_score: Optional[float] = None
    rul_estimated: Optional[float] = None
    created_at: Optional[str] = None
    resolved_at: Optional[str] = None


class AlertCreate(BaseModel):
    motor_id: int
    severity: SeverityEnum
    message: str
    anomaly_score: Optional[float] = None
    rul_estimated: Optional[float] = None


def _row_to_alert(r) -> AlertOut:
    return AlertOut(
        id=r[0], motor_id=r[1], severity=str(r[2]), message=r[3],
        anomaly_score=r[4], rul_estimated=r[5],
        created_at=r[6].isoformat() if r[6] else None,
        resolved_at=r[7].isoformat() if r[7] else None,
    )


@router.get("/", response_model=list[AlertOut])
def list_alerts(
    motor_id: Optional[int] = Query(default=None),
    resolved: Optional[bool] = Query(default=None),
    db: Session = Depends(get_db),
    _=Depends(get_current_user),
):
    conditions = []
    params: dict = {}
    if motor_id is not None:
        conditions.append("motor_id = :mid")
        params["mid"] = motor_id
    if resolved is True:
        conditions.append("resolved_at IS NOT NULL")
    elif resolved is False:
        conditions.append("resolved_at IS NULL")
    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    rows = db.execute(text(f"""
        SELECT id, motor_id, severity, message, anomaly_score, rul_estimated, created_at, resolved_at
        FROM alerts {where} ORDER BY created_at DESC
    """), params).fetchall()
    return [_row_to_alert(r) for r in rows]


@router.post("/", response_model=AlertOut, status_code=201)
def create_alert(
    data: AlertCreate,
    db: Session = Depends(get_db),
    _=Depends(require_role(UserRoleEnum.admin, UserRoleEnum.technician)),
):
    row = db.execute(text("""
        INSERT INTO alerts (motor_id, severity, message, anomaly_score, rul_estimated)
        VALUES (:motor_id, :severity, :message, :anomaly_score, :rul_estimated)
        RETURNING id, motor_id, severity, message, anomaly_score, rul_estimated, created_at, resolved_at
    """), data.model_dump()).fetchone()
    db.commit()
    return _row_to_alert(row)


@router.patch("/{alert_id}/resolve", response_model=AlertOut)
def resolve_alert(
    alert_id: int,
    db: Session = Depends(get_db),
    _=Depends(require_role(UserRoleEnum.admin, UserRoleEnum.technician)),
):
    exists = db.execute(text("SELECT id, resolved_at FROM alerts WHERE id = :id"), {"id": alert_id}).fetchone()
    if not exists:
        raise HTTPException(status_code=404, detail="Alerta não encontrado")
    if exists[1] is not None:
        raise HTTPException(status_code=409, detail="Alerta já foi resolvido")
    row = db.execute(text("""
        UPDATE alerts SET resolved_at = :now WHERE id = :id
        RETURNING id, motor_id, severity, message, anomaly_score, rul_estimated, created_at, resolved_at
    """), {"now": datetime.now(timezone.utc), "id": alert_id}).fetchone()
    db.commit()
    return _row_to_alert(row)
