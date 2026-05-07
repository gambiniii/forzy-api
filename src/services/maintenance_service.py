from datetime import datetime
from typing import Optional

from fastapi import HTTPException
from sqlalchemy.orm import Session

from src.models.enums import MaintenanceTypeEnum
from src.models.maintenance import Maintenance


def list_maintenance(db: Session, machine_id: Optional[int]) -> list[Maintenance]:
    query = db.query(Maintenance)
    if machine_id:
        query = query.filter(Maintenance.machine_id == machine_id)
    return query.order_by(Maintenance.scheduled_at).all()


def create_maintenance(db: Session, machine_id: int, type: MaintenanceTypeEnum,
                       scheduled_at: datetime, notes: Optional[str]) -> Maintenance:
    m = Maintenance(machine_id=machine_id, type=type, scheduled_at=scheduled_at, notes=notes)
    db.add(m)
    db.commit()
    db.refresh(m)
    return m


def update_maintenance(db: Session, maintenance_id: int, fields: dict) -> Maintenance:
    m = db.query(Maintenance).filter(Maintenance.id == maintenance_id).first()
    if not m:
        raise HTTPException(status_code=404, detail="Registro não encontrado")
    for k, v in fields.items():
        setattr(m, k, v)
    db.commit()
    db.refresh(m)
    return m
