from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.db.postgres import get_db
from src.models.enums import MaintenanceTypeEnum, UserRoleEnum
from src.routers.auth import get_current_user, require_role

router = APIRouter()


class MaintenanceOut(BaseModel):
    id: int
    motor_id: int
    type: str
    scheduled_at: Optional[str] = None
    completed_at: Optional[str] = None
    notes: Optional[str] = None
    created_at: Optional[str] = None


class MaintenanceCreate(BaseModel):
    motor_id: int
    type: MaintenanceTypeEnum
    scheduled_at: datetime
    notes: Optional[str] = None


class MaintenanceUpdate(BaseModel):
    scheduled_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    notes: Optional[str] = None


def _row_to_maintenance(r) -> MaintenanceOut:
    return MaintenanceOut(
        id=r[0], motor_id=r[1], type=str(r[2]),
        scheduled_at=r[3].isoformat() if r[3] else None,
        completed_at=r[4].isoformat() if r[4] else None,
        notes=r[5],
        created_at=r[6].isoformat() if r[6] else None,
    )


@router.get("/", response_model=list[MaintenanceOut])
def list_maintenance(
    motor_id: Optional[int] = Query(default=None),
    db: Session = Depends(get_db),
    _=Depends(get_current_user),
):
    if motor_id is not None:
        rows = db.execute(text("""
            SELECT id, motor_id, type, scheduled_at, completed_at, notes, created_at
            FROM maintenance WHERE motor_id = :mid ORDER BY scheduled_at
        """), {"mid": motor_id}).fetchall()
    else:
        rows = db.execute(text("""
            SELECT id, motor_id, type, scheduled_at, completed_at, notes, created_at
            FROM maintenance ORDER BY scheduled_at
        """)).fetchall()
    return [_row_to_maintenance(r) for r in rows]


@router.post("/", response_model=MaintenanceOut, status_code=201)
def create_maintenance(
    data: MaintenanceCreate,
    db: Session = Depends(get_db),
    _=Depends(require_role(UserRoleEnum.admin, UserRoleEnum.technician)),
):
    row = db.execute(text("""
        INSERT INTO maintenance (motor_id, type, scheduled_at, notes)
        VALUES (:motor_id, :type, :scheduled_at, :notes)
        RETURNING id, motor_id, type, scheduled_at, completed_at, notes, created_at
    """), data.model_dump()).fetchone()
    db.commit()
    return _row_to_maintenance(row)


@router.patch("/{maintenance_id}", response_model=MaintenanceOut)
def update_maintenance(
    maintenance_id: int,
    data: MaintenanceUpdate,
    db: Session = Depends(get_db),
    _=Depends(require_role(UserRoleEnum.admin, UserRoleEnum.technician)),
):
    exists = db.execute(text("SELECT id FROM maintenance WHERE id = :id"), {"id": maintenance_id}).fetchone()
    if not exists:
        raise HTTPException(status_code=404, detail="Registro não encontrado")
    fields = data.model_dump(exclude_none=True)
    if not fields:
        raise HTTPException(status_code=400, detail="Nenhum campo para atualizar")
    set_clause = ", ".join(f"{k} = :{k}" for k in fields)
    fields["id"] = maintenance_id
    row = db.execute(text(f"""
        UPDATE maintenance SET {set_clause} WHERE id = :id
        RETURNING id, motor_id, type, scheduled_at, completed_at, notes, created_at
    """), fields).fetchone()
    db.commit()
    return _row_to_maintenance(row)
