from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.db.postgres import get_db
from src.routers.auth import get_current_user, require_role
from src.models.enums import UserRoleEnum

router = APIRouter()


class ComponentOut(BaseModel):
    id: int
    motor_id: int
    name: str
    type: Optional[str] = None
    fabricante: Optional[str] = None
    status: str
    created_at: Optional[str] = None


class ComponentCreate(BaseModel):
    motor_id: int
    name: str
    type: Optional[str] = None
    fabricante: Optional[str] = None
    status: str = "active"


def _row_to_out(r) -> ComponentOut:
    return ComponentOut(
        id=r[0], motor_id=r[1], name=r[2], type=r[3],
        fabricante=r[4], status=str(r[5]),
        created_at=r[6].isoformat() if r[6] else None,
    )


@router.get("/", response_model=list[ComponentOut])
def list_components(
    motor_id: Optional[int] = Query(default=None),
    db: Session = Depends(get_db),
    _=Depends(get_current_user),
):
    if motor_id is not None:
        rows = db.execute(text(
            "SELECT id, motor_id, name, type, fabricante, status, created_at FROM components WHERE motor_id = :mid ORDER BY id"
        ), {"mid": motor_id}).fetchall()
    else:
        rows = db.execute(text(
            "SELECT id, motor_id, name, type, fabricante, status, created_at FROM components ORDER BY id"
        )).fetchall()
    return [_row_to_out(r) for r in rows]


@router.get("/{component_id}", response_model=ComponentOut)
def get_component(
    component_id: int,
    db: Session = Depends(get_db),
    _=Depends(get_current_user),
):
    row = db.execute(text(
        "SELECT id, motor_id, name, type, fabricante, status, created_at FROM components WHERE id = :id"
    ), {"id": component_id}).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Componente não encontrado")
    return _row_to_out(row)


@router.post("/", response_model=ComponentOut, status_code=201)
def create_component(
    data: ComponentCreate,
    db: Session = Depends(get_db),
    _=Depends(require_role(UserRoleEnum.admin, UserRoleEnum.technician)),
):
    row = db.execute(text("""
        INSERT INTO components (motor_id, name, type, fabricante, status)
        VALUES (:motor_id, :name, :type, :fabricante, :status)
        RETURNING id, motor_id, name, type, fabricante, status, created_at
    """), data.model_dump()).fetchone()
    db.commit()
    return _row_to_out(row)
