from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.db.postgres import get_db
from src.routers.auth import get_current_user, require_role
from src.models.enums import UserRoleEnum

router = APIRouter()


class MotorOut(BaseModel):
    id: int
    name: str
    type: Optional[str] = None
    serial: Optional[str] = None
    status: str
    ativo_id: int
    nameplate_voltage: Optional[float] = None
    nameplate_current: Optional[float] = None
    nameplate_rpm: Optional[float] = None
    nameplate_power_kw: Optional[float] = None
    nameplate_frequency: Optional[float] = None
    nameplate_cos_phi: Optional[float] = None
    sensor_port1_tag: Optional[str] = None
    sensor_port2_tag: Optional[str] = None
    created_at: Optional[str] = None


class MotorCreate(BaseModel):
    name: str
    type: Optional[str] = None
    serial: Optional[str] = None
    status: str = "active"
    ativo_id: int
    nameplate_voltage: Optional[float] = None
    nameplate_current: Optional[float] = None
    nameplate_rpm: Optional[float] = None
    nameplate_power_kw: Optional[float] = None
    nameplate_frequency: Optional[float] = None
    nameplate_cos_phi: Optional[float] = None
    sensor_port1_tag: Optional[str] = None
    sensor_port2_tag: Optional[str] = None


def _row_to_out(r) -> MotorOut:
    return MotorOut(
        id=r[0], name=r[1], type=r[2], serial=r[3], status=r[4],
        ativo_id=r[5],
        nameplate_voltage=r[6], nameplate_current=r[7], nameplate_rpm=r[8],
        nameplate_power_kw=r[9], nameplate_frequency=r[10], nameplate_cos_phi=r[11],
        sensor_port1_tag=r[12], sensor_port2_tag=r[13],
        created_at=r[14].isoformat() if r[14] else None,
    )


@router.get("/", response_model=list[MotorOut])
def list_motors(db: Session = Depends(get_db), _=Depends(get_current_user)):
    rows = db.execute(text("""
        SELECT id, name, type, serial, status, ativo_id,
               nameplate_voltage, nameplate_current, nameplate_rpm,
               nameplate_power_kw, nameplate_frequency, nameplate_cos_phi,
               sensor_port1_tag, sensor_port2_tag, created_at
        FROM motors ORDER BY id
    """)).fetchall()
    return [_row_to_out(r) for r in rows]


@router.get("/{motor_id}", response_model=MotorOut)
def get_motor(motor_id: int, db: Session = Depends(get_db), _=Depends(get_current_user)):
    row = db.execute(text("""
        SELECT id, name, type, serial, status, ativo_id,
               nameplate_voltage, nameplate_current, nameplate_rpm,
               nameplate_power_kw, nameplate_frequency, nameplate_cos_phi,
               sensor_port1_tag, sensor_port2_tag, created_at
        FROM motors WHERE id = :id
    """), {"id": motor_id}).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Motor não encontrado")
    return _row_to_out(row)


@router.post("/", response_model=MotorOut, status_code=201)
def create_motor(
    data: MotorCreate,
    db: Session = Depends(get_db),
    _=Depends(require_role(UserRoleEnum.admin, UserRoleEnum.technician)),
):
    row = db.execute(text("""
        INSERT INTO motors (name, type, serial, status, ativo_id,
            nameplate_voltage, nameplate_current, nameplate_rpm,
            nameplate_power_kw, nameplate_frequency, nameplate_cos_phi,
            sensor_port1_tag, sensor_port2_tag)
        VALUES (:name, :type, :serial, :status, :ativo_id,
            :nameplate_voltage, :nameplate_current, :nameplate_rpm,
            :nameplate_power_kw, :nameplate_frequency, :nameplate_cos_phi,
            :sensor_port1_tag, :sensor_port2_tag)
        RETURNING id, name, type, serial, status, ativo_id,
            nameplate_voltage, nameplate_current, nameplate_rpm,
            nameplate_power_kw, nameplate_frequency, nameplate_cos_phi,
            sensor_port1_tag, sensor_port2_tag, created_at
    """), data.model_dump()).fetchone()
    db.commit()
    return _row_to_out(row)
