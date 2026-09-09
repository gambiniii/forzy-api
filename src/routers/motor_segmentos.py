from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.db.postgres import SessionLocal
from src.routers.auth import get_current_user
from src.models.motor_segmento import MotorSegmento

router = APIRouter()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class ModoFalhaOut(BaseModel):
    nome: str
    assinatura: str
    precocidade: str


class MotorSegmentoOut(BaseModel):
    id: str
    label: str
    grupo_funcional: str
    descricao: str
    sinal_velocidade: str
    sinal_aceleracao: str
    sinal_temperatura: str
    componentes_internos: list[str]
    modos_falha: list[ModoFalhaOut]
    destacavel: bool
    inspecao: Optional[str] = None

    class Config:
        from_attributes = True


@router.get("/", response_model=list[MotorSegmentoOut])
def list_motor_segmentos(
    _=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Catálogo completo dos 23 segmentos do modelo 3D — fonte única de verdade,
    espelhada em forzy-web/src/config/motorParts.ts e
    rag_module/tools/motor_parts_tool.py (ver migrations/006_motor_segmento.sql)."""
    return db.query(MotorSegmento).order_by(MotorSegmento.id).all()


@router.get("/{segmento_id}", response_model=MotorSegmentoOut)
def get_motor_segmento(
    segmento_id: str,
    _=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    segmento = db.query(MotorSegmento).filter(MotorSegmento.id == segmento_id).first()
    if segmento is None:
        raise HTTPException(status_code=404, detail="Segmento não encontrado.")
    return segmento
