from typing import Optional

from fastapi import HTTPException
from sqlalchemy.orm import Session

from src.models.componente import Componente, ComponenteLimite

DEFAULTS = dict(
    vib_atencao=1.8, vib_critico=4.5,      # mm/s — velocidade de vibração (leitura_sensor.rpm)
    temp_atencao=70.0, temp_critico=90.0,  # °C
    acel_atencao=0.63, acel_critico=0.78,  # g — aceleração (leitura_sensor.vibracao)
)


def get_or_create_limites(db: Session, componente_id: int) -> ComponenteLimite:
    """Retorna os limites do componente, criando um registro com os
    defaults do Metric Contract (ISO 10816-1 Classe I) se ainda não existir."""
    if not db.query(Componente).filter(Componente.id == componente_id).first():
        raise HTTPException(status_code=404, detail="Componente não encontrado")

    limite = db.query(ComponenteLimite).filter(ComponenteLimite.componente_id == componente_id).first()
    if not limite:
        limite = ComponenteLimite(componente_id=componente_id, **DEFAULTS)
        db.add(limite)
        db.commit()
        db.refresh(limite)
    return limite


def upsert_limites(
    db: Session,
    componente_id: int,
    vib_atencao: Optional[float] = None,
    vib_critico: Optional[float] = None,
    temp_atencao: Optional[float] = None,
    temp_critico: Optional[float] = None,
    acel_atencao: Optional[float] = None,
    acel_critico: Optional[float] = None,
) -> ComponenteLimite:
    limite = get_or_create_limites(db, componente_id)
    if vib_atencao is not None:
        limite.vib_atencao = vib_atencao
    if vib_critico is not None:
        limite.vib_critico = vib_critico
    if temp_atencao is not None:
        limite.temp_atencao = temp_atencao
    if temp_critico is not None:
        limite.temp_critico = temp_critico
    if acel_atencao is not None:
        limite.acel_atencao = acel_atencao
    if acel_critico is not None:
        limite.acel_critico = acel_critico
    db.commit()
    db.refresh(limite)
    return limite
