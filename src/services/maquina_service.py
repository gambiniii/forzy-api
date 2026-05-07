from typing import Optional

from fastapi import HTTPException
from sqlalchemy.orm import Session

from src.models.enums import StatusEnum
from src.models.maquina import Maquina


def list_maquinas(db: Session) -> list[Maquina]:
    return db.query(Maquina).all()


def get_maquina(db: Session, maquina_id: int) -> Maquina:
    maquina = db.query(Maquina).filter(Maquina.id == maquina_id).first()
    if not maquina:
        raise HTTPException(status_code=404, detail="Máquina não encontrada")
    return maquina


def create_maquina(db: Session, nome: str, tipo: Optional[str], fabricante: Optional[str],
                   ano_instalacao: Optional[int], status: StatusEnum,
                   localizacao: Optional[str]) -> Maquina:
    maquina = Maquina(
        nome=nome,
        tipo=tipo,
        fabricante=fabricante,
        ano_instalacao=ano_instalacao,
        status=status,
        localizacao=localizacao,
    )
    db.add(maquina)
    db.commit()
    db.refresh(maquina)
    return maquina


def update_maquina(db: Session, maquina_id: int, fields: dict) -> Maquina:
    maquina = get_maquina(db, maquina_id)
    for k, v in fields.items():
        setattr(maquina, k, v)
    db.commit()
    db.refresh(maquina)
    return maquina


def delete_maquina(db: Session, maquina_id: int) -> None:
    maquina = get_maquina(db, maquina_id)
    db.delete(maquina)
    db.commit()
