from typing import Optional
from fastapi import HTTPException
from sqlalchemy.orm import Session
from src.models.planta import Planta


def list_plantas(db: Session) -> list[Planta]:
    return db.query(Planta).order_by(Planta.nome).all()


def get_planta(db: Session, planta_id: int) -> Planta:
    planta = db.query(Planta).filter(Planta.id == planta_id).first()
    if not planta:
        raise HTTPException(status_code=404, detail="Planta não encontrada")
    return planta


def create_planta(db: Session, nome: str, localizacao: Optional[str],
                  cidade: Optional[str], estado: Optional[str], ativo: bool) -> Planta:
    planta = Planta(nome=nome, localizacao=localizacao, cidade=cidade, estado=estado, ativo=ativo)
    db.add(planta)
    db.commit()
    db.refresh(planta)
    return planta


def update_planta(db: Session, planta_id: int, fields: dict) -> Planta:
    planta = get_planta(db, planta_id)
    for k, v in fields.items():
        setattr(planta, k, v)
    db.commit()
    db.refresh(planta)
    return planta


def delete_planta(db: Session, planta_id: int) -> None:
    planta = get_planta(db, planta_id)
    db.delete(planta)
    db.commit()
