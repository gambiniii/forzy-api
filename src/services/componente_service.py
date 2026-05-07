from datetime import date
from typing import Optional

from fastapi import HTTPException
from sqlalchemy.orm import Session

from src.models.componente import Componente, EspecificacaoMotor
from src.models.enums import StatusEnum
from src.models.maquina import Maquina


def list_componentes(db: Session, maquina_id: Optional[int]) -> list[Componente]:
    query = db.query(Componente)
    if maquina_id:
        query = query.filter(Componente.maquina_id == maquina_id)
    return query.all()


def get_componente(db: Session, componente_id: int) -> Componente:
    comp = db.query(Componente).filter(Componente.id == componente_id).first()
    if not comp:
        raise HTTPException(status_code=404, detail="Componente não encontrado")
    return comp


def create_componente(db: Session, maquina_id: int, nome: str, tipo: Optional[str],
                      fabricante: Optional[str], data_instalacao: Optional[date],
                      status: StatusEnum,
                      especificacao_motor: Optional[dict]) -> Componente:
    if not db.query(Maquina).filter(Maquina.id == maquina_id).first():
        raise HTTPException(status_code=404, detail="Máquina não encontrada")

    comp = Componente(
        maquina_id=maquina_id,
        nome=nome,
        tipo=tipo,
        fabricante=fabricante,
        data_instalacao=data_instalacao,
        status=status,
    )
    db.add(comp)
    db.flush()

    if especificacao_motor:
        db.add(EspecificacaoMotor(componente_id=comp.id, **especificacao_motor))

    db.commit()
    db.refresh(comp)
    return comp


def update_componente(db: Session, componente_id: int, fields: dict) -> Componente:
    comp = get_componente(db, componente_id)
    for k, v in fields.items():
        setattr(comp, k, v)
    db.commit()
    db.refresh(comp)
    return comp


def upsert_especificacao_motor(db: Session, componente_id: int, fields: dict) -> EspecificacaoMotor:
    get_componente(db, componente_id)  # valida existência
    esp = db.query(EspecificacaoMotor).filter(EspecificacaoMotor.componente_id == componente_id).first()
    if esp:
        for k, v in fields.items():
            setattr(esp, k, v)
    else:
        esp = EspecificacaoMotor(componente_id=componente_id, **fields)
        db.add(esp)
    db.commit()
    db.refresh(esp)
    return esp


def delete_componente(db: Session, componente_id: int) -> None:
    comp = get_componente(db, componente_id)
    db.delete(comp)
    db.commit()
