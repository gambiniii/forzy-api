from typing import Optional

from fastapi import HTTPException
from sqlalchemy.orm import Session, joinedload

from src.models.atributo import Atributo, ComponenteAtributoValor
from src.models.componente import Componente


def list_atributos(db: Session) -> list[Atributo]:
    return db.query(Atributo).all()


def create_atributo(db: Session, nome: str, unidade: Optional[str], tipo_dado: Optional[str]) -> Atributo:
    atrib = Atributo(nome=nome, unidade=unidade, tipo_dado=tipo_dado)
    db.add(atrib)
    db.commit()
    db.refresh(atrib)
    return atrib


def delete_atributo(db: Session, atributo_id: int) -> None:
    atrib = db.query(Atributo).filter(Atributo.id == atributo_id).first()
    if not atrib:
        raise HTTPException(status_code=404, detail="Atributo não encontrado")
    db.delete(atrib)
    db.commit()


def list_valores(db: Session, componente_id: Optional[int]) -> list[ComponenteAtributoValor]:
    query = db.query(ComponenteAtributoValor).options(joinedload(ComponenteAtributoValor.atributo))
    if componente_id:
        query = query.filter(ComponenteAtributoValor.componente_id == componente_id)
    return query.all()


def create_valor(db: Session, componente_id: int, atributo_id: int,
                 valor_float: Optional[float], valor_int: Optional[int],
                 valor_string: Optional[str]) -> ComponenteAtributoValor:
    if not db.query(Componente).filter(Componente.id == componente_id).first():
        raise HTTPException(status_code=404, detail="Componente não encontrado")
    if not db.query(Atributo).filter(Atributo.id == atributo_id).first():
        raise HTTPException(status_code=404, detail="Atributo não encontrado")

    valor = ComponenteAtributoValor(
        componente_id=componente_id,
        atributo_id=atributo_id,
        valor_float=valor_float,
        valor_int=valor_int,
        valor_string=valor_string,
    )
    db.add(valor)
    db.commit()
    db.refresh(valor)
    return valor


def update_valor(db: Session, valor_id: int, fields: dict) -> ComponenteAtributoValor:
    valor = db.query(ComponenteAtributoValor).filter(ComponenteAtributoValor.id == valor_id).first()
    if not valor:
        raise HTTPException(status_code=404, detail="Valor não encontrado")
    for k, v in fields.items():
        setattr(valor, k, v)
    db.commit()
    db.refresh(valor)
    return valor


def delete_valor(db: Session, valor_id: int) -> None:
    valor = db.query(ComponenteAtributoValor).filter(ComponenteAtributoValor.id == valor_id).first()
    if not valor:
        raise HTTPException(status_code=404, detail="Valor não encontrado")
    db.delete(valor)
    db.commit()
