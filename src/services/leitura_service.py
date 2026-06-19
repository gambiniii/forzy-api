from datetime import datetime
from typing import Optional

from fastapi import HTTPException
from sqlalchemy.orm import Session

from src.models.componente import Componente
from src.models.leitura_sensor import LeituraSensor


def ingest_leitura(db: Session, componente_id: int, timestamp: Optional[datetime] = None,
                   temperatura: Optional[float] = None, umidade: Optional[float] = None,
                   corrente: Optional[float] = None, voltagem: Optional[float] = None,
                   rpm: Optional[float] = None, vibracao: Optional[float] = None,
                   inclinacao: Optional[float] = None) -> LeituraSensor:
    if not db.query(Componente).filter(Componente.id == componente_id).first():
        raise HTTPException(status_code=404, detail="Componente não encontrado")

    data = dict(
        componente_id=componente_id,
        temperatura=temperatura,
        umidade=umidade,
        corrente=corrente,
        voltagem=voltagem,
        rpm=rpm,
        vibracao=vibracao,
        inclinacao=inclinacao,
    )
    if timestamp:
        data["timestamp"] = timestamp

    leitura = LeituraSensor(**{k: v for k, v in data.items() if v is not None or k == "componente_id"})
    db.add(leitura)
    db.commit()
    db.refresh(leitura)
    return leitura


def get_leituras(db: Session, componente_id: int, inicio: Optional[datetime],
                 fim: Optional[datetime], limit: int) -> list[LeituraSensor]:
    query = db.query(LeituraSensor).filter(LeituraSensor.componente_id == componente_id)
    if inicio:
        query = query.filter(LeituraSensor.timestamp >= inicio)
    if fim:
        query = query.filter(LeituraSensor.timestamp <= fim)
    return query.order_by(LeituraSensor.timestamp.desc()).limit(limit).all()


def get_ultima_leitura(db: Session, componente_id: int) -> LeituraSensor:
    leitura = (
        db.query(LeituraSensor)
        .filter(LeituraSensor.componente_id == componente_id)
        .order_by(LeituraSensor.timestamp.desc())
        .first()
    )
    if not leitura:
        raise HTTPException(status_code=404, detail="Nenhuma leitura encontrada para este componente")
    return leitura
