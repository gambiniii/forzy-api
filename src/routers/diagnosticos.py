from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.db.postgres import SessionLocal
from src.routers.auth import get_current_user
from src.services import diagnostico_service

router = APIRouter()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class DiagnosticoOut(BaseModel):
    id: int
    componente_id: int
    timestamp: datetime
    is_anomaly: bool
    overall_status: str
    lstm_severity: Optional[str]
    risk_level: Optional[str]
    rul_hours: Optional[float]
    maintenance_window_days: Optional[float]
    health_score: Optional[float]
    health_index: Optional[float]
    recommendation: Optional[str]

    class Config:
        from_attributes = True


@router.get("/componente/{componente_id}", response_model=list[DiagnosticoOut])
def get_diagnosticos(
    componente_id: int,
    limit: int = Query(50, le=200),
    inicio: Optional[datetime] = Query(None),
    fim: Optional[datetime] = Query(None),
    apenas_anomalias: bool = Query(False),
    _=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return diagnostico_service.list_diagnosticos(
        db, componente_id, limit, inicio, fim, apenas_anomalias
    )
