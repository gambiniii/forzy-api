from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.db.postgres import SessionLocal
from src.routers.auth import get_current_user
from src.services import anomalia_service

router = APIRouter()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class AnomaliaOut(BaseModel):
    id: int
    componente_id: int
    timestamp: datetime
    overall_status: str
    lstm_severity: Optional[str]
    risk_level: Optional[str]
    rul_hours: Optional[float]
    maintenance_window_days: Optional[float]
    health_score: Optional[float]
    recommendation: Optional[str]

    class Config:
        from_attributes = True


@router.get("/componente/{componente_id}", response_model=list[AnomaliaOut])
def get_anomalias(
    componente_id: int,
    limit: int = Query(50, le=200),
    inicio: Optional[datetime] = Query(None),
    fim: Optional[datetime] = Query(None),
    _=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return anomalia_service.list_anomalias(db, componente_id, limit, inicio, fim)
