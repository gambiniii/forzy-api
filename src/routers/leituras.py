from datetime import datetime
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class LeituraCreate(BaseModel):
    componente_id: int
    timestamp: Optional[datetime] = None
    temperatura: Optional[float] = None
    umidade: Optional[float] = None
    corrente: Optional[float] = None
    voltagem: Optional[float] = None
    rpm: Optional[float] = None
    vibracao: Optional[float] = None
    inclinacao: Optional[float] = None


def _register(r: APIRouter):
    from src.controllers.leitura_controller import (
        ctrl_ingest_leitura, ctrl_get_leituras, ctrl_get_ultima_leitura,
    )
    r.post("/",                                      response_model=None, status_code=201)(ctrl_ingest_leitura)
    r.get("/componente/{componente_id}",             response_model=None)(ctrl_get_leituras)
    r.get("/componente/{componente_id}/ultima",      response_model=None)(ctrl_get_ultima_leitura)


_register(router)
