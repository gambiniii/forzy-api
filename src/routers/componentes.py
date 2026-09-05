from datetime import date
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

from src.models.enums import StatusEnum

router = APIRouter()


class EspecificacaoMotorSchema(BaseModel):
    potencia_kw: Optional[int] = None
    tensao_nominal: Optional[int] = None
    corrente_nominal: Optional[int] = None
    rpm_nominal: Optional[int] = None
    frequencia_hz: Optional[int] = None
    numero_polos: Optional[int] = None
    rendimento: Optional[int] = None


class ComponenteCreate(BaseModel):
    maquina_id: int
    nome: str
    tipo: Optional[str] = None
    fabricante: Optional[str] = None
    data_instalacao: Optional[date] = None
    status: StatusEnum = StatusEnum.active
    especificacao_motor: Optional[EspecificacaoMotorSchema] = None


class ComponenteUpdate(BaseModel):
    nome: Optional[str] = None
    tipo: Optional[str] = None
    fabricante: Optional[str] = None
    data_instalacao: Optional[date] = None
    status: Optional[StatusEnum] = None


class LimitesSchema(BaseModel):
    vib_atencao: Optional[float] = None
    vib_critico: Optional[float] = None
    temp_atencao: Optional[float] = None
    temp_critico: Optional[float] = None


def _register(r: APIRouter):
    from src.controllers.componente_controller import (
        ctrl_list_componentes, ctrl_get_componente, ctrl_create_componente,
        ctrl_update_componente, ctrl_upsert_especificacao_motor, ctrl_delete_componente,
        ctrl_get_limites, ctrl_upsert_limites,
    )
    r.get("/",                                    response_model=None)(ctrl_list_componentes)
    r.get("/{componente_id}",                     response_model=None)(ctrl_get_componente)
    r.post("/",                                   response_model=None, status_code=201)(ctrl_create_componente)
    r.patch("/{componente_id}",                   response_model=None)(ctrl_update_componente)
    r.put("/{componente_id}/especificacao-motor", response_model=None)(ctrl_upsert_especificacao_motor)
    r.get("/{componente_id}/limites",             response_model=None)(ctrl_get_limites)
    r.put("/{componente_id}/limites",             response_model=None)(ctrl_upsert_limites)
    r.delete("/{componente_id}", status_code=204)(ctrl_delete_componente)


_register(router)
