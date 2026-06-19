from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

from src.models.enums import StatusEnum

router = APIRouter()


class MaquinaCreate(BaseModel):
    nome: str
    tipo: Optional[str] = None
    fabricante: Optional[str] = None
    ano_instalacao: Optional[int] = None
    status: StatusEnum = StatusEnum.active
    planta_id: int


class MaquinaUpdate(BaseModel):
    nome: Optional[str] = None
    tipo: Optional[str] = None
    fabricante: Optional[str] = None
    ano_instalacao: Optional[int] = None
    status: Optional[StatusEnum] = None
    planta_id: Optional[int] = None


def _register(r: APIRouter):
    from src.controllers.maquina_controller import (
        ctrl_list_maquinas, ctrl_get_maquina, ctrl_create_maquina,
        ctrl_update_maquina, ctrl_delete_maquina,
    )
    r.get("/",               response_model=None)(ctrl_list_maquinas)
    r.get("/{maquina_id}",  response_model=None)(ctrl_get_maquina)
    r.post("/",              response_model=None, status_code=201)(ctrl_create_maquina)
    r.patch("/{maquina_id}", response_model=None)(ctrl_update_maquina)
    r.delete("/{maquina_id}", status_code=204)(ctrl_delete_maquina)


_register(router)
