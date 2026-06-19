from typing import Optional
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class PlantaCreate(BaseModel):
    nome: str
    localizacao: Optional[str] = None
    cidade: Optional[str] = None
    estado: Optional[str] = None
    ativo: bool = True


class PlantaUpdate(BaseModel):
    nome: Optional[str] = None
    localizacao: Optional[str] = None
    cidade: Optional[str] = None
    estado: Optional[str] = None
    ativo: Optional[bool] = None


def _register(r: APIRouter):
    from src.controllers.planta_controller import (
        ctrl_list_plantas, ctrl_get_planta, ctrl_create_planta,
        ctrl_update_planta, ctrl_delete_planta,
    )
    r.get("/",              response_model=None)(ctrl_list_plantas)
    r.get("/{planta_id}",  response_model=None)(ctrl_get_planta)
    r.post("/",             response_model=None, status_code=201)(ctrl_create_planta)
    r.patch("/{planta_id}", response_model=None)(ctrl_update_planta)
    r.delete("/{planta_id}", status_code=204)(ctrl_delete_planta)


_register(router)
