from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class AtributoCreate(BaseModel):
    nome: str
    unidade: Optional[str] = None
    tipo_dado: Optional[str] = None  # float | int | string


class ValorCreate(BaseModel):
    componente_id: int
    atributo_id: int
    valor_float: Optional[float] = None
    valor_int: Optional[int] = None
    valor_string: Optional[str] = None


class ValorUpdate(BaseModel):
    valor_float: Optional[float] = None
    valor_int: Optional[int] = None
    valor_string: Optional[str] = None


def _register(r: APIRouter):
    from src.controllers.atributo_controller import (
        ctrl_list_atributos, ctrl_create_atributo, ctrl_delete_atributo,
        ctrl_list_valores, ctrl_create_valor, ctrl_update_valor, ctrl_delete_valor,
    )
    r.get("/",                    response_model=None)(ctrl_list_atributos)
    r.post("/",                   response_model=None, status_code=201)(ctrl_create_atributo)
    r.delete("/{atributo_id}",    status_code=204)(ctrl_delete_atributo)
    r.get("/valores",             response_model=None)(ctrl_list_valores)
    r.post("/valores",            response_model=None, status_code=201)(ctrl_create_valor)
    r.patch("/valores/{valor_id}", response_model=None)(ctrl_update_valor)
    r.delete("/valores/{valor_id}", status_code=204)(ctrl_delete_valor)


_register(router)
