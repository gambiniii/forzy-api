"""Controle do modo demonstração (ver src/services/sensor_demo.py) — uso do
apresentador via terminal/Postman durante ensaio e apresentação, não faz
parte da UI de produção (não faria sentido um operador real ver um botão
"disparar falha")."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from src.routers.auth import get_current_user
from src.services import sensor_demo

router = APIRouter()

MODOS_VALIDOS = {"F1", "F2", "F3", "F4", "F5", "F6"}


class FalhaIn(BaseModel):
    componente_id: int
    modo: str
    severidade: str = "moderada"


@router.post("/falha")
def disparar_falha(body: FalhaIn, _=Depends(get_current_user)):
    if body.modo not in MODOS_VALIDOS:
        raise HTTPException(status_code=400, detail=f"modo inválido — use um de {sorted(MODOS_VALIDOS)}")
    if body.severidade not in sensor_demo.SEVERIDADES:
        raise HTTPException(status_code=400, detail=f"severidade inválida — use uma de {sorted(sensor_demo.SEVERIDADES)}")
    sensor_demo.ativar_falha(body.componente_id, body.modo, body.severidade)
    return {"ok": True, "componente_id": body.componente_id, "modo": body.modo, "severidade": body.severidade}


class ResetIn(BaseModel):
    componente_id: int


@router.post("/reset")
def resetar(body: ResetIn, _=Depends(get_current_user)):
    sensor_demo.resetar_falha(body.componente_id)
    return {"ok": True, "componente_id": body.componente_id}


@router.get("/status")
def status(componente_id: Optional[int] = None, _=Depends(get_current_user)):
    ids = [componente_id] if componente_id is not None else [sensor_demo.COMPONENTE_ID_S1, sensor_demo.COMPONENTE_ID_S2]
    return {cid: sensor_demo.status_falha(cid) for cid in ids}
