"""Controle do modo demonstração (ver src/services/sensor_demo.py) — uso do
apresentador via terminal/Postman durante ensaio e apresentação, não faz
parte da UI de produção (não faria sentido um operador real ver um botão
"disparar falha")."""

import asyncio
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from src.config import settings
from src.routers.auth import get_current_user
from src.services import sensor_demo

router = APIRouter()
logger = logging.getLogger("app.demo_control")

MODOS_VALIDOS = {"F1", "F2", "F3", "F4", "F5", "F6"}

# Tempo pro sinal perturbado (rampa de sensor_demo._perturbar) chegar no efeito
# pleno e pra pelo menos um ciclo do replay já ter escrito no banco, antes de
# rodar o diagnóstico automaticamente — ver ativar_falha() abaixo.
ATRASO_DIAGNOSTICO_AUTOMATICO_S = 12.0


class FalhaIn(BaseModel):
    componente_id: int
    modo: str
    severidade: str = "moderada"


async def _rodar_diagnostico_apos_atraso(componente_id: int) -> None:
    await asyncio.sleep(ATRASO_DIAGNOSTICO_AUTOMATICO_S)
    try:
        from src.services.diagnostico_scheduler import _run_once
        await _run_once(componente_id)
        logger.info("componente=%d — diagnóstico automático rodado após ativar falha.", componente_id)
    except Exception as e:
        logger.warning("componente=%d — diagnóstico automático falhou: %s", componente_id, e)


@router.post("/falha")
async def disparar_falha(body: FalhaIn, _=Depends(get_current_user)):
    if settings.SENSOR_MODE != "demo":
        raise HTTPException(
            status_code=409,
            detail=(
                "SENSOR_MODE não está em 'demo' NESTA instância — a falha ficaria só na memória, "
                "sem nenhum efeito, porque só o demo_loop() aplica a perturbação antes de gravar. "
                "Defina SENSOR_MODE=demo no .env desta API e reinicie o processo antes de tentar de novo."
            ),
        )
    if body.modo not in MODOS_VALIDOS:
        raise HTTPException(status_code=400, detail=f"modo inválido — use um de {sorted(MODOS_VALIDOS)}")
    if body.severidade not in sensor_demo.SEVERIDADES:
        raise HTTPException(status_code=400, detail=f"severidade inválida — use uma de {sorted(sensor_demo.SEVERIDADES)}")
    sensor_demo.ativar_falha(body.componente_id, body.modo, body.severidade)
    asyncio.create_task(_rodar_diagnostico_apos_atraso(body.componente_id))
    return {
        "ok": True, "componente_id": body.componente_id, "modo": body.modo, "severidade": body.severidade,
        "aviso": f"Diagnóstico automático em ~{ATRASO_DIAGNOSTICO_AUTOMATICO_S:.0f}s — "
                 "ou chame /diagnosticos/componente/{id}/rodar-agora você mesmo a qualquer momento.",
    }


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
