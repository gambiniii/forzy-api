from typing import Optional

from fastapi import Depends, Query

from src.routers.auth import get_current_user
from src.services import aneel_service


async def ctrl_get_conformidade_tensao(
    limit: int = Query(default=100, le=1000),
    offset: int = Query(default=0),
    distribuidora: Optional[str] = Query(default=None),
    _=Depends(get_current_user),
) -> dict:
    return await aneel_service.fetch_conformidade_tensao(limit, offset, distribuidora)


async def ctrl_contexto_tensao_componente(
    componente_id: int,
    _=Depends(get_current_user),
) -> dict:
    return await aneel_service.contexto_tensao_componente(componente_id)
