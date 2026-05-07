from typing import Optional

import httpx
from fastapi import HTTPException

from src.config import settings

ANEEL_API_URL = "https://dadosabertos.aneel.gov.br/api/3/action/datastore_search"


async def fetch_conformidade_tensao(limit: int, offset: int,
                                    distribuidora: Optional[str]) -> dict:
    params: dict = {
        "resource_id": settings.ANEEL_RESOURCE_ID,
        "limit": limit,
        "offset": offset,
    }
    if distribuidora:
        params["q"] = distribuidora

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(ANEEL_API_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Erro ao acessar ANEEL: {e}")

    if not data.get("success"):
        raise HTTPException(status_code=502, detail="ANEEL API retornou erro")

    result = data["result"]
    return {
        "total": result.get("total"),
        "offset": offset,
        "limit": limit,
        "records": result.get("records", []),
        "source": "ANEEL — Indicadores de Conformidade do Nível de Tensão em Regime Permanente",
    }


async def contexto_tensao_componente(componente_id: int) -> dict:
    """
    Cruza a última leitura de voltagem do componente com os indicadores ANEEL
    para determinar se a variação é problema de rede ou de equipamento.
    TODO: implementar cruzamento real quando a lógica de negócio for definida.
    """
    return {
        "componente_id": componente_id,
        "status": "em desenvolvimento",
        "planned_output": {
            "voltage_deviation_pct": None,
            "aneel_conformity": None,
            "source_of_issue": None,   # "grid" | "equipment" | "undetermined"
            "estimated_cost_impact": None,
        },
    }
