from fastapi import APIRouter

router = APIRouter()


def _register(r: APIRouter):
    from src.controllers.aneel_controller import (
        ctrl_get_conformidade_tensao, ctrl_contexto_tensao_componente,
    )
    r.get("/conformidade-tensao")(ctrl_get_conformidade_tensao)
    r.get("/conformidade-tensao/contexto/{componente_id}")(ctrl_contexto_tensao_componente)


_register(router)
