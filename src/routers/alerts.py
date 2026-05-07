from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

from src.models.enums import SeverityEnum

router = APIRouter()


class AlertCreate(BaseModel):
    machine_id: int
    severity: SeverityEnum
    message: str
    anomaly_score: Optional[float] = None
    rul_estimated: Optional[float] = None


def _register(r: APIRouter):
    from src.controllers.alert_controller import (
        ctrl_list_alerts, ctrl_create_alert, ctrl_resolve_alert,
    )
    r.get("/",                    response_model=None)(ctrl_list_alerts)
    r.post("/",                   response_model=None, status_code=201)(ctrl_create_alert)
    r.patch("/{alert_id}/resolve", response_model=None)(ctrl_resolve_alert)


_register(router)
