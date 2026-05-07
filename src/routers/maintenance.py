from datetime import datetime
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

from src.models.enums import MaintenanceTypeEnum

router = APIRouter()


class MaintenanceCreate(BaseModel):
    machine_id: int
    type: MaintenanceTypeEnum
    scheduled_at: datetime
    notes: Optional[str] = None


class MaintenanceUpdate(BaseModel):
    scheduled_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    notes: Optional[str] = None


def _register(r: APIRouter):
    from src.controllers.maintenance_controller import (
        ctrl_list_maintenance, ctrl_create_maintenance, ctrl_update_maintenance,
    )
    r.get("/",                   response_model=None)(ctrl_list_maintenance)
    r.post("/",                  response_model=None, status_code=201)(ctrl_create_maintenance)
    r.patch("/{maintenance_id}", response_model=None)(ctrl_update_maintenance)


_register(router)
