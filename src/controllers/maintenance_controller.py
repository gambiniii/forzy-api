from typing import Optional

from fastapi import Depends, Query
from sqlalchemy.orm import Session

from src.db.postgres import get_db
from src.models.enums import UserRoleEnum
from src.models.maintenance import Maintenance
from src.routers.auth import get_current_user, require_role
from src.routers.maintenance import MaintenanceCreate, MaintenanceUpdate
from src.services import maintenance_service


def ctrl_list_maintenance(
    machine_id: Optional[int] = Query(default=None),
    db: Session = Depends(get_db),
    _=Depends(get_current_user),
) -> list[Maintenance]:
    return maintenance_service.list_maintenance(db, machine_id)


def ctrl_create_maintenance(
    data: MaintenanceCreate,
    db: Session = Depends(get_db),
    _=Depends(require_role(UserRoleEnum.admin, UserRoleEnum.technician)),
) -> Maintenance:
    return maintenance_service.create_maintenance(db, **data.model_dump())


def ctrl_update_maintenance(
    maintenance_id: int,
    data: MaintenanceUpdate,
    db: Session = Depends(get_db),
    _=Depends(require_role(UserRoleEnum.admin, UserRoleEnum.technician)),
) -> Maintenance:
    return maintenance_service.update_maintenance(db, maintenance_id, data.model_dump(exclude_none=True))
