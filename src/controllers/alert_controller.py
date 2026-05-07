from typing import Optional

from fastapi import Depends, Query
from sqlalchemy.orm import Session

from src.db.postgres import get_db
from src.models.alert import Alert
from src.models.enums import UserRoleEnum
from src.routers.alerts import AlertCreate
from src.routers.auth import get_current_user, require_role
from src.services import alert_service


def ctrl_list_alerts(
    machine_id: Optional[int] = Query(default=None),
    resolved: Optional[bool] = Query(default=None),
    db: Session = Depends(get_db),
    _=Depends(get_current_user),
) -> list[Alert]:
    return alert_service.list_alerts(db, machine_id, resolved)


def ctrl_create_alert(
    data: AlertCreate,
    db: Session = Depends(get_db),
    _=Depends(require_role(UserRoleEnum.admin, UserRoleEnum.technician)),
) -> Alert:
    return alert_service.create_alert(db, **data.model_dump())


def ctrl_resolve_alert(
    alert_id: int,
    db: Session = Depends(get_db),
    _=Depends(require_role(UserRoleEnum.admin, UserRoleEnum.technician)),
) -> Alert:
    return alert_service.resolve_alert(db, alert_id)
