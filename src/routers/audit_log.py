from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.db.postgres import get_db
from src.models.audit_log import AuditLog
from src.models.enums import UserRoleEnum
from src.models.user import User
from src.routers.auth import get_current_user, require_role
from src.services import audit_log_service

router = APIRouter()


class AuditLogCreate(BaseModel):
    action: str
    target: str
    justification: Optional[str] = None


@router.get("/", response_model=None)
def list_audit_log(
    target: Optional[str] = None,
    db: Session = Depends(get_db),
    _=Depends(get_current_user),
) -> list[AuditLog]:
    return audit_log_service.list_logs(db, target)


@router.post("/", response_model=None, status_code=201)
def create_audit_log(
    data: AuditLogCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(UserRoleEnum.admin, UserRoleEnum.technician)),
) -> AuditLog:
    return audit_log_service.create_log(
        db,
        user_id=current_user.id,
        action=data.action,
        target=data.target,
        justification=data.justification,
    )
