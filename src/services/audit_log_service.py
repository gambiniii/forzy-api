from typing import Optional

from sqlalchemy.orm import Session

from src.models.audit_log import AuditLog


def create_log(
    db: Session,
    user_id: int,
    action: str,
    target: str,
    justification: Optional[str] = None,
) -> AuditLog:
    log = AuditLog(user_id=user_id, action=action, target=target, justification=justification)
    db.add(log)
    db.commit()
    db.refresh(log)
    return log


def list_logs(db: Session, target: Optional[str] = None, limit: int = 50) -> list[AuditLog]:
    q = db.query(AuditLog)
    if target:
        q = q.filter(AuditLog.target == target)
    return q.order_by(AuditLog.created_at.desc()).limit(limit).all()
