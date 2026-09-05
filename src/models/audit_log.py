from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from src.db.postgres import Base


class AuditLog(Base):
    """Registro de decisão humana (handoff) — quem aprovou/decidiu o quê,
    quando e por quê. Sem hash/criptografia (fora de escopo da CS3)."""
    __tablename__ = "audit_log"

    id             = Column(Integer, primary_key=True, index=True)
    user_id        = Column(Integer, ForeignKey("users.id"), nullable=False)
    action         = Column(String(100), nullable=False)   # ex.: "aprovar_acao_critica"
    target         = Column(String(100), nullable=False)   # ex.: "componente:2"
    justification  = Column(String(1000), nullable=True)
    created_at     = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    user = relationship("User")
