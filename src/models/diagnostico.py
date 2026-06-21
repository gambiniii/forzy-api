from sqlalchemy import Boolean, Column, Integer, Float, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from src.db.postgres import Base


class Diagnostico(Base):
    __tablename__ = "diagnostico"

    id                      = Column(Integer, primary_key=True, index=True)
    componente_id           = Column(Integer, ForeignKey("componente.id"), nullable=False)
    timestamp               = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    is_anomaly              = Column(Boolean, nullable=False, default=False)
    overall_status          = Column(String(20), nullable=False)   # healthy | warning | critical | motor_desligado
    lstm_severity           = Column(String(20), nullable=True)    # normal | low | medium | high | critical
    risk_level              = Column(String(20), nullable=True)    # low | medium | high | critical | unknown
    rul_hours               = Column(Float, nullable=True)
    maintenance_window_days = Column(Float, nullable=True)
    health_score            = Column(Float, nullable=True)
    health_index            = Column(Float, nullable=True)         # 0-100%
    recommendation          = Column(String(500), nullable=True)

    componente = relationship("Componente", back_populates="diagnosticos")
