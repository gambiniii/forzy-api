from sqlalchemy import Column, Integer, Float, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from src.db.postgres import Base


class Anomalia(Base):
    __tablename__ = "anomalia"

    id                      = Column(Integer, primary_key=True, index=True)
    componente_id           = Column(Integer, ForeignKey("componente.id"), nullable=False)
    timestamp               = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    overall_status          = Column(String(20), nullable=False)   # warning | critical
    lstm_severity           = Column(String(20), nullable=True)    # low | medium | high | critical
    risk_level              = Column(String(20), nullable=True)    # low | medium | high | critical
    rul_hours               = Column(Float, nullable=True)
    maintenance_window_days = Column(Float, nullable=True)
    health_score            = Column(Float, nullable=True)
    recommendation          = Column(String(500), nullable=True)

    componente = relationship("Componente", back_populates="anomalias")
