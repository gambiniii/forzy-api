from sqlalchemy import Column, Integer, Float, Text, Enum, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from src.db.postgres import Base
from src.models.enums import SeverityEnum


class Alert(Base):
    __tablename__ = "alerts"

    id             = Column(Integer, primary_key=True, index=True)
    machine_id     = Column(Integer, ForeignKey("maquina.id"), nullable=False)
    severity       = Column(Enum(SeverityEnum), nullable=False)
    message        = Column(Text, nullable=False)
    anomaly_score  = Column(Float, nullable=True)
    rul_estimated  = Column(Float, nullable=True)  # Remaining Useful Life em horas
    created_at     = Column(DateTime(timezone=True), server_default=func.now())
    resolved_at    = Column(DateTime(timezone=True), nullable=True)

    machine = relationship("Maquina", back_populates="alertas")
