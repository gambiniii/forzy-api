from sqlalchemy import Column, Integer, Text, Enum, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from src.db.postgres import Base
from src.models.enums import MaintenanceTypeEnum


class Maintenance(Base):
    __tablename__ = "maintenance"

    id           = Column(Integer, primary_key=True, index=True)
    machine_id   = Column(Integer, ForeignKey("maquina.id"), nullable=False)
    type         = Column(Enum(MaintenanceTypeEnum), nullable=False)
    scheduled_at = Column(DateTime(timezone=True), nullable=False)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    notes        = Column(Text)
    created_at   = Column(DateTime(timezone=True), server_default=func.now())

    machine = relationship("Maquina", back_populates="manutencoes")
