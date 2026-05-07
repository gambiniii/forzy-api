from sqlalchemy import Column, Integer, Float, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from src.db.postgres import Base


class LeituraSensor(Base):
    __tablename__ = "leitura_sensor"

    id            = Column(Integer, primary_key=True, index=True)
    componente_id = Column(Integer, ForeignKey("componente.id"), nullable=False)
    timestamp     = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    temperatura   = Column(Float, nullable=True)   # °C
    umidade       = Column(Float, nullable=True)   # %
    corrente      = Column(Float, nullable=True)   # A
    voltagem      = Column(Float, nullable=True)   # V
    rpm           = Column(Float, nullable=True)
    vibracao      = Column(Float, nullable=True)   # g
    inclinacao    = Column(Float, nullable=True)   # °

    componente = relationship("Componente", back_populates="leituras_sensor")
