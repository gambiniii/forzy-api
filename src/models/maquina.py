from sqlalchemy import Column, Integer, String, Enum, DateTime
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from src.db.postgres import Base
from src.models.enums import StatusEnum


class Maquina(Base):
    __tablename__ = "maquina"

    id             = Column(Integer, primary_key=True, index=True)
    nome           = Column(String(100), nullable=False)
    tipo           = Column(String(100))
    fabricante     = Column(String(100))
    ano_instalacao = Column(Integer)
    status         = Column(Enum(StatusEnum), default=StatusEnum.active)
    localizacao    = Column(String(200))

    componentes  = relationship("Componente", back_populates="maquina")
    alertas      = relationship("Alert", back_populates="machine")
    manutencoes  = relationship("Maintenance", back_populates="machine")
