from sqlalchemy import Column, Integer, String, Boolean, DateTime
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from src.db.postgres import Base


class Planta(Base):
    __tablename__ = "planta"

    id          = Column(Integer, primary_key=True, index=True)
    nome        = Column(String(150), nullable=False)
    localizacao = Column(String(200))
    cidade      = Column(String(100))
    estado      = Column(String(2))
    ativo       = Column(Boolean, default=True, nullable=False)
    created_at  = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    maquinas = relationship("Maquina", back_populates="planta")
