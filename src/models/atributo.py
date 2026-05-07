from sqlalchemy import Column, Integer, String, Float, ForeignKey
from sqlalchemy.orm import relationship
from src.db.postgres import Base


class Atributo(Base):
    __tablename__ = "atributo"

    id         = Column(Integer, primary_key=True, index=True)
    nome       = Column(String(100), nullable=False, unique=True)
    unidade    = Column(String(50))
    tipo_dado  = Column(String(50))  # float | int | string

    valores = relationship("ComponenteAtributoValor", back_populates="atributo")


class ComponenteAtributoValor(Base):
    __tablename__ = "componente_atributo_valor"

    id            = Column(Integer, primary_key=True, index=True)
    componente_id = Column(Integer, ForeignKey("componente.id"), nullable=False)
    atributo_id   = Column(Integer, ForeignKey("atributo.id"), nullable=False)
    valor_float   = Column(Float, nullable=True)
    valor_int     = Column(Integer, nullable=True)
    valor_string  = Column(String(255), nullable=True)

    componente = relationship("Componente", back_populates="atributo_valores")
    atributo   = relationship("Atributo", back_populates="valores")
