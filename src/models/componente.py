from sqlalchemy import Column, Integer, String, Date, Enum, ForeignKey
from sqlalchemy.orm import relationship
from src.db.postgres import Base
from src.models.enums import StatusEnum


class Componente(Base):
    __tablename__ = "componente"

    id              = Column(Integer, primary_key=True, index=True)
    maquina_id      = Column(Integer, ForeignKey("maquina.id"), nullable=False)
    nome            = Column(String(100), nullable=False)
    tipo            = Column(String(100))
    fabricante      = Column(String(100))
    data_instalacao = Column(Date, nullable=True)
    status          = Column(Enum(StatusEnum), default=StatusEnum.active)

    maquina             = relationship("Maquina", back_populates="componentes")
    especificacao_motor = relationship("EspecificacaoMotor", back_populates="componente", uselist=False)
    atributo_valores    = relationship("ComponenteAtributoValor", back_populates="componente")
    leituras_sensor     = relationship("LeituraSensor", back_populates="componente")
    diagnosticos        = relationship("Diagnostico", back_populates="componente")


class EspecificacaoMotor(Base):
    __tablename__ = "especificacao_motor"

    componente_id    = Column(Integer, ForeignKey("componente.id"), primary_key=True)
    potencia_kw      = Column(Integer)
    tensao_nominal   = Column(Integer)
    corrente_nominal = Column(Integer)
    rpm_nominal      = Column(Integer)
    frequencia_hz    = Column(Integer)
    numero_polos     = Column(Integer)
    rendimento       = Column(Integer)

    componente = relationship("Componente", back_populates="especificacao_motor")
