from sqlalchemy import Column, Boolean, String, Text, DateTime
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func
from src.db.postgres import Base


class MotorSegmento(Base):
    """Catálogo dos 23 segmentos do modelo 3D (public/3d/Engine1.obj, forzy-web)
    do motor WEG W22 3cv monofásico, carcaça 100L. Fonte única de verdade —
    forzy-web/src/config/motorParts.ts e rag_module/tools/motor_parts_tool.py
    devem ser gerados/sincronizados a partir desta tabela, não editados à mão.
    Ver migrations/006_motor_segmento.sql."""
    __tablename__ = "motor_segmento"

    id                   = Column(String, primary_key=True)  # nome do objeto no OBJ (empty_2..empty_24)
    label                = Column(String, nullable=False)
    grupo_funcional      = Column(String, nullable=False)  # mecanico | eletrico | termico | estrutural | vedacao
    descricao            = Column(Text, nullable=False)
    sinal_velocidade     = Column(String, nullable=False, default="nenhuma")
    sinal_aceleracao     = Column(String, nullable=False, default="nenhuma")
    sinal_temperatura    = Column(String, nullable=False, default="nenhuma")
    componentes_internos = Column(JSONB, nullable=False, default=list)
    modos_falha          = Column(JSONB, nullable=False, default=list)  # [{nome, assinatura, precocidade}]
    destacavel           = Column(Boolean, nullable=False, default=False)
    inspecao             = Column(Text, nullable=True)
    atualizado_em        = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
