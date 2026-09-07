from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.db.postgres import SessionLocal
from src.routers.auth import get_current_user
from src.services import diagnostico_service

router = APIRouter()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class DiagnosticoOut(BaseModel):
    id: int
    componente_id: int
    timestamp: datetime
    is_anomaly: bool
    overall_status: str
    lstm_severity: Optional[str]
    risk_level: Optional[str]
    rul_hours: Optional[float]
    maintenance_window_days: Optional[float]
    health_score: Optional[float]
    health_index: Optional[float]
    recommendation: Optional[str]
    confidence: Optional[float] = None
    threshold_status: Optional[str] = None
    threshold_message: Optional[str] = None
    breached_metrics: Optional[str] = None

    class Config:
        from_attributes = True


@router.get("/componente/{componente_id}", response_model=list[DiagnosticoOut])
def get_diagnosticos(
    componente_id: int,
    limit: int = Query(50, le=200),
    inicio: Optional[datetime] = Query(None),
    fim: Optional[datetime] = Query(None),
    apenas_anomalias: bool = Query(False),
    _=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return diagnostico_service.list_diagnosticos(
        db, componente_id, limit, inicio, fim, apenas_anomalias
    )


@router.get("/componente/{componente_id}/atribuicao")
def get_atribuicao(
    componente_id: int,
    janela_min: int = Query(15, ge=2, le=120, description="Janela de leituras em minutos"),
    _=Depends(get_current_user),
):
    """Atribuição por componente a partir dos modelos de novidade Forzy.

    Roda o detector treinado para o (motor, regime) da janela atual e devolve
    quais SEGMENTOS DO MODELO 3D devem ser destacados, quais componentes físicos
    são candidatos e por quê.

    É consultivo: não substitui o `overall_status` do diagnóstico, que continua
    vindo do pipeline existente. Serve para o modelo 3D e para o agente.
    """
    from datetime import timedelta, timezone

    from ml_module.forzy import infer
    from src.services import leitura_service

    db = SessionLocal()
    try:
        agora = datetime.now(timezone.utc)
        linhas = leitura_service.get_leituras(
            db, componente_id, agora - timedelta(minutes=janela_min), None, 5000
        )
    finally:
        db.close()

    # `get_leituras` devolve em ordem DECRESCENTE; a inferência precisa de série
    # temporal crescente, senão as janelas móveis andam para trás.
    linhas = sorted(linhas, key=lambda r: r.timestamp)

    # Semântica das colunas (os nomes enganam): `rpm` guarda a VELOCIDADE de
    # vibração em mm/s e `vibracao` guarda a ACELERAÇÃO em g.
    leituras = [
        {"timestamp": r.timestamp, "v_rms": r.rpm, "a_rms": r.vibracao, "temp_c": r.temperatura}
        for r in linhas
        if r.rpm is not None and r.vibracao is not None and r.temperatura is not None
    ]

    segundos_parado = None
    if leituras:
        ativos = [l for l in leituras if (l["v_rms"] or 0) > 0.30]
        if ativos and ativos[-1] is not leituras[-1]:
            segundos_parado = (leituras[-1]["timestamp"] - ativos[-1]["timestamp"]).total_seconds()

    resultado = infer.avaliar(leituras, componente_id, segundos_parado)
    resultado["componente_id"] = componente_id
    resultado["n_leituras"] = len(leituras)
    return resultado


@router.get("/modelos/metricas")
def get_metricas_modelos(_=Depends(get_current_user)):
    """Métricas da bancada de injeção de falhas dos modelos de novidade.

    As falhas são INJETADAS, não observadas — o histórico real é 100% saudável.
    Portanto estes números são um limite superior otimista: provam que o detector
    separa desvios com a assinatura física esperada, não que detectaria uma falha
    real, que tem componentes não modelados (espectro, modulação, carga).
    """
    from ml_module.forzy import infer

    m = infer.metricas_treino()
    if not m:
        return {"disponivel": False,
                "motivo": "Modelos não treinados. Rode: python -m ml_module.forzy.train"}
    return {"disponivel": True, **m}
