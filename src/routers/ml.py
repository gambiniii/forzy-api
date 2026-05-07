from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from src.models.enums import UserRoleEnum
from src.routers.auth import get_current_user, require_role

router = APIRouter()

# ─────────────────────────────────────────────────────────────────────────────
# ESQUELETO ML — endpoints prontos, retornando mocks por enquanto.
# Pipeline planejada:
#   1. Isolation Forest  → anomalia estática (baseline rápido, sem labels)
#   2. LSTM Autoencoder  → anomalia temporal (coração do projeto)
#   3. XGBoost           → RUL (Remaining Useful Life) + janela de manutenção
#
# Quando os modelos forem treinados, substitua os blocos "#TODO" abaixo.
# ─────────────────────────────────────────────────────────────────────────────


class PredictRequest(BaseModel):
    componente_id: int
    window_minutes: int = 60


class AnomalyResponse(BaseModel):
    componente_id: int
    anomaly_score: float
    is_anomaly: bool
    severity: str
    model_used: str
    detail: Optional[str] = None


class RULResponse(BaseModel):
    componente_id: int
    rul_hours: float
    maintenance_window_days: float
    confidence: float
    detail: Optional[str] = None


@router.post("/anomaly", response_model=AnomalyResponse)
def detect_anomaly(
    req: PredictRequest,
    _=Depends(require_role(UserRoleEnum.admin, UserRoleEnum.technician)),
):
    """
    Detecta anomalia com base nas últimas leituras do componente (leitura_sensor).

    TODO:
    1. Buscar leituras do PostgreSQL (últimos req.window_minutes minutos)
    2. Rodar Isolation Forest como baseline
    3. Rodar LSTM Autoencoder — calcular erro de reconstrução
    4. Combinar scores e definir severidade
    5. Se anomalia, criar alerta via /alerts automaticamente
    """
    # MOCK — remover após treinamento dos modelos
    return AnomalyResponse(
        componente_id=req.componente_id,
        anomaly_score=0.0,
        is_anomaly=False,
        severity="normal",
        model_used="mock",
        detail="Modelo ainda não treinado — aguardando dados reais",
    )


@router.post("/rul", response_model=RULResponse)
def predict_rul(
    req: PredictRequest,
    _=Depends(require_role(UserRoleEnum.admin, UserRoleEnum.technician)),
):
    """
    Prediz Remaining Useful Life (RUL) e janela de manutenção.

    TODO:
    1. Buscar leituras do PostgreSQL (leitura_sensor)
    2. Calcular features de tendência (rolling mean, std, taxa de degradação)
    3. Rodar XGBoost treinado
    4. Retornar RUL em horas e sugestão de janela
    """
    # MOCK — remover após treinamento dos modelos
    return RULResponse(
        componente_id=req.componente_id,
        rul_hours=720.0,
        maintenance_window_days=30.0,
        confidence=0.0,
        detail="Modelo ainda não treinado — aguardando dados reais",
    )


@router.get("/status")
def ml_status(_=Depends(get_current_user)):
    """Status dos modelos — usado pelo dashboard para saber se ML está ativo."""
    return {
        "isolation_forest": "not_trained",
        "lstm_autoencoder": "not_trained",
        "xgboost_rul":      "not_trained",
        "message": "Modelos serão ativados após coleta e treinamento com dados reais",
    }
