from typing import Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.db.postgres import get_db
from src.routers.auth import get_current_user

router = APIRouter()


class LastReading(BaseModel):
    timestamp: Optional[str] = None
    temperatura: Optional[float] = None
    umidade: Optional[float] = None
    corrente: Optional[float] = None
    voltagem: Optional[float] = None
    rpm: Optional[float] = None
    vibracao: Optional[float] = None
    inclinacao: Optional[float] = None


class AnalysisReport(BaseModel):
    motor_id: int
    health_score: float
    health_label: str
    iso_zone: str
    narrative: str
    recent_alerts: List[Any]
    recent_maintenances: List[Any]
    last_reading: Optional[LastReading] = None


def _health_label(score: float) -> str:
    if score >= 80:
        return "Saudável"
    if score >= 60:
        return "Atenção"
    if score >= 40:
        return "Crítico"
    return "Falha"


def _iso_zone(risk_level: Optional[str]) -> str:
    if not risk_level:
        return "A"
    rl = risk_level.lower()
    if "critical" in rl:
        return "D"
    if "high" in rl:
        return "C"
    if "medium" in rl:
        return "B"
    return "A"


@router.get("/{motor_id}/report", response_model=AnalysisReport)
def get_analysis_report(
    motor_id: int,
    db: Session = Depends(get_db),
    _=Depends(get_current_user),
):
    comp_row = db.execute(text(
        "SELECT id FROM componente WHERE maquina_id = :mid LIMIT 1"
    ), {"mid": motor_id}).fetchone()
    if not comp_row:
        raise HTTPException(status_code=404, detail="Nenhum componente para este motor")
    comp_id = comp_row[0]

    diag_row = db.execute(text("""
        SELECT health_score, health_index, overall_status, lstm_severity,
               risk_level, rul_hours, maintenance_window_days, recommendation
        FROM diagnostico WHERE componente_id = :cid
        ORDER BY timestamp DESC LIMIT 1
    """), {"cid": comp_id}).fetchone()

    health_score = 100.0
    health_label = "Saudável"
    iso_zone = "A"
    narrative = "Motor operando normalmente."

    if diag_row:
        raw_hs = diag_row[0] if diag_row[0] is not None else diag_row[1]
        if raw_hs is not None:
            health_score = float(raw_hs * 100) if raw_hs <= 1.0 else float(raw_hs)
        health_label = _health_label(health_score)
        iso_zone = _iso_zone(diag_row[4])
        narrative = diag_row[7] or f"Status: {diag_row[2]}. Severidade LSTM: {diag_row[3]}."

    alert_rows = db.execute(text("""
        SELECT id, motor_id, severity, message, anomaly_score, rul_estimated, created_at, resolved_at
        FROM alerts WHERE motor_id = :mid ORDER BY created_at DESC LIMIT 5
    """), {"mid": motor_id}).fetchall()
    recent_alerts = [
        {
            "id": r[0], "motor_id": r[1], "severity": str(r[2]), "message": r[3],
            "anomaly_score": r[4], "rul_estimated": r[5],
            "created_at": r[6].isoformat() if r[6] else None,
            "resolved_at": r[7].isoformat() if r[7] else None,
        }
        for r in alert_rows
    ]

    maint_rows = db.execute(text("""
        SELECT id, motor_id, type, scheduled_at, completed_at, notes, created_at
        FROM maintenance WHERE motor_id = :mid ORDER BY scheduled_at DESC LIMIT 5
    """), {"mid": motor_id}).fetchall()
    recent_maintenances = [
        {
            "id": r[0], "motor_id": r[1], "type": str(r[2]),
            "scheduled_at": r[3].isoformat() if r[3] else None,
            "completed_at": r[4].isoformat() if r[4] else None,
            "notes": r[5],
            "created_at": r[6].isoformat() if r[6] else None,
        }
        for r in maint_rows
    ]

    reading_row = db.execute(text("""
        SELECT timestamp, temperatura, umidade, corrente, voltagem, rpm, vibracao, inclinacao
        FROM leitura_sensor WHERE componente_id = :cid ORDER BY timestamp DESC LIMIT 1
    """), {"cid": comp_id}).fetchone()
    last_reading = None
    if reading_row:
        last_reading = LastReading(
            timestamp=reading_row[0].isoformat() if reading_row[0] else None,
            temperatura=reading_row[1], umidade=reading_row[2],
            corrente=reading_row[3], voltagem=reading_row[4],
            rpm=reading_row[5], vibracao=reading_row[6], inclinacao=reading_row[7],
        )

    return AnalysisReport(
        motor_id=motor_id,
        health_score=health_score,
        health_label=health_label,
        iso_zone=iso_zone,
        narrative=narrative,
        recent_alerts=recent_alerts,
        recent_maintenances=recent_maintenances,
        last_reading=last_reading,
    )
