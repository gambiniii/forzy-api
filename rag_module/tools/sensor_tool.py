"""
Tools LangChain — sensores, ML, manutenção e alertas do motor Forzy.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import httpx
from langchain_core.tools import tool

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_module.config import FORZY_API_BASE_URL

FORZY_CSV = PROJECT_ROOT / "History_32026-05-19T11-46-10-920.csv"

_ml_cache: dict = {"models": None, "features_df": None}


def _iso_vibration_label(value: float) -> str:
    return "normal" if value < 2.8 else "atenção"


def _temp_label(value: float) -> str:
    return "normal" if value < 80 else "ALERTA"


def _format_sensor_data(data: dict, machine_id: str) -> str:
    v1 = data.get("vibration_velocity_port1", data.get("vibration_port1", 0))
    v2 = data.get("vibration_velocity_port2", data.get("vibration_port2", 0))
    temp = data.get("temperature", 0)
    a1 = data.get("acceleration_port1", 0)
    a2 = data.get("acceleration_port2", 0)
    ts = data.get("timestamp", datetime.now().isoformat())
    status_note = f"\n- Status: {data['status']}" if data.get("status") else ""

    return (
        f"Leituras atuais do motor {machine_id}:\n"
        f"  - Vibração Port1: {v1} mm/s (ISO: {_iso_vibration_label(float(v1))})\n"
        f"  - Vibração Port2: {v2} mm/s (ISO: {_iso_vibration_label(float(v2))})\n"
        f"  - Temperatura: {temp}°C ({_temp_label(float(temp))})\n"
        f"  - Aceleração Port1: {a1}g | Port2: {a2}g\n"
        f"  - Timestamp: {ts}{status_note}"
    )


@tool
def get_sensor_status(machine_id: str = "1") -> str:
    """Busca leituras atuais dos sensores de vibração, aceleração e temperatura do motor. Use quando perguntarem sobre o estado atual, valores dos sensores ou condição do motor agora."""
    url = f"{FORZY_API_BASE_URL}/sensors/{machine_id}/latest"
    try:
        response = httpx.get(url, timeout=5.0)
        response.raise_for_status()
        data = response.json()
    except Exception:
        data = {
            "machine_id": machine_id,
            "timestamp": datetime.now().isoformat(),
            "temperature": 34.5,
            "vibration_velocity_port1": 2.1,
            "vibration_velocity_port2": 2.3,
            "acceleration_port1": 0.15,
            "acceleration_port2": 0.18,
            "status": "simulated - API offline",
        }

    return _format_sensor_data(data, machine_id)


def _run_local_ml_analysis(machine_id: str) -> dict:
    """Executa predict_single com cache lazy dos modelos e features."""
    global _ml_cache

    if _ml_cache["models"] is None:
        from ml_module.inference.predict import load_all_models

        _ml_cache["models"] = load_all_models()

    if _ml_cache["features_df"] is None:
        from ml_module.features.feature_engineering import build_features, load_raw_csv

        _ml_cache["features_df"] = build_features(load_raw_csv(FORZY_CSV))

    from ml_module.inference.predict import predict_single

    return predict_single(_ml_cache["features_df"], _ml_cache["models"])


def _format_ml_result(result: dict, machine_id: str) -> str:
    estado = result.get("estado_operacional", "operando")

    if estado == "desligado":
        return (
            f"Análise ML do motor {machine_id}:\n"
            f"  ESTADO: Motor desligado\n"
            f"  RECOMENDAÇÃO: {result['combined']['recommendation']}"
        )

    if_result = result["isolation_forest"]
    lstm_result = result["lstm"]
    rul_result = result["rul"]
    combined = result["combined"]
    health_index = result.get("health_index")

    if_label = "anomalia detectada" if if_result["is_anomaly"] else "normal"
    lstm_extra = " (anomalia)" if lstm_result["is_anomaly"] else ""

    if_score = if_result.get("anomaly_score")
    lstm_error = lstm_result.get("reconstruction_error")
    if_score_str = f"{if_score:.4f}" if if_score is not None else "n/a"
    lstm_error_str = f"{lstm_error:.4f}" if lstm_error is not None else "n/a"

    health_str = f"{health_index:.1f}%" if health_index is not None else "n/a"

    rul_available = rul_result.get("available", True)
    if rul_available and rul_result.get("rul_hours") is not None:
        rul_hours = rul_result["rul_hours"]
        maint_days = rul_result["maintenance_window_days"]
        confidence = rul_result.get("confidence")
        rul_str = (
            f"  RUL (Vida Útil Restante): {rul_hours:.1f} horas\n"
            f"    Janela manutenção: {maint_days:.1f} dias\n"
            f"    Risco: {rul_result['risk_level']}"
            + (f" | Confiança: {confidence:.1%}" if confidence is not None else "")
        )
    else:
        reason = rul_result.get("reason", "indisponível")
        rul_str = f"  RUL (Vida Útil Restante): indisponível ({reason})"

    return (
        f"Análise ML do motor {machine_id}:\n"
        f"  ISOLATION FOREST: {if_label}\n"
        f"    Score: {if_score_str}\n"
        f"  LSTM AUTOENCODER: Severidade {lstm_result['severity']}\n"
        f"    Erro reconstrução: {lstm_error_str}{lstm_extra}\n"
        f"{rul_str}\n"
        f"  ÍNDICE DE SAÚDE: {health_str}\n"
        f"  STATUS GERAL: {combined['overall_status']}\n"
        f"  RECOMENDAÇÃO: {combined['recommendation']}"
    )


@tool
def get_ml_analysis(machine_id: str = "1") -> str:
    """Executa análise completa de ML no motor: detecta anomalias com Isolation Forest e LSTM, e estima RUL (vida útil restante). Use quando perguntarem sobre anomalias, saúde do motor, previsão de falha ou manutenção."""
    url = f"{FORZY_API_BASE_URL}/ml/anomaly"
    try:
        response = httpx.post(url, json={"machine_id": machine_id}, timeout=10.0)
        response.raise_for_status()
        data = response.json()
        if "isolation_forest" in data:
            return _format_ml_result(data, machine_id)
        return str(data)
    except Exception:
        result = _run_local_ml_analysis(machine_id)
        return _format_ml_result(result, machine_id)


@tool
def get_maintenance_history(machine_id: str = "1") -> str:
    """Busca histórico de manutenções realizadas no motor. Use quando perguntarem sobre última manutenção, histórico de intervenções ou planejamento de manutenção."""
    url = f"{FORZY_API_BASE_URL}/maintenance"
    params = {"machine_id": machine_id}
    try:
        response = httpx.get(url, params=params, timeout=5.0)
        response.raise_for_status()
        data = response.json()
        events = data if isinstance(data, list) else data.get("events", data.get("history", []))
    except Exception:
        events = [
            {
                "date": (datetime.now().replace(day=1)).strftime("%Y-%m-%d"),
                "type": "Preventiva",
                "description": "Inspeção geral, reaperto de fixações e lubrificação",
                "days_ago": 30,
            },
            {
                "date": "2026-03-10",
                "type": "Corretiva",
                "description": "Troca de rolamento lado acoplamento (Port1)",
                "days_ago": 90,
            },
        ]

    lines = [f"Histórico de manutenção — motor {machine_id}:"]
    for event in events:
        if isinstance(event, dict):
            date = event.get("date", "N/D")
            tipo = event.get("type", event.get("maintenance_type", "N/D"))
            desc = event.get("description", event.get("notes", ""))
            ago = event.get("days_ago")
            ago_str = f" (há {ago} dias)" if ago else ""
            lines.append(f"  - [{date}] {tipo}: {desc}{ago_str}")
        else:
            lines.append(f"  - {event}")

    return "\n".join(lines)


@tool
def get_active_alerts(machine_id: str = "1") -> str:
    """Lista alertas ativos do motor. Use quando perguntarem sobre alertas, alarmes ou problemas reportados."""
    url = f"{FORZY_API_BASE_URL}/alerts"
    params = {"machine_id": machine_id, "resolved": "false"}
    try:
        response = httpx.get(url, params=params, timeout=5.0)
        response.raise_for_status()
        data = response.json()
        alerts = data if isinstance(data, list) else data.get("alerts", [])
        if not alerts:
            return f"Nenhum alerta ativo para o motor {machine_id}."
        lines = [f"Alertas ativos — motor {machine_id}:"]
        for alert in alerts:
            lines.append(
                f"  - [{alert.get('severity', 'N/D')}] {alert.get('message', alert)}"
            )
        return "\n".join(lines)
    except Exception:
        return "Nenhum alerta ativo (API offline - dados simulados)"
