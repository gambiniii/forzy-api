"""
Tools LangChain — sensores, ML, manutenção e alertas em tempo real (API interna).
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

_ml_cache: dict = {"models": None, "features_df": None}
FORZY_CSV = PROJECT_ROOT / "History_32026-05-19T11-46-10-920.csv"


def _api_get(path: str, params: dict | None = None, timeout: float = 6.0):
    """GET autenticado na API interna com token admin."""
    import os
    token = os.getenv("AGENT_API_TOKEN", "")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    return httpx.get(f"{FORZY_API_BASE_URL}{path}", params=params, headers=headers, timeout=timeout)


def _iso_label(v: float | None) -> str:
    if v is None: return "n/d"
    return "normal" if v < 2.8 else "atenção" if v < 4.5 else "CRÍTICA"


def _temp_label(v: float | None) -> str:
    if v is None: return "n/d"
    return "normal" if v < 80 else "ALERTA"


@tool
def get_sensor_status(component_id: str = "1") -> str:
    """Busca a leitura mais recente dos sensores (temperatura, vibração, RPM).
    Use para saber o estado ATUAL do motor neste momento."""
    try:
        r = _api_get(f"/sensors/component/{component_id}/latest")
        r.raise_for_status()
        d = r.json()
        if not d:
            return f"Sem leituras disponíveis para componente {component_id}."
        ts = d.get("timestamp", "n/d")
        temp = d.get("temperatura"); vib = d.get("vibracao"); rpm = d.get("rpm")
        return (
            f"Última leitura — componente {component_id} [{ts}]:\n"
            f"  Temperatura: {temp}°C ({_temp_label(temp)})\n"
            f"  Vibração: {vib} mm/s ({_iso_label(vib)})\n"
            f"  RPM: {rpm or 'n/d'} | Corrente: {d.get('corrente') or 'n/d'}A | "
            f"Voltagem: {d.get('voltagem') or 'n/d'}V"
        )
    except Exception as exc:
        return f"Erro ao buscar leitura do sensor: {exc}"


@tool
def get_active_alerts(motor_id: str = "1") -> str:
    """Lista alertas ATIVOS (não resolvidos) do motor.
    Use quando perguntarem sobre alarmes, problemas em aberto ou notificações."""
    try:
        r = _api_get("/alerts/", params={"motor_id": motor_id, "resolved": "false"})
        r.raise_for_status()
        alerts = r.json()
        if not alerts:
            return f"Nenhum alerta ativo para motor {motor_id}."
        lines = [f"Alertas ativos — motor {motor_id}:"]
        for a in alerts:
            lines.append(f"  [{a.get('severity','?').upper()}] {a.get('message','?')} | Score: {a.get('anomaly_score','n/d')}")
        return "\n".join(lines)
    except Exception as exc:
        return f"Erro ao buscar alertas: {exc}"


@tool
def get_maintenance_history(motor_id: str = "1") -> str:
    """Busca histórico de manutenções (preventiva, corretiva, preditiva) do motor.
    Use para perguntas sobre última manutenção, intervenções ou planejamento."""
    try:
        r = _api_get("/maintenance/", params={"motor_id": motor_id})
        r.raise_for_status()
        records = r.json()
        if not records:
            return f"Nenhum registro de manutenção para motor {motor_id}."
        lines = [f"Manutenções — motor {motor_id}:"]
        for m in records:
            sched = m.get("scheduled_at", "n/d")[:10] if m.get("scheduled_at") else "n/d"
            done  = m.get("completed_at", "pendente")[:10] if m.get("completed_at") else "pendente"
            lines.append(f"  [{sched}] {m.get('type','?').upper()} | Realizada: {done} | {m.get('notes') or 'sem obs.'}")
        return "\n".join(lines)
    except Exception as exc:
        return f"Erro ao buscar manutenções: {exc}"


@tool
def get_ml_analysis(motor_id: str = "1") -> str:
    """Executa análise ML completa: anomalias (Isolation Forest + LSTM) e RUL (vida útil restante).
    Use para perguntas sobre saúde, anomalias, risco ou previsão de falha."""
    url = f"{FORZY_API_BASE_URL}/ml/anomaly"
    try:
        r = httpx.post(url, json={"machine_id": motor_id}, timeout=12.0)
        r.raise_for_status()
        data = r.json()
    except Exception:
        try:
            data = _run_local_ml(motor_id)
        except Exception as exc:
            return f"Análise ML indisponível: {exc}"

    return _format_ml(data, motor_id)


def _run_local_ml(motor_id: str) -> dict:
    global _ml_cache
    if _ml_cache["models"] is None:
        from ml_module.inference.predict import load_all_models
        _ml_cache["models"] = load_all_models()
    if _ml_cache["features_df"] is None:
        from ml_module.features.feature_engineering import build_features, load_raw_csv
        _ml_cache["features_df"] = build_features(load_raw_csv(FORZY_CSV))
    from ml_module.inference.predict import predict_single
    return predict_single(_ml_cache["features_df"], _ml_cache["models"])


def _format_ml(data: dict, motor_id: str) -> str:
    if data.get("estado_operacional") == "desligado":
        return f"Análise ML motor {motor_id}:\n  Estado: DESLIGADO\n  {data.get('combined', {}).get('recommendation','')}"

    if_r    = data.get("isolation_forest", {})
    lstm_r  = data.get("lstm", {})
    rul_r   = data.get("rul", {})
    comb    = data.get("combined", {})
    hi      = data.get("health_index")

    if_label = "anomalia" if if_r.get("is_anomaly") else "normal"
    lstm_sev = lstm_r.get("severity", "n/d")
    hi_str   = f"{hi:.1f}%" if hi is not None else "n/d"

    if rul_r.get("available") and rul_r.get("rul_hours") is not None:
        rul_str = f"{rul_r['rul_hours']:.1f}h (risco: {rul_r.get('risk_level','?')})"
    else:
        rul_str = f"indisponível ({rul_r.get('reason','?')})"

    return (
        f"Análise ML — motor {motor_id}:\n"
        f"  Isolation Forest: {if_label} (score: {if_r.get('anomaly_score','n/d')})\n"
        f"  LSTM Autoencoder: severidade {lstm_sev} (erro: {lstm_r.get('reconstruction_error','n/d')})\n"
        f"  RUL: {rul_str}\n"
        f"  Índice de saúde: {hi_str}\n"
        f"  Status geral: {comb.get('overall_status','n/d')}\n"
        f"  Recomendação: {comb.get('recommendation','n/d')}"
    )
