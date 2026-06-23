"""
Tool LangChain — consulta direta ao banco de dados Forzy (PostgreSQL).
Busca leituras recentes e diagnósticos ML persistidos, fornecendo contexto
factual e atualizado ao agente RAG.
"""

from __future__ import annotations

import sys
from pathlib import Path

from langchain_core.tools import tool

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _get_session():
    # Importar todos os modelos antes de usar a sessão para garantir que o
    # mapper do SQLAlchemy resolva todos os relacionamentos (Planta, Maquina etc.)
    import src.models.user         # noqa: F401
    import src.models.planta       # noqa: F401
    import src.models.maquina      # noqa: F401
    import src.models.componente   # noqa: F401
    import src.models.atributo     # noqa: F401
    import src.models.leitura_sensor  # noqa: F401
    import src.models.alert        # noqa: F401
    import src.models.maintenance  # noqa: F401
    import src.models.diagnostico  # noqa: F401

    from src.db.postgres import SessionLocal
    return SessionLocal()


def _iso_vibracao(v: float | None) -> str:
    if v is None:
        return "n/d"
    if v < 2.8:
        return "normal"
    if v < 4.5:
        return "atenção"
    return "CRÍTICA"


def _temp_label(v: float | None) -> str:
    if v is None:
        return "n/d"
    return "normal" if v < 80 else "ALERTA"


@tool
def get_db_leituras(componente_id: int = 1, limit: int = 5) -> str:
    """Busca as últimas leituras de sensor (vibração, temperatura, RPM, corrente, voltagem)
    diretamente do banco de dados PostgreSQL. Use quando precisar de dados históricos
    recentes do componente/motor para embasar um diagnóstico ou tendência."""
    try:
        from src.models.leitura_sensor import LeituraSensor

        db = _get_session()
        try:
            rows = (
                db.query(LeituraSensor)
                .filter(LeituraSensor.componente_id == componente_id)
                .order_by(LeituraSensor.timestamp.desc())
                .limit(limit)
                .all()
            )
        finally:
            db.close()

        if not rows:
            return f"Nenhuma leitura encontrada no banco para o componente {componente_id}."

        lines = [f"Historico de leituras — componente {componente_id} (ultimas {len(rows)}):"]
        for r in rows:
            ts = r.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC") if r.timestamp else "n/d"
            vib  = r.vibracao
            temp = r.temperatura
            lines.append(f"  [{ts}]")
            lines.append(f"  - Vibracao: {vib if vib is not None else 'n/d'} g ({_iso_vibracao(vib)})")
            lines.append(f"  - Temperatura: {temp if temp is not None else 'n/d'} C ({_temp_label(temp)})")
            lines.append(f"  - RPM: {r.rpm if r.rpm is not None else 'n/d'}")
            lines.append(f"  - Corrente: {r.corrente if r.corrente is not None else 'n/d'} A")
            lines.append(f"  - Voltagem: {r.voltagem if r.voltagem is not None else 'n/d'} V")
        return "\n".join(lines)

    except Exception as exc:
        return f"Erro ao consultar leituras no banco: {exc}"


@tool
def get_db_diagnosticos(componente_id: int = 1, limit: int = 3) -> str:
    """Busca os diagnosticos ML mais recentes persistidos no banco de dados PostgreSQL.
    Cada diagnostico contem: status geral, severidade LSTM, nivel de risco, RUL (vida util
    restante em horas), indice de saude e recomendacao gerada pelo modelo. Use quando
    precisar de analise historica de saude do motor ou tendencia de degradacao."""
    try:
        from src.models.diagnostico import Diagnostico

        db = _get_session()
        try:
            rows = (
                db.query(Diagnostico)
                .filter(Diagnostico.componente_id == componente_id)
                .order_by(Diagnostico.timestamp.desc())
                .limit(limit)
                .all()
            )
        finally:
            db.close()

        if not rows:
            return f"Nenhum diagnostico encontrado no banco para o componente {componente_id}."

        lines = [f"Diagnosticos ML — componente {componente_id} (ultimos {len(rows)}):"]
        for d in rows:
            ts = d.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC") if d.timestamp else "n/d"
            rul    = f"{d.rul_hours:.1f} h" if d.rul_hours is not None else "n/d"
            janela = f"{d.maintenance_window_days:.1f} dias" if d.maintenance_window_days is not None else "n/d"
            health = (
                f"{d.health_index:.1f}%" if d.health_index is not None else
                f"{d.health_score * 100:.1f}%" if d.health_score is not None else "n/d"
            )
            anomaly = "SIM" if d.is_anomaly else "nao"
            lines.append(f"  [{ts}]")
            lines.append(f"  - Anomalia: {anomaly}")
            lines.append(f"  - Status geral: {d.overall_status}")
            lines.append(f"  - Severidade LSTM: {d.lstm_severity or 'n/d'}")
            lines.append(f"  - Nivel de risco: {d.risk_level or 'n/d'}")
            lines.append(f"  - Indice de saude: {health}")
            lines.append(f"  - RUL: {rul}")
            lines.append(f"  - Janela de manutencao: {janela}")
            lines.append(f"  - Recomendacao: {d.recommendation or 'n/d'}")
        return "\n".join(lines)

    except Exception as exc:
        return f"Erro ao consultar diagnosticos no banco: {exc}"
