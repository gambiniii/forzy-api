"""
Tools LangChain — acesso completo ao banco PostgreSQL da Forzy.
Fornece leituras, diagnósticos, motores, plantas, alertas e manutenções.
"""

from __future__ import annotations

import sys
from pathlib import Path

from langchain_core.tools import tool

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _db():
    from src.db.postgres import SessionLocal
    return SessionLocal()


def _exec(sql: str, params: dict | None = None):
    from sqlalchemy import text
    from src.db.postgres import SessionLocal
    db = SessionLocal()
    try:
        result = db.execute(text(sql), params or {})
        return result.fetchall()
    finally:
        db.close()


# ── Sensores ──────────────────────────────────────────────────────────────────

@tool
def get_db_leituras(componente_id: int = 1, limit: int = 10) -> str:
    """Busca as últimas leituras de sensor (vibração, temperatura, RPM, corrente, voltagem)
    do banco PostgreSQL. Use para dados históricos recentes, tendências ou embasar diagnóstico."""
    try:
        rows = _exec("""
            SELECT timestamp, temperatura, umidade, corrente, voltagem, rpm, vibracao, inclinacao
            FROM leitura_sensor
            WHERE componente_id = :cid
            ORDER BY timestamp DESC LIMIT :lim
        """, {"cid": componente_id, "lim": limit})

        if not rows:
            return f"Nenhuma leitura encontrada para componente {componente_id}."

        lines = [f"Últimas {len(rows)} leituras — componente {componente_id}:"]
        for r in rows:
            ts = r[0].strftime("%d/%m/%Y %H:%M UTC") if r[0] else "n/d"
            vib = r[6]
            temp = r[1]
            vib_label = ("normal" if vib is None or vib < 2.8 else "atenção" if vib < 4.5 else "CRÍTICA")
            temp_label = "normal" if temp is None or temp < 80 else "ALERTA"
            lines.append(
                f"  [{ts}] Temp: {temp or 'n/d'}°C ({temp_label}) | "
                f"Vibr: {vib or 'n/d'} mm/s ({vib_label}) | "
                f"RPM: {r[5] or 'n/d'} | Corrente: {r[3] or 'n/d'}A | Voltagem: {r[4] or 'n/d'}V"
            )
        return "\n".join(lines)
    except Exception as exc:
        return f"Erro ao consultar leituras: {exc}"


@tool
def get_sensor_trends(componente_id: int = 1, hours: int = 24) -> str:
    """Analisa tendências dos sensores nas últimas N horas: médias, máximos, mínimos
    e variação de temperatura, vibração e RPM. Use para perguntas sobre tendência ou evolução."""
    try:
        rows = _exec("""
            SELECT
                COUNT(*) as total,
                AVG(temperatura) as avg_temp, MAX(temperatura) as max_temp, MIN(temperatura) as min_temp,
                AVG(rpm) as avg_rpm, MAX(rpm) as max_rpm,
                AVG(vibracao) as avg_vib, MAX(vibracao) as max_vib,
                MIN(timestamp) as desde, MAX(timestamp) as ate
            FROM leitura_sensor
            WHERE componente_id = :cid
              AND timestamp >= NOW() - INTERVAL ':h hours'
        """.replace(":h", str(hours)), {"cid": componente_id})

        if not rows or rows[0][0] == 0:
            rows = _exec("""
                SELECT
                    COUNT(*) as total,
                    AVG(temperatura), MAX(temperatura), MIN(temperatura),
                    AVG(rpm), MAX(rpm),
                    AVG(vibracao), MAX(vibracao),
                    MIN(timestamp), MAX(timestamp)
                FROM leitura_sensor
                WHERE componente_id = :cid
            """, {"cid": componente_id})

        r = rows[0]
        total = r[0]
        if not total:
            return "Sem dados de tendência disponíveis."

        def fmt(v): return f"{float(v):.2f}" if v is not None else "n/d"

        return (
            f"Tendências — componente {componente_id} (últimas {hours}h, {total} amostras):\n"
            f"  Temperatura: média {fmt(r[1])}°C | máx {fmt(r[2])}°C | mín {fmt(r[3])}°C\n"
            f"  RPM:         média {fmt(r[4])} | máx {fmt(r[5])}\n"
            f"  Vibração:    média {fmt(r[6])} mm/s | máx {fmt(r[7])} mm/s\n"
            f"  Período:     {r[8]} → {r[9]}"
        )
    except Exception as exc:
        return f"Erro ao calcular tendências: {exc}"


# ── Diagnósticos ──────────────────────────────────────────────────────────────

@tool
def get_db_diagnosticos(componente_id: int = 1, limit: int = 5) -> str:
    """Busca diagnósticos ML do banco: status, severidade LSTM, risco, RUL, índice de saúde
    e recomendações. Use para histórico de saúde do motor ou tendência de degradação."""
    try:
        rows = _exec("""
            SELECT timestamp, is_anomaly, overall_status, lstm_severity,
                   risk_level, rul_hours, maintenance_window_days,
                   health_score, health_index, recommendation
            FROM diagnostico
            WHERE componente_id = :cid
            ORDER BY timestamp DESC LIMIT :lim
        """, {"cid": componente_id, "lim": limit})

        if not rows:
            return f"Nenhum diagnóstico encontrado para componente {componente_id}."

        lines = [f"Diagnósticos ML — componente {componente_id} (últimos {len(rows)}):"]
        for d in rows:
            ts = d[0].strftime("%d/%m/%Y %H:%M UTC") if d[0] else "n/d"
            hs = d[8] if d[8] is not None else (d[7] * 100 if d[7] is not None else None)
            health = f"{hs:.1f}%" if hs is not None else "n/d"
            rul = f"{d[5]:.1f}h" if d[5] is not None else "n/d"
            janela = f"{d[6]:.1f} dias" if d[6] is not None else "n/d"
            lines.append(
                f"  [{ts}] Status: {d[2]} | Anomalia: {'SIM' if d[1] else 'não'} | "
                f"Severidade: {d[3] or 'n/d'} | Risco: {d[4] or 'n/d'}\n"
                f"    Saúde: {health} | RUL: {rul} | Próx. manutenção: {janela}\n"
                f"    → {d[9] or 'sem recomendação'}"
            )
        return "\n".join(lines)
    except Exception as exc:
        return f"Erro ao consultar diagnósticos: {exc}"


# ── Visão geral do sistema ─────────────────────────────────────────────────────

@tool
def get_system_overview() -> str:
    """Retorna visão geral completa de toda a aplicação: plantas, motores, componentes,
    total de leituras, último diagnóstico e alertas ativos. Use para perguntas sobre
    o estado geral do sistema ou para iniciar uma análise."""
    try:
        ativos = _exec("SELECT id, nome, localizacao, ativo FROM planta ORDER BY id")
        motors = _exec("SELECT id, nome, tipo, status, planta_id FROM maquina ORDER BY id")
        comps = _exec("SELECT id, maquina_id, nome, tipo, status FROM componente ORDER BY id")
        leit_count = _exec("SELECT COUNT(*), MAX(timestamp) FROM leitura_sensor")
        diag = _exec("""
            SELECT overall_status, health_score, health_index, risk_level, recommendation
            FROM diagnostico
            ORDER BY timestamp DESC LIMIT 1
        """)
        alerts = _exec("SELECT COUNT(*) FROM alerts WHERE resolved_at IS NULL")
        maint = _exec("SELECT COUNT(*) FROM maintenance WHERE completed_at IS NULL")

        lines = ["=== VISÃO GERAL DO SISTEMA FORZY DIGITAL TWIN ===\n"]

        lines.append("PLANTAS / ATIVOS:")
        for a in ativos:
            lines.append(f"  [{a[0]}] {a[1]} — {a[2]} | Status: {a[3]}")

        lines.append("\nMOTORES:")
        for m in motors:
            lines.append(f"  [{m[0]}] {m[1]} ({m[2]}) | Status: {m[3]} | Planta: {m[4]}")

        lines.append("\nCOMPONENTES:")
        for c in comps:
            lines.append(f"  [{c[0]}] {c[2]} ({c[3]}) | Motor: {c[1]} | Status: {c[4]}")

        if leit_count:
            total, ultima = leit_count[0]
            ts = ultima.strftime("%d/%m/%Y %H:%M UTC") if ultima else "n/d"
            lines.append(f"\nLEITURAS DE SENSOR: {total} registros | Última em: {ts}")

        if diag:
            d = diag[0]
            hs = d[2] if d[2] is not None else (d[1] * 100 if d[1] is not None else None)
            lines.append(
                f"\nDIAGNÓSTICO MAIS RECENTE:\n"
                f"  Status: {d[0]} | Saúde: {f'{hs:.1f}%' if hs else 'n/d'} | "
                f"Risco: {d[3] or 'n/d'}\n  → {d[4] or 'sem recomendação'}"
            )

        lines.append(
            f"\nALERTAS ATIVOS: {alerts[0][0] if alerts else 0} | "
            f"MANUTENÇÕES PENDENTES: {maint[0][0] if maint else 0}"
        )

        return "\n".join(lines)
    except Exception as exc:
        return f"Erro ao buscar visão geral: {exc}"


@tool
def get_motor_details(motor_id: int = 1) -> str:
    """Retorna especificações técnicas completas de um motor: placa de identificação,
    tensão nominal, corrente, RPM, potência e fator de potência.
    Use quando precisar das specs do motor para análise ou relatório."""
    try:
        rows = _exec("""
            SELECT mq.id, mq.nome, mq.tipo, mq.status,
                   esp.tensao_nominal, esp.corrente_nominal, esp.rpm_nominal,
                   esp.potencia_kw, esp.frequencia_hz, esp.numero_polos, esp.rendimento,
                   pl.nome as planta, pl.localizacao
            FROM maquina mq
            LEFT JOIN componente c ON c.maquina_id = mq.id
            LEFT JOIN especificacao_motor esp ON esp.componente_id = c.id
            LEFT JOIN planta pl ON pl.id = mq.planta_id
            WHERE mq.id = :mid
        """, {"mid": motor_id})

        if not rows:
            return f"Motor {motor_id} não encontrado."

        r = rows[0]
        return (
            f"Motor [{r[0]}] {r[1]}\n"
            f"  Tipo: {r[2]} | Status: {r[3]}\n"
            f"  Planta: {r[11]} — {r[12]}\n"
            f"  Placa de identificação:\n"
            f"    Tensão: {r[4]}V | Corrente: {r[5]}A | RPM: {r[6]}\n"
            f"    Potência: {r[7]}kW | Frequência: {r[8]}Hz | Polos: {r[9]} | Rendimento: {r[10]}%"
        )
    except Exception as exc:
        return f"Erro ao buscar motor: {exc}"


# ── Alertas e Manutenção ───────────────────────────────────────────────────────

@tool
def get_alerts_history(motor_id: int = 1, limit: int = 10) -> str:
    """Busca histórico completo de alertas (ativos e resolvidos) de um motor.
    Use para avaliar frequência de problemas ou histórico de incidentes."""
    try:
        rows = _exec("""
            SELECT id, severity, message, anomaly_score, rul_estimated,
                   created_at, resolved_at
            FROM alerts WHERE motor_id = :mid
            ORDER BY created_at DESC LIMIT :lim
        """, {"mid": motor_id, "lim": limit})

        if not rows:
            return f"Nenhum alerta registrado para motor {motor_id}."

        lines = [f"Histórico de alertas — motor {motor_id} ({len(rows)} registros):"]
        for r in rows:
            ts = r[5].strftime("%d/%m/%Y %H:%M") if r[5] else "n/d"
            status = "✓ resolvido" if r[6] else "⚠ ATIVO"
            lines.append(
                f"  [{ts}] {status} | {r[1].upper()} — {r[2]}\n"
                f"    Score: {f'{r[3]:.4f}' if r[3] else 'n/d'} | RUL: {f'{r[4]:.1f}h' if r[4] else 'n/d'}"
            )
        return "\n".join(lines)
    except Exception as exc:
        return f"Erro ao buscar alertas: {exc}"


@tool
def compare_motors() -> str:
    """Compara S1 (componente_id=1) e S2 (componente_id=2) lado a lado:
    última leitura, diagnóstico mais recente, alertas ativos e tendência.
    Use quando o usuário perguntar sobre os dois motores, qual está melhor ou pior,
    diferenças entre S1 e S2, ou estado geral da planta."""
    try:
        results = ["=== COMPARAÇÃO S1 vs S2 ===\n"]
        for cid, label in [(2, "S1 — Motor WEG W22 Unidade S1 (componente_id=2)"), (3, "S2 — Motor WEG W22 Unidade S2 (componente_id=3)")]:
            results.append(f"── {label} ──")

            # Última leitura
            leit = _exec("""
                SELECT timestamp, temperatura, vibracao, rpm
                FROM leitura_sensor WHERE componente_id = :cid
                ORDER BY timestamp DESC LIMIT 1
            """, {"cid": cid})
            if leit:
                r = leit[0]
                ts = r[0].strftime("%d/%m %H:%M") if r[0] else "n/d"
                vib_label = ("normal" if r[2] is None or r[2] < 2.8 else "atenção" if r[2] < 4.5 else "CRÍTICA")
                results.append(
                    f"  Última leitura [{ts}]: Temp={r[1] or 'n/d'}°C | "
                    f"Vibração={r[2] or 'n/d'} mm/s ({vib_label}) | RPM={r[3] or 'n/d'}"
                )
            else:
                results.append("  Sem leituras.")

            # Diagnóstico mais recente
            diag = _exec("""
                SELECT overall_status, health_score, health_index, risk_level, is_anomaly, rul_hours, recommendation
                FROM diagnostico WHERE componente_id = :cid
                ORDER BY timestamp DESC LIMIT 1
            """, {"cid": cid})
            if diag:
                d = diag[0]
                hs = d[2] if d[2] is not None else (d[1] * 100 if d[1] is not None else None)
                rul = f"{d[5]:.1f}h" if d[5] is not None else "n/d"
                results.append(
                    f"  ML: status={d[0]} | saúde={f'{hs:.1f}%' if hs else 'n/d'} | "
                    f"risco={d[3] or 'n/d'} | anomalia={'SIM' if d[4] else 'não'} | RUL={rul}"
                )
                results.append(f"  → {d[6] or 'sem recomendação'}")
            else:
                results.append("  Sem diagnósticos.")

            # Alertas ativos
            alerts = _exec("SELECT COUNT(*) FROM alerts WHERE motor_id = :mid AND resolved_at IS NULL", {"mid": cid})
            results.append(f"  Alertas ativos: {alerts[0][0] if alerts else 0}")
            results.append("")

        return "\n".join(results)
    except Exception as exc:
        return f"Erro ao comparar motores: {exc}"


@tool
def get_maintenance_records(motor_id: int = 1, limit: int = 10) -> str:
    """Busca registros de manutenção (preventiva, corretiva, preditiva) de um motor.
    Use para histórico de intervenções, planejamento ou relatório de manutenção."""
    try:
        rows = _exec("""
            SELECT id, type, scheduled_at, completed_at, notes, created_at
            FROM maintenance WHERE motor_id = :mid
            ORDER BY scheduled_at DESC LIMIT :lim
        """, {"mid": motor_id, "lim": limit})

        if not rows:
            return f"Nenhum registro de manutenção para motor {motor_id}."

        lines = [f"Manutenções — motor {motor_id} ({len(rows)} registros):"]
        for r in rows:
            sched = r[2].strftime("%d/%m/%Y") if r[2] else "n/d"
            done = r[3].strftime("%d/%m/%Y") if r[3] else "pendente"
            lines.append(
                f"  [{sched}] {r[1].upper()} | Realizada: {done}\n"
                f"    Obs: {r[4] or 'sem observações'}"
            )
        return "\n".join(lines)
    except Exception as exc:
        return f"Erro ao buscar manutenções: {exc}"
