"""
Tool LangChain — gera relatório PDF de análise do motor Forzy.
Retorna um token PDF::<filename> que o router converte em report_url.
"""

from __future__ import annotations

import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.tools import tool

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


def _get_db_data(componente_id: int) -> tuple[dict | None, list[dict]]:
    """Busca último diagnóstico e últimas leituras do banco."""
    import src.models.user, src.models.planta, src.models.maquina  # noqa: F401
    import src.models.componente, src.models.atributo              # noqa: F401
    import src.models.leitura_sensor, src.models.alert             # noqa: F401
    import src.models.maintenance, src.models.diagnostico          # noqa: F401

    from src.db.postgres import SessionLocal
    from src.models.diagnostico import Diagnostico
    from src.models.leitura_sensor import LeituraSensor

    db = SessionLocal()
    try:
        diag = (
            db.query(Diagnostico)
            .filter(Diagnostico.componente_id == componente_id)
            .order_by(Diagnostico.timestamp.desc())
            .first()
        )
        leituras = (
            db.query(LeituraSensor)
            .filter(LeituraSensor.componente_id == componente_id)
            .order_by(LeituraSensor.timestamp.desc())
            .limit(10)
            .all()
        )
        diag_dict = None
        if diag:
            diag_dict = {
                "timestamp": diag.timestamp,
                "overall_status": diag.overall_status,
                "is_anomaly": diag.is_anomaly,
                "lstm_severity": diag.lstm_severity,
                "risk_level": diag.risk_level,
                "health_index": diag.health_index,
                "health_score": diag.health_score,
                "rul_hours": diag.rul_hours,
                "maintenance_window_days": diag.maintenance_window_days,
                "recommendation": diag.recommendation,
            }
        leituras_list = [
            {
                "timestamp": l.timestamp,
                "vibracao": l.vibracao,
                "temperatura": l.temperatura,
                "rpm": l.rpm,
                "corrente": l.corrente,
                "voltagem": l.voltagem,
            }
            for l in leituras
        ]
        return diag_dict, leituras_list
    finally:
        db.close()


def _status_color(status: str):
    from reportlab.lib import colors
    return {
        "critical": colors.HexColor("#ef4444"),
        "warning":  colors.HexColor("#f59e0b"),
        "healthy":  colors.HexColor("#22c55e"),
    }.get(status, colors.HexColor("#6b7280"))


def _build_pdf(componente_id: int, diag: dict | None, leituras: list[dict]) -> Path:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
    )

    filename = f"relatorio_motor_{componente_id}_{uuid.uuid4().hex[:8]}.pdf"
    filepath = REPORTS_DIR / filename

    doc = SimpleDocTemplate(
        str(filepath),
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
    )

    styles = getSampleStyleSheet()
    PURPLE = colors.HexColor("#7c3aed")
    DARK   = colors.HexColor("#0f172a")
    GRAY   = colors.HexColor("#64748b")
    BG     = colors.HexColor("#f8fafc")

    title_style = ParagraphStyle(
        "Title", parent=styles["Normal"],
        fontSize=18, fontName="Helvetica-Bold",
        textColor=DARK, spaceAfter=2,
    )
    sub_style = ParagraphStyle(
        "Sub", parent=styles["Normal"],
        fontSize=10, fontName="Helvetica",
        textColor=GRAY, spaceAfter=12,
    )
    section_style = ParagraphStyle(
        "Section", parent=styles["Normal"],
        fontSize=11, fontName="Helvetica-Bold",
        textColor=PURPLE, spaceBefore=14, spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "Body", parent=styles["Normal"],
        fontSize=9, fontName="Helvetica",
        textColor=DARK, leading=14,
    )

    now = datetime.now(timezone.utc).strftime("%d/%m/%Y %H:%M UTC")
    story = []

    # ── Cabeçalho ──────────────────────────────────────────────────────────
    story.append(Paragraph("Relatório de Análise do Motor", title_style))
    story.append(Paragraph(f"Forzy Digital Twin · Componente {componente_id} · Gerado em {now}", sub_style))
    story.append(HRFlowable(width="100%", thickness=1, color=PURPLE, spaceAfter=10))

    # ── Diagnóstico ML ─────────────────────────────────────────────────────
    story.append(Paragraph("Diagnóstico ML", section_style))

    if diag:
        status = diag["overall_status"]
        status_color = _status_color(status)
        health = (
            f"{diag['health_index']:.1f}%" if diag["health_index"] is not None else
            f"{diag['health_score'] * 100:.1f}%" if diag["health_score"] is not None else "N/D"
        )
        rul = f"{diag['rul_hours']:.1f} h" if diag["rul_hours"] is not None else "N/D"
        janela = f"{diag['maintenance_window_days']:.1f} dias" if diag["maintenance_window_days"] is not None else "N/D"
        ts_diag = diag["timestamp"].strftime("%d/%m/%Y %H:%M UTC") if diag["timestamp"] else "N/D"

        diag_data = [
            ["Campo", "Valor"],
            ["Data do diagnóstico", ts_diag],
            ["Status geral", status.upper()],
            ["Anomalia detectada", "SIM" if diag["is_anomaly"] else "Não"],
            ["Severidade LSTM", diag["lstm_severity"] or "N/D"],
            ["Nível de risco", diag["risk_level"] or "N/D"],
            ["Índice de saúde", health],
            ["RUL (vida útil restante)", rul],
            ["Janela de manutenção", janela],
        ]

        tbl = Table(diag_data, colWidths=[80 * mm, 90 * mm])
        tbl.setStyle(TableStyle([
            ("BACKGROUND",   (0, 0), (-1, 0), PURPLE),
            ("TEXTCOLOR",    (0, 0), (-1, 0), colors.white),
            ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",     (0, 0), (-1, -1), 9),
            ("BACKGROUND",   (0, 1), (-1, -1), BG),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, BG]),
            ("TEXTCOLOR",    (0, 1), (0, -1), GRAY),
            ("TEXTCOLOR",    (1, 3), (1, 3), status_color),  # status
            ("FONTNAME",     (1, 3), (1, 3), "Helvetica-Bold"),
            ("GRID",         (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ("ROWPADDING",   (0, 0), (-1, -1), 6),
            ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
        ]))
        story.append(tbl)
        story.append(Spacer(1, 8))

        if diag["recommendation"]:
            story.append(Paragraph("Recomendação:", ParagraphStyle(
                "Rec", parent=body_style, fontName="Helvetica-Bold", spaceAfter=3,
            )))
            story.append(Paragraph(diag["recommendation"], body_style))
    else:
        story.append(Paragraph("Nenhum diagnóstico disponível no banco de dados.", body_style))

    # ── Leituras recentes ──────────────────────────────────────────────────
    story.append(Paragraph("Leituras Recentes dos Sensores", section_style))

    if leituras:
        leit_data = [["Timestamp", "Vibração (g)", "Temp (°C)", "RPM", "Corrente (A)", "Voltagem (V)"]]
        for l in leituras:
            ts = l["timestamp"].strftime("%d/%m %H:%M") if l["timestamp"] else "N/D"
            leit_data.append([
                ts,
                f"{l['vibracao']:.3f}" if l["vibracao"] is not None else "N/D",
                f"{l['temperatura']:.1f}" if l["temperatura"] is not None else "N/D",
                f"{l['rpm']:.2f}" if l["rpm"] is not None else "N/D",
                f"{l['corrente']:.2f}" if l["corrente"] is not None else "N/D",
                f"{l['voltagem']:.1f}" if l["voltagem"] is not None else "N/D",
            ])

        col_w = [38 * mm, 28 * mm, 26 * mm, 22 * mm, 28 * mm, 28 * mm]
        tbl2 = Table(leit_data, colWidths=col_w)
        tbl2.setStyle(TableStyle([
            ("BACKGROUND",     (0, 0), (-1, 0), PURPLE),
            ("TEXTCOLOR",      (0, 0), (-1, 0), colors.white),
            ("FONTNAME",       (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",       (0, 0), (-1, -1), 8),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, BG]),
            ("GRID",           (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ("ROWPADDING",     (0, 0), (-1, -1), 5),
            ("ALIGN",          (1, 0), (-1, -1), "CENTER"),
            ("VALIGN",         (0, 0), (-1, -1), "MIDDLE"),
        ]))
        story.append(tbl2)
    else:
        story.append(Paragraph("Nenhuma leitura disponível.", body_style))

    # ── Referências técnicas ───────────────────────────────────────────────
    story.append(Paragraph("Referências Técnicas", section_style))
    refs = [
        "Motor WEG W22 3cv · Norma ABNT NBR 17094 · IP55 · 3600 rpm",
        "Sensor: Pepperl+Fuchs VIM32PL-E1AC8-0RE-IO-1V1401 · Range: 0–128 mm/s",
        "ISO 10816: Vibração < 2.8 mm/s = normal · 2.8–4.5 mm/s = atenção · > 4.5 mm/s = crítica",
        "Temperatura máxima de operação: 80°C",
    ]
    for ref in refs:
        story.append(Paragraph(f"• {ref}", body_style))

    story.append(Spacer(1, 14))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#e2e8f0")))
    story.append(Paragraph(
        f"Documento gerado automaticamente pelo sistema Forzy Digital Twin em {now}.",
        ParagraphStyle("Footer", parent=body_style, textColor=GRAY, fontSize=8, spaceBefore=6),
    ))

    doc.build(story)
    return filepath


@tool
def generate_motor_report(componente_id: int = 1) -> str:
    """Gera um relatório PDF completo de análise do motor com diagnóstico ML e leituras recentes.
    Use quando o usuário pedir um relatório, documento, laudo ou resumo exportável do motor.
    Retorna um token que o sistema converte em link de download."""
    try:
        diag, leituras = _get_db_data(componente_id)
        filepath = _build_pdf(componente_id, diag, leituras)
        return f"PDF::{filepath.name}"
    except Exception as exc:
        return f"Erro ao gerar relatório: {exc}"
