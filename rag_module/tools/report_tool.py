"""
Tools LangChain — geração de relatórios PDF, Excel e Word do motor Forzy.
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


def _get_db_data(componente_id: int, motor_id: int = 1):
    from sqlalchemy import text
    from src.db.postgres import SessionLocal
    db = SessionLocal()
    try:
        diag = db.execute(text("""
            SELECT timestamp, overall_status, is_anomaly, lstm_severity,
                   risk_level, health_score, health_index, rul_hours,
                   maintenance_window_days, recommendation
            FROM diagnostico WHERE componente_id = :cid
            ORDER BY timestamp DESC LIMIT 1
        """), {"cid": componente_id}).fetchone()

        leituras = db.execute(text("""
            SELECT timestamp, temperatura, umidade, corrente, voltagem, rpm, vibracao, inclinacao
            FROM leitura_sensor WHERE componente_id = :cid
            ORDER BY timestamp DESC LIMIT 20
        """), {"cid": componente_id}).fetchall()

        alerts = db.execute(text("""
            SELECT severity, message, created_at, resolved_at
            FROM alerts WHERE motor_id = :mid
            ORDER BY created_at DESC LIMIT 5
        """), {"mid": motor_id}).fetchall()

        motor = db.execute(text("""
            SELECT mq.nome, mq.tipo, NULL, esp.tensao_nominal, esp.corrente_nominal,
                   esp.rpm_nominal, esp.potencia_kw, pl.nome as planta, pl.localizacao
            FROM maquina mq
            LEFT JOIN componente c ON c.maquina_id = mq.id
            LEFT JOIN especificacao_motor esp ON esp.componente_id = c.id
            LEFT JOIN planta pl ON pl.id = mq.planta_id
            WHERE mq.id = :mid
        """), {"mid": motor_id}).fetchone()

        return diag, list(leituras), list(alerts), motor
    finally:
        db.close()


# ── PDF ────────────────────────────────────────────────────────────────────────

def _build_pdf(componente_id: int, diag, leituras: list, alerts: list, motor) -> Path:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    filename = f"relatorio_motor_{componente_id}_{uuid.uuid4().hex[:8]}.pdf"
    filepath = REPORTS_DIR / filename

    doc = SimpleDocTemplate(str(filepath), pagesize=A4,
                            leftMargin=20*mm, rightMargin=20*mm,
                            topMargin=20*mm, bottomMargin=20*mm)

    styles = getSampleStyleSheet()
    PURPLE = colors.HexColor("#7c3aed")
    DARK   = colors.HexColor("#0f172a")
    GRAY   = colors.HexColor("#64748b")
    BG     = colors.HexColor("#f8fafc")

    title_s   = ParagraphStyle("T", parent=styles["Normal"], fontSize=18, fontName="Helvetica-Bold", textColor=DARK, spaceAfter=2)
    sub_s     = ParagraphStyle("S", parent=styles["Normal"], fontSize=10, fontName="Helvetica", textColor=GRAY, spaceAfter=12)
    section_s = ParagraphStyle("Se", parent=styles["Normal"], fontSize=11, fontName="Helvetica-Bold", textColor=PURPLE, spaceBefore=14, spaceAfter=6)
    body_s    = ParagraphStyle("B", parent=styles["Normal"], fontSize=9, fontName="Helvetica", textColor=DARK, leading=14)

    now = datetime.now(timezone.utc).strftime("%d/%m/%Y %H:%M UTC")
    story = []

    story.append(Paragraph("Relatório de Análise do Motor", title_s))
    motor_name = motor[0] if motor else f"Componente {componente_id}"
    planta = f"{motor[7]} — {motor[8]}" if motor and motor[7] else "N/D"
    story.append(Paragraph(f"Forzy Digital Twin · {motor_name} · {planta} · Gerado em {now}", sub_s))
    story.append(HRFlowable(width="100%", thickness=1, color=PURPLE, spaceAfter=10))

    if motor:
        story.append(Paragraph("Especificações do Motor", section_s))
        spec_data = [
            ["Campo", "Valor"],
            ["Motor", motor[0]], ["Tipo", motor[1] or "N/D"], ["Serial", motor[2] or "N/D"],
            ["Tensão nominal", f"{motor[3]}V" if motor[3] else "N/D"],
            ["Corrente nominal", f"{motor[4]}A" if motor[4] else "N/D"],
            ["Rotação nominal (rpm)", str(motor[5]) if motor[5] else "N/D"],
            ["Potência", f"{motor[6]}kW" if motor[6] else "N/D"],
        ]
        tbl = Table(spec_data, colWidths=[70*mm, 100*mm])
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0,0),(-1,0), PURPLE), ("TEXTCOLOR",(0,0),(-1,0), colors.white),
            ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"), ("FONTSIZE",(0,0),(-1,-1),9),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white, BG]),
            ("TEXTCOLOR",(0,1),(0,-1), GRAY),
            ("GRID",(0,0),(-1,-1),0.5, colors.HexColor("#e2e8f0")),
            ("ROWPADDING",(0,0),(-1,-1),6), ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ]))
        story.append(tbl); story.append(Spacer(1,8))

    story.append(Paragraph("Diagnóstico ML", section_s))
    if diag:
        status_map = {"critical": colors.HexColor("#ef4444"), "warning": colors.HexColor("#f59e0b"), "healthy": colors.HexColor("#22c55e")}
        hs = diag[6] if diag[6] is not None else (diag[5]*100 if diag[5] else None)
        diag_data = [
            ["Campo", "Valor"],
            ["Data", diag[0].strftime("%d/%m/%Y %H:%M UTC") if diag[0] else "N/D"],
            ["Status", str(diag[1]).upper()], ["Anomalia", "SIM" if diag[2] else "Não"],
            ["Severidade LSTM", diag[3] or "N/D"], ["Nível de risco", diag[4] or "N/D"],
            ["Índice de saúde", f"{hs:.1f}%" if hs else "N/D"],
            ["RUL", f"{diag[7]:.1f}h" if diag[7] else "N/D"],
            ["Próx. manutenção", f"{diag[8]:.1f} dias" if diag[8] else "N/D"],
        ]
        tbl = Table(diag_data, colWidths=[70*mm, 100*mm])
        status_color = status_map.get(str(diag[1]).lower(), GRAY)
        tbl.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),PURPLE), ("TEXTCOLOR",(0,0),(-1,0),colors.white),
            ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"), ("FONTSIZE",(0,0),(-1,-1),9),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,BG]),
            ("TEXTCOLOR",(0,1),(0,-1),GRAY), ("TEXTCOLOR",(1,2),(1,2),status_color),
            ("FONTNAME",(1,2),(1,2),"Helvetica-Bold"),
            ("GRID",(0,0),(-1,-1),0.5,colors.HexColor("#e2e8f0")),
            ("ROWPADDING",(0,0),(-1,-1),6), ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ]))
        story.append(tbl); story.append(Spacer(1,6))
        if diag[9]:
            story.append(Paragraph(f"<b>Recomendação:</b> {diag[9]}", body_s))
    else:
        story.append(Paragraph("Nenhum diagnóstico disponível.", body_s))

    story.append(Paragraph("Leituras Recentes dos Sensores", section_s))
    if leituras:
        leit_data = [["Data/Hora", "Temp (°C)", "Velocidade (mm/s)", "Aceleração (g)", "Corrente (A)", "Voltagem (V)"]]
        for l in leituras:
            ts = l[0].strftime("%d/%m %H:%M") if l[0] else "N/D"
            leit_data.append([ts,
                f"{l[1]:.1f}" if l[1] is not None else "—",
                f"{l[5]:.3f}" if l[5] is not None else "—",
                f"{l[6]:.3f}" if l[6] is not None else "—",
                f"{l[3]:.2f}" if l[3] is not None else "—",
                f"{l[4]:.1f}" if l[4] is not None else "—",
            ])
        tbl = Table(leit_data, colWidths=[35*mm,25*mm,25*mm,32*mm,28*mm,28*mm])
        tbl.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),PURPLE), ("TEXTCOLOR",(0,0),(-1,0),colors.white),
            ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"), ("FONTSIZE",(0,0),(-1,-1),8),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,BG]),
            ("GRID",(0,0),(-1,-1),0.5,colors.HexColor("#e2e8f0")),
            ("ROWPADDING",(0,0),(-1,-1),5), ("ALIGN",(1,0),(-1,-1),"CENTER"),
            ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ]))
        story.append(tbl)
    else:
        story.append(Paragraph("Nenhuma leitura disponível.", body_s))

    if alerts:
        story.append(Paragraph("Alertas Recentes", section_s))
        for a in alerts:
            ts = a[2].strftime("%d/%m/%Y %H:%M") if a[2] else "N/D"
            status = "resolvido" if a[3] else "ATIVO"
            story.append(Paragraph(f"• [{ts}] {str(a[0]).upper()} — {a[1]} ({status})", body_s))

    story.append(Paragraph("Referências Técnicas", section_s))
    for ref in [
        "Motor WEG W22 3cv · ABNT NBR 17094 · IP55 · 3600 rpm",
        "Sensor: Pepperl+Fuchs VIM32PL-E1AC8-0RE-IO-1V1401 · Range: 0–128 mm/s",
        "ISO 10816: < 2.8 mm/s = normal · 2.8–4.5 mm/s = atenção · > 4.5 mm/s = crítica",
        "Temperatura máxima de operação: 80°C",
    ]:
        story.append(Paragraph(f"• {ref}", body_s))

    story.append(Spacer(1,14))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#e2e8f0")))
    story.append(Paragraph(
        f"Documento gerado automaticamente pelo Forzy Digital Twin em {now}.",
        ParagraphStyle("Footer", parent=body_s, textColor=GRAY, fontSize=8, spaceBefore=6)
    ))
    doc.build(story)
    return filepath


# ── Excel ──────────────────────────────────────────────────────────────────────

def _build_excel(componente_id: int, diag, leituras: list, alerts: list, motor) -> Path:
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

    filename = f"relatorio_motor_{componente_id}_{uuid.uuid4().hex[:8]}.xlsx"
    filepath = REPORTS_DIR / filename

    wb = openpyxl.Workbook()
    PURPLE = "7C3AED"
    LIGHT  = "F3F0FF"
    WHITE  = "FFFFFF"
    GRAY   = "64748B"

    def header_style(cell, bg=PURPLE):
        cell.font = Font(bold=True, color=WHITE, size=10)
        cell.fill = PatternFill("solid", fgColor=bg)
        cell.alignment = Alignment(horizontal="center", vertical="center")

    def border_thin():
        s = Side(style="thin", color="E2E8F0")
        return Border(left=s, right=s, top=s, bottom=s)

    now = datetime.now(timezone.utc).strftime("%d/%m/%Y %H:%M UTC")

    # ── Aba: Resumo ──
    ws = wb.active
    ws.title = "Resumo"
    ws["A1"] = "RELATÓRIO FORZY DIGITAL TWIN"
    ws["A1"].font = Font(bold=True, size=14, color=PURPLE)
    ws["A2"] = f"Motor: {motor[0] if motor else 'N/D'} | Gerado em: {now}"
    ws["A2"].font = Font(color=GRAY, size=9)
    ws.append([])

    if motor:
        ws.append(["ESPECIFICAÇÕES DO MOTOR"])
        ws["A4"].font = Font(bold=True, color=PURPLE)
        for label, val in [
            ("Nome", motor[0]), ("Tipo", motor[1]), ("Serial", motor[2]),
            ("Tensão nominal", f"{motor[3]}V"), ("Corrente nominal", f"{motor[4]}A"),
            ("Rotação nominal (rpm)", motor[5]), ("Potência", f"{motor[6]}kW"),
            ("Planta", motor[7]), ("Localização", motor[8]),
        ]:
            ws.append([label, val])

    ws.append([])
    if diag:
        ws.append(["DIAGNÓSTICO ML MAIS RECENTE"])
        ws[f"A{ws.max_row}"].font = Font(bold=True, color=PURPLE)
        hs = diag[6] if diag[6] is not None else (diag[5]*100 if diag[5] else None)
        for label, val in [
            ("Data", diag[0].strftime("%d/%m/%Y %H:%M UTC") if diag[0] else "N/D"),
            ("Status", str(diag[1]).upper()), ("Anomalia", "SIM" if diag[2] else "Não"),
            ("Severidade LSTM", diag[3] or "N/D"), ("Nível de risco", diag[4] or "N/D"),
            ("Índice de saúde", f"{hs:.1f}%" if hs else "N/D"),
            ("RUL (h)", f"{diag[7]:.1f}" if diag[7] else "N/D"),
            ("Próx. manutenção (dias)", f"{diag[8]:.1f}" if diag[8] else "N/D"),
            ("Recomendação", diag[9] or "N/D"),
        ]:
            ws.append([label, val])

    ws.column_dimensions["A"].width = 28
    ws.column_dimensions["B"].width = 50

    # ── Aba: Leituras ──
    ws2 = wb.create_sheet("Leituras de Sensor")
    headers = ["Timestamp", "Temperatura (°C)", "Umidade (%)", "Corrente (A)", "Voltagem (V)", "Velocidade de vibração (mm/s)", "Aceleração (g)", "Inclinação"]
    ws2.append(headers)
    for i, h in enumerate(headers, 1):
        cell = ws2.cell(1, i, h)
        header_style(cell)
        cell.border = border_thin()
    for l in leituras:
        row = [
            l[0].strftime("%d/%m/%Y %H:%M:%S") if l[0] else "",
            round(l[1], 2) if l[1] else None,
            round(l[2], 2) if l[2] else None,
            round(l[3], 3) if l[3] else None,
            round(l[4], 1) if l[4] else None,
            round(l[5], 4) if l[5] else None,
            round(l[6], 4) if l[6] else None,
            round(l[7], 4) if l[7] else None,
        ]
        ws2.append(row)
        for i in range(1, 9):
            ws2.cell(ws2.max_row, i).border = border_thin()
    for col in ws2.columns:
        ws2.column_dimensions[col[0].column_letter].width = 20

    # ── Aba: Alertas ──
    ws3 = wb.create_sheet("Alertas")
    headers3 = ["Severidade", "Mensagem", "Score Anomalia", "RUL Estimado (h)", "Criado em", "Resolvido em"]
    ws3.append(headers3)
    for i, h in enumerate(headers3, 1):
        header_style(ws3.cell(1, i, h))
    for a in alerts:
        ws3.append([
            str(a[0]).upper(), a[1],
            round(a[2], 4) if a[2] else None,
            round(a[3], 1) if a[3] else None,
            a[4].strftime("%d/%m/%Y %H:%M") if a[4] else None,
            a[5].strftime("%d/%m/%Y %H:%M") if a[5] else None,
        ])
    ws3.column_dimensions["A"].width = 14
    ws3.column_dimensions["B"].width = 50
    for c in ["C","D","E","F"]:
        ws3.column_dimensions[c].width = 20

    wb.save(str(filepath))
    return filepath


# ── Word ───────────────────────────────────────────────────────────────────────

def _build_word(componente_id: int, diag, leituras: list, alerts: list, motor) -> Path:
    from docx import Document
    from docx.shared import Pt, RGBColor, Inches
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement

    filename = f"relatorio_motor_{componente_id}_{uuid.uuid4().hex[:8]}.docx"
    filepath = REPORTS_DIR / filename

    doc = Document()
    PURPLE = RGBColor(0x7C, 0x3A, 0xED)
    DARK   = RGBColor(0x0F, 0x17, 0x2A)
    GRAY   = RGBColor(0x64, 0x74, 0x8B)
    now    = datetime.now(timezone.utc).strftime("%d/%m/%Y %H:%M UTC")

    def add_heading(text, level=1, color=PURPLE):
        p = doc.add_heading(text, level)
        for run in p.runs:
            run.font.color.rgb = color
        return p

    def add_table_row(table, cells, bold_first=False):
        row = table.add_row()
        for i, val in enumerate(cells):
            cell = row.cells[i]
            cell.text = str(val) if val is not None else "—"
            if bold_first and i == 0:
                cell.paragraphs[0].runs[0].bold = True

    title = doc.add_heading("Relatório de Análise do Motor", 0)
    for run in title.runs:
        run.font.color.rgb = DARK
    sub = doc.add_paragraph()
    run = sub.add_run(f"Forzy Digital Twin · {motor[0] if motor else f'Componente {componente_id}'} · Gerado em {now}")
    run.font.size = Pt(9); run.font.color.rgb = GRAY

    if motor:
        add_heading("Especificações do Motor", 1)
        tbl = doc.add_table(rows=1, cols=2)
        tbl.style = "Light Shading Accent 1"
        tbl.rows[0].cells[0].text = "Campo"
        tbl.rows[0].cells[1].text = "Valor"
        specs = [("Nome", motor[0]), ("Tipo", motor[1] or "N/D"), ("Serial", motor[2] or "N/D"),
                 ("Tensão nominal", f"{motor[3]}V"), ("Corrente nominal", f"{motor[4]}A"),
                 ("Rotação nominal (rpm)", str(motor[5])), ("Potência", f"{motor[6]}kW"),
                 ("Planta", motor[7] or "N/D"), ("Localização", motor[8] or "N/D")]
        for label, val in specs:
            add_table_row(tbl, [label, val], bold_first=True)

    doc.add_paragraph()
    add_heading("Diagnóstico ML", 1)
    if diag:
        hs = diag[6] if diag[6] is not None else (diag[5]*100 if diag[5] else None)
        tbl2 = doc.add_table(rows=1, cols=2)
        tbl2.style = "Light Shading Accent 1"
        tbl2.rows[0].cells[0].text = "Campo"
        tbl2.rows[0].cells[1].text = "Valor"
        for label, val in [
            ("Data", diag[0].strftime("%d/%m/%Y %H:%M UTC") if diag[0] else "N/D"),
            ("Status", str(diag[1]).upper()), ("Anomalia", "SIM" if diag[2] else "Não"),
            ("Severidade LSTM", diag[3] or "N/D"), ("Nível de risco", diag[4] or "N/D"),
            ("Índice de saúde", f"{hs:.1f}%" if hs else "N/D"),
            ("RUL", f"{diag[7]:.1f}h" if diag[7] else "N/D"),
            ("Próx. manutenção", f"{diag[8]:.1f} dias" if diag[8] else "N/D"),
        ]:
            add_table_row(tbl2, [label, val], bold_first=True)
        doc.add_paragraph()
        rec = doc.add_paragraph()
        rec.add_run("Recomendação: ").bold = True
        rec.add_run(diag[9] or "Sem recomendação disponível.")
    else:
        doc.add_paragraph("Nenhum diagnóstico disponível no banco de dados.")

    doc.add_paragraph()
    add_heading("Leituras Recentes dos Sensores", 1)
    if leituras:
        tbl3 = doc.add_table(rows=1, cols=6)
        tbl3.style = "Light Shading Accent 1"
        for i, h in enumerate(["Data/Hora", "Temp (°C)", "Velocidade (mm/s)", "Aceleração (g)", "Corrente (A)", "Voltagem (V)"]):
            tbl3.rows[0].cells[i].text = h
        for l in leituras:
            row = tbl3.add_row()
            row.cells[0].text = l[0].strftime("%d/%m %H:%M") if l[0] else "—"
            row.cells[1].text = f"{l[1]:.1f}" if l[1] is not None else "—"
            row.cells[2].text = f"{l[5]:.3f}" if l[5] is not None else "—"
            row.cells[3].text = f"{l[6]:.3f}" if l[6] is not None else "—"
            row.cells[4].text = f"{l[3]:.2f}" if l[3] is not None else "—"
            row.cells[5].text = f"{l[4]:.1f}" if l[4] is not None else "—"

    if alerts:
        doc.add_paragraph()
        add_heading("Alertas Recentes", 1)
        for a in alerts:
            ts = a[2].strftime("%d/%m/%Y %H:%M") if a[2] else "N/D"
            status = "resolvido" if a[3] else "ATIVO"
            p = doc.add_paragraph(style="List Bullet")
            p.add_run(f"[{ts}] {str(a[0]).upper()} — ").bold = True
            p.add_run(f"{a[1]} ({status})")

    doc.add_paragraph()
    add_heading("Referências Técnicas", 1)
    for ref in [
        "Motor WEG W22 3cv · ABNT NBR 17094 · IP55 · 3600 rpm",
        "Sensor: Pepperl+Fuchs VIM32PL-E1AC8-0RE-IO-1V1401 · Range: 0–128 mm/s",
        "ISO 10816: < 2.8 mm/s = normal · 2.8–4.5 mm/s = atenção · > 4.5 mm/s = crítica",
        "Temperatura máxima de operação: 80°C",
    ]:
        doc.add_paragraph(ref, style="List Bullet")

    footer = doc.sections[0].footer.paragraphs[0]
    footer.text = f"Forzy Digital Twin — documento gerado automaticamente em {now}"
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.save(str(filepath))
    return filepath


# ── Tool principal ─────────────────────────────────────────────────────────────

@tool
def generate_report(format: str = "pdf", componente_id: int = 1, motor_id: int = 1) -> str:
    """Gera relatório completo do motor nos formatos PDF, Excel (.xlsx) ou Word (.docx).
    Inclui especificações do motor, diagnóstico ML, leituras recentes e alertas.

    Parâmetros:
      format: 'pdf' | 'excel' | 'xlsx' | 'word' | 'docx'
      componente_id: ID do componente (padrão: 1)
      motor_id: ID do motor (padrão: 1)

    Use quando o usuário pedir relatório, laudo, documento ou exportação de dados.
    Retorna token REPORT:: com nome do arquivo para download."""
    try:
        fmt = format.lower().strip()
        diag, leituras, alerts, motor = _get_db_data(componente_id, motor_id)

        if fmt in ("excel", "xlsx"):
            # busca alertas com todos os campos necessários
            from sqlalchemy import text
            from src.db.postgres import SessionLocal
            db = SessionLocal()
            try:
                alerts_full = db.execute(text("""
                    SELECT severity, message, anomaly_score, rul_estimated, created_at, resolved_at
                    FROM alerts WHERE motor_id = :mid ORDER BY created_at DESC LIMIT 10
                """), {"mid": motor_id}).fetchall()
            finally:
                db.close()
            filepath = _build_excel(componente_id, diag, leituras, alerts_full, motor)
            return f"REPORT::{filepath.name}"

        if fmt in ("word", "docx"):
            filepath = _build_word(componente_id, diag, leituras, alerts, motor)
            return f"REPORT::{filepath.name}"

        filepath = _build_pdf(componente_id, diag, leituras, alerts, motor)
        return f"REPORT::{filepath.name}"

    except Exception as exc:
        return f"Erro ao gerar relatório: {exc}"


# mantém compatibilidade com código anterior
@tool
def generate_motor_report(componente_id: int = 1) -> str:
    """Alias: gera relatório PDF do motor. Prefira generate_report com o parâmetro format."""
    try:
        diag, leituras, alerts, motor = _get_db_data(componente_id)
        filepath = _build_pdf(componente_id, diag, leituras, alerts, motor)
        return f"PDF::{filepath.name}"
    except Exception as exc:
        return f"Erro ao gerar relatório: {exc}"
