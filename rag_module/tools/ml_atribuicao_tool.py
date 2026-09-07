"""
Tools do agente para os modelos de novidade Forzy — atribuição por componente.

Dão ao agente acesso ao que os 6 modelos (2 motores x 3 regimes) estão dizendo
agora, e às métricas da bancada de injeção de falhas, para que ele possa explicar
o diagnóstico com números reais e com as ressalvas corretas.
"""

from __future__ import annotations

import sys
from pathlib import Path

from langchain_core.tools import tool

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_MOTOR = {2: "S1 (MOTOR-01)", 3: "S2 (MOTOR-02)", 1: "FIAP"}


@tool
def get_ml_atribuicao(componente_id: int = 2, janela_min: int = 15) -> str:
    """Roda os modelos de novidade na janela mais recente e devolve a atribuição
    por componente: qual falha é provável, quais peças do motor são candidatas e
    quais segmentos do modelo 3D estão destacados.

    Use quando o usuário perguntar o que o modelo está detectando agora, por que
    uma peça ficou vermelha no modelo 3D, ou qual componente pode estar com
    problema.

    Parâmetros:
      componente_id: 2 = motor S1, 3 = motor S2, 1 = motor da FIAP
      janela_min: minutos de leituras a considerar (padrão 15)
    """
    from datetime import datetime, timedelta, timezone

    # Toda a query fica dentro de try/except como nas outras 20 tools: uma falha de
    # banco (conexão caída, pool esgotado) tem que virar string de erro, não
    # exceção propagando pelo ToolNode do LangGraph e derrubando o turno inteiro.
    # SQL direto em vez do ORM, seguindo o padrão do db_tool. Usar
    # `leitura_service` arrastava os mappers do SQLAlchemy, e se algum modelo não
    # tivesse sido importado ainda a tool falhava com erro de mapper em vez de
    # devolver dado — fragilidade desnecessária para uma consulta de leitura.
    try:
        from sqlalchemy import text

        from ml_module.forzy import infer
        from src.db.postgres import SessionLocal

        db = SessionLocal()
        try:
            agora = datetime.now(timezone.utc)
            rows = db.execute(text("""
                SELECT timestamp, rpm, vibracao, temperatura
                  FROM leitura_sensor
                 WHERE componente_id = :cid
                   AND timestamp >= :desde
                   AND rpm IS NOT NULL
                   AND vibracao IS NOT NULL
                   AND temperatura IS NOT NULL
                 ORDER BY timestamp ASC
                 LIMIT 5000
            """), {"cid": int(componente_id),
                   "desde": agora - timedelta(minutes=int(janela_min))}).fetchall()
        finally:
            db.close()
    except Exception as exc:
        return f"Erro ao consultar a atribuição do componente {componente_id}: {exc}"

    # `rpm` guarda a VELOCIDADE de vibração em mm/s e `vibracao` guarda a
    # ACELERAÇÃO em g — os nomes das colunas enganam.
    leituras = [
        {"timestamp": r[0], "v_rms": r[1], "a_rms": r[2], "temp_c": r[3]}
        for r in rows
    ]

    seg_parado = None
    if leituras:
        ativos = [l for l in leituras if (l["v_rms"] or 0) > 0.30]
        if ativos and ativos[-1] is not leituras[-1]:
            seg_parado = (leituras[-1]["timestamp"] - ativos[-1]["timestamp"]).total_seconds()

    r = infer.avaliar(leituras, int(componente_id), seg_parado)
    nome = _MOTOR.get(int(componente_id), str(componente_id))

    if not r.get("disponivel"):
        txt = f"MOTOR {nome} — atribuição não disponível.\nMotivo: {r.get('motivo')}"
        if r.get("heat_soak"):
            txt += ("\n\nATENÇÃO: o motor foi desligado há menos de 15 minutos. Temperatura "
                    "elevada agora é HEAT SOAK (o calor dos enrolamentos migra para a carcaça "
                    "com o ventilador IC411 já parado) — medido: pico de 46 °C cerca de 7,9 min "
                    "após desligar. Isso é normal, não é sobreaquecimento.")
        for lim in r.get("limitacoes", []):
            txt += f"\n- {lim}"
        return txt

    a = r.get("atribuicao") or {}
    linhas_saida = [
        f"MOTOR {nome} — atribuição dos modelos de novidade",
        f"Regime atual      : {r['regime']}",
        f"Modelo usado      : {r['modelo']}  (autoencoder 16-8-4-8-16)",
        f"Score normalizado : {r['score_normalizado']}  (1,0 = ponto de operação)",
        f"Severidade        : {r['severidade']}",
        f"Alarma            : {'SIM' if r['alarma'] else 'não'}",
        f"Leituras na janela: {r.get('n_leituras')}",
        "",
    ]

    if a.get("atribuido"):
        linhas_saida += [
            f"FALHA PROVÁVEL: {a['titulo']}  (pontuação z = {a['pontuacao']})",
            f"Indicador: {a['precocidade']}",
            f"Por quê: {a['explicacao']}",
            "",
            "COMPONENTES CANDIDATOS (ordenados por probabilidade):",
            *[f"  {i+1}. {c}" for i, c in enumerate(a.get("componentes_candidatos", []))],
            "",
            f"Segmentos destacados no modelo 3D: {', '.join(r.get('segmentos', [])) or 'nenhum'}",
        ]
        if a.get("alternativas"):
            linhas_saida += ["", "HIPÓTESES ALTERNATIVAS:",
                             *[f"  - {x['falha']} (z = {x['pontuacao']})" for x in a["alternativas"]]]
    else:
        linhas_saida += [f"SEM ATRIBUIÇÃO DE COMPONENTE. Motivo: {a.get('motivo')}",
                         "",
                         "Este estado é deliberado: num histórico 100% saudável, forçar uma "
                         "atribuição sempre elegeria alguma peça, e ela estaria errada."]

    if a.get("top_features"):
        linhas_saida += ["", "FEATURES COM MAIOR DESVIO (z-score contra o baseline do regime):",
                         *[f"  {f['feature']:18} z = {f['z']:>8}   grupo: {f['grupo']}"
                           for f in a["top_features"]]]

    if r.get("limitacoes"):
        linhas_saida += ["", "RESSALVAS:", *[f"  - {x}" for x in r["limitacoes"]]]

    return "\n".join(linhas_saida)


@tool
def get_ml_metricas() -> str:
    """Métricas dos modelos de novidade, medidas contra a bancada de injeção de
    falhas (6 modos x 3 severidades).

    Use quando o usuário perguntar quão bom é o modelo, qual a acurácia, ou se
    pode confiar no diagnóstico. SEMPRE cite a ressalva de que as falhas são
    injetadas, não observadas.
    """
    try:
        from ml_module.forzy import infer
        m = infer.metricas_treino()
    except Exception as exc:
        return f"Erro ao carregar as métricas dos modelos: {exc}"
    if not m:
        return ("Modelos de novidade ainda não treinados. "
                "Rode: python -m ml_module.forzy.train")

    linhas = [
        "MODELOS DE NOVIDADE FORZY — métricas da bancada de injeção de falhas",
        "",
        "Arquitetura: 6 modelos (2 motores x 3 regimes), treinados APENAS em dado saudável.",
        "O histórico real não tem um único rótulo de falha, então a validação é por injeção",
        "de degradações fisicamente calibradas na série de teste saudável.",
        "",
        f"{'detector':<20} {'PR-AUC':>9} {'ROC-AUC':>9} {'Recall@FP1%':>12} {'FP controle':>12}",
        "-" * 66,
    ]
    for nome, v in (m.get("por_detector") or {}).items():
        linhas.append(
            f"{nome:<20} {str(v.get('pr_auc_macro')):>9} {str(v.get('roc_auc_macro')):>9} "
            f"{str(v.get('recall_macro_fp1')):>12} {str(v.get('fp_controle')):>12}"
        )

    cel = m.get("celulas") or []
    if cel:
        linhas += ["", "PR-AUC por modo de falha (autoencoder):"]
        modos = {}
        for c in cel:
            modos.setdefault(c["modo"], []).append(c["pr_auc"])
        rot = {"F1": "Desbalanceamento", "F2": "Sobreaquecimento", "F3": "Rolamento",
               "F4": "Sensor travado", "F5": "Deriva de calibração", "F6": "Parada não programada"}
        for k in sorted(modos):
            media = sum(modos[k]) / len(modos[k])
            linhas.append(f"  {k} {rot.get(k, k):<24} {media:.4f}")

    linhas += [
        "",
        "RESSALVA OBRIGATÓRIA: as falhas são INJETADAS, não observadas. Isso prova que o",
        "detector separa desvios com a assinatura física esperada e quantifica a margem",
        "entre normal e anômalo. NÃO prova que detectaria uma falha real, que tem",
        "componentes não modelados (espectro, modulação, interação com a carga).",
        "As métricas são um LIMITE SUPERIOR OTIMISTA.",
        "",
        "Base de treino: cerca de 19 minutos de regime estacionário por motor, num único",
        "dia, sem variação de carga. A faixa de condições normais está subamostrada.",
    ]
    return "\n".join(linhas)
