"""
Geração periódica de diagnóstico ML a partir do que já está no Postgres.

Desacoplado do WebSocket de dispositivo (src/routers/leituras.py) — roda
independente de qualquer cliente conectado, usando as leituras que o
forzy_poller já grava em leitura_sensor. Roda uma vez por motor cadastrado
em COMPONENTE_IDS (um motor não afeta o diagnóstico do outro).

Também implementa, por ciclo:
  - Circuit Breaker (CS3 §8): sanity check de faixa física, gate de dado
    insuficiente/offline — quando dispara, o diagnóstico é salvo como
    "retido" em vez de classificado, e nenhum alerta é gerado.
  - Metric Contract (CS3 §7): classificação NOMINAL/ATENÇÃO/CRÍTICO por
    threshold simples de vibração/temperatura (independente do ML),
    gerando um alerta em `alerts` na transição de estado.
"""

import asyncio
import logging
from datetime import datetime, timezone

from src.services.forzy_poller import COMPONENTE_IDS

logger = logging.getLogger("app.diagnostico_scheduler")

DIAGNOSTIC_INTERVAL_SECONDS = 300  # 5 min
MIN_ROWS = 5                       # abaixo disso (após filtro de sanidade), pula o ciclo
MAX_ROWS = 500                     # janela máxima puxada do banco para feature engineering
OFFLINE_THRESHOLD_SECONDS = 30     # mesmo critério do indicador Online/Offline do front
CONFIDENCE_THRESHOLD = 60.0        # abaixo disso, diagnóstico ML é rebaixado para "retido"

# Faixa de sanidade física (Circuit Breaker — CS3 §7.5/§8.1).
# "rpm" é o nome da coluna, mas guarda a Velocidade/vibração (mm/s) usada
# nas zonas ISO 10816 — ver nota de nomenclatura no restante do projeto.
VIBRATION_RANGE = (0.0, 50.0)
TEMPERATURE_RANGE = (-10.0, 150.0)

# Estado de threshold por componente, mantido em memória — usado só para
# saber se houve TRANSIÇÃO de estado (evita recriar o mesmo alerta a cada
# ciclo enquanto o motor permanece em atenção/crítico).
_LAST_THRESHOLD_STATE: dict[int, str] = {}

_SEVERITY_RANK = {"nominal": 0, "atencao": 1, "critico": 2}


def _is_valid_reading(r) -> bool:
    if r.rpm is None or r.temperatura is None:
        return False
    return (
        VIBRATION_RANGE[0] <= r.rpm <= VIBRATION_RANGE[1]
        and TEMPERATURE_RANGE[0] <= r.temperatura <= TEMPERATURE_RANGE[1]
    )


def _classify_threshold(latest, limites) -> tuple[str, str, list[str]]:
    """Classifica a leitura mais recente contra os limites do Metric Contract.
    Independe do resultado do ML — é uma regra determinística e auditável.

    NOTA: `latest.rpm` guarda a Velocidade ISO 10816 (mm/s) — não confundir com
    `latest.vibracao` (aceleração em g do acelerômetro, usada só nas features
    de ML). `vib_critico`/`vib_atencao` são calibrados para essa velocidade,
    não para aceleração — não trocar por `latest.vibracao` aqui.
    """
    vib, temp, acel = latest.rpm, latest.temperatura, latest.vibracao
    status = "nominal"
    msgs: list[str] = []
    breached: list[str] = []

    def bump(novo_status: str, msg: str, metrica: str) -> None:
        nonlocal status
        if _SEVERITY_RANK[novo_status] > _SEVERITY_RANK[status]:
            status = novo_status
        msgs.append(msg)
        if metrica not in breached:
            breached.append(metrica)

    if vib is not None:
        if vib >= limites.vib_critico:
            bump("critico", f"Vibração em {vib:.2f} mm/s, acima do limite crítico de {limites.vib_critico:.2f} mm/s", "velocidade")
        elif vib >= limites.vib_atencao:
            bump("atencao", f"Vibração em {vib:.2f} mm/s, acima do limite de atenção de {limites.vib_atencao:.2f} mm/s", "velocidade")

    if temp is not None:
        if temp >= limites.temp_critico:
            bump("critico", f"Temperatura em {temp:.1f}°C, acima do limite crítico de {limites.temp_critico:.1f}°C", "temperatura")
        elif temp >= limites.temp_atencao:
            bump("atencao", f"Temperatura em {temp:.1f}°C, acima do limite de atenção de {limites.temp_atencao:.1f}°C", "temperatura")

    # Aceleração: é o canal que melhor denuncia rolamento, e sem ele o modelo 3D
    # nunca conseguia destacar os mancais. `getattr` porque bancos que ainda não
    # rodaram a migration 005 não têm as colunas.
    acel_at = getattr(limites, "acel_atencao", None)
    acel_cr = getattr(limites, "acel_critico", None)
    if acel is not None and acel_at is not None and acel_cr is not None:
        if acel >= acel_cr:
            bump("critico", f"Aceleração em {acel:.3f} g, acima do limite crítico de {acel_cr:.3f} g", "aceleracao")
        elif acel >= acel_at:
            bump("atencao", f"Aceleração em {acel:.3f} g, acima do limite de atenção de {acel_at:.3f} g", "aceleracao")

    message = " ".join(msgs) if msgs else "Todos os parâmetros dentro dos limites contratuais."
    return status, message, breached


def _get_limites(componente_id: int):
    from src.db.postgres import SessionLocal
    from src.services import limite_service

    db = SessionLocal()
    try:
        return limite_service.get_or_create_limites(db, componente_id)
    finally:
        db.close()


def _handle_threshold_alert(componente_id: int, status: str, message: str) -> None:
    """Cria um alerta em `alerts` só na TRANSIÇÃO para atenção/crítico."""
    last = _LAST_THRESHOLD_STATE.get(componente_id, "nominal")
    _LAST_THRESHOLD_STATE[componente_id] = status
    if status == last or status not in ("atencao", "critico"):
        return

    from src.db.postgres import SessionLocal
    import sqlalchemy as sa

    db = SessionLocal()
    try:
        maquina_id = db.execute(
            sa.text("SELECT maquina_id FROM componente WHERE id = :c"), {"c": componente_id}
        ).scalar()
        if maquina_id is None:
            return
        severity = "critical" if status == "critico" else "medium"
        db.execute(sa.text("""
            INSERT INTO alerts (motor_id, severity, message, anomaly_score, rul_estimated)
            VALUES (:mid, :sev, :msg, NULL, NULL)
        """), {"mid": maquina_id, "sev": severity, "msg": message})
        db.commit()
        logger.info("componente=%d — alerta de limite criado (%s): %s", componente_id, status, message)
    except Exception as e:
        logger.error("componente=%d — erro ao criar alerta de limite: %s", componente_id, e)
        db.rollback()
    finally:
        db.close()


def _save_retido(componente_id: int, motivo: str) -> None:
    from src.db.postgres import SessionLocal
    from src.services import diagnostico_service

    db = SessionLocal()
    try:
        diagnostico_service.save_diagnostico(
            db,
            componente_id=componente_id,
            overall_status="retido",
            is_anomaly=False,
            recommendation=f"Diagnóstico retido — {motivo}",
            threshold_status="retido",
            threshold_message=motivo,
        )
        logger.info("componente=%d — diagnóstico retido: %s", componente_id, motivo)
    except Exception as e:
        logger.error("componente=%d — erro ao salvar diagnostico retido: %s", componente_id, e)
        db.rollback()
    finally:
        db.close()


async def _run_once(componente_id: int) -> None:
    from src.db.postgres import SessionLocal
    from src.services import leitura_service, diagnostico_service
    from src.ws.manager import manager

    if not manager.ml_ready:
        logger.debug("Modelos ML ainda não carregados — pulando ciclo de diagnóstico.")
        return

    db = SessionLocal()
    try:
        rows = leitura_service.get_leituras(db, componente_id, None, None, MAX_ROWS)
    finally:
        db.close()

    # --- Circuit Breaker: disponibilidade do dado ---
    if not rows:
        _save_retido(componente_id, "Sem leituras disponíveis para este componente.")
        return

    latest = rows[0]  # get_leituras retorna DESC — rows[0] é a mais recente
    now = datetime.now(timezone.utc)
    if (now - latest.timestamp).total_seconds() > OFFLINE_THRESHOLD_SECONDS:
        _save_retido(
            componente_id,
            f"Sensor offline — última leitura há mais de {OFFLINE_THRESHOLD_SECONDS}s.",
        )
        return

    # --- Circuit Breaker: integridade do dado (faixa física de sanidade) ---
    valid_rows = [r for r in rows if _is_valid_reading(r)]
    descartadas = len(rows) - len(valid_rows)
    if descartadas > 0:
        logger.warning(
            "componente=%d — %d leitura(s) fora da faixa física válida, descartada(s).",
            componente_id, descartadas,
        )

    if len(valid_rows) < MIN_ROWS:
        _save_retido(
            componente_id,
            f"Dados insuficientes — {len(valid_rows)} leitura(s) válida(s) na janela (mínimo {MIN_ROWS}).",
        )
        return

    # --- Metric Contract: classificação por threshold (independe do ML) ---
    limites = _get_limites(componente_id)
    threshold_status, threshold_message, breached_metrics = _classify_threshold(latest, limites)
    _handle_threshold_alert(componente_id, threshold_status, threshold_message)

    # --- Pipeline de ML (IF + LSTM + RUL) ---
    # get_leituras devolve em ordem DECRESCENTE; a inferência (rolling features
    # em build_features + o gate operacional, que lê .iloc[-1]) precisa de série
    # temporal CRESCENTE, senão as janelas móveis andam para trás e o gate acaba
    # decidindo operando/desligado pela leitura mais ANTIGA da janela, não a mais
    # nova — mesmo ajuste já usado em diagnosticos.py (rota /atribuicao).
    valid_rows_asc = sorted(valid_rows, key=lambda r: r.timestamp)
    rows_dict = [
        {"timestamp": r.timestamp, "rpm": r.rpm, "vibracao": r.vibracao, "temperatura": r.temperatura}
        for r in valid_rows_asc
    ]

    from ml_module.inference.forzy_adapter import leituras_to_raw_df
    from ml_module.features.feature_engineering import build_features
    from ml_module.inference.predict import predict_single

    raw_df = leituras_to_raw_df(rows_dict)
    if raw_df.empty:
        _save_retido(componente_id, "Falha ao montar features a partir das leituras.")
        return

    loop = asyncio.get_event_loop()
    features_df = await loop.run_in_executor(None, build_features, raw_df)
    result = await loop.run_in_executor(None, predict_single, features_df, manager.ml_models, raw_df)

    estado = result.get("estado_operacional", "operando")
    if estado == "desligado":
        logger.info("componente=%d — motor DESLIGADO (gate operacional), diagnóstico não salvo.", componente_id)
        return

    combined = result["combined"]
    rul = result["rul"]
    lstm = result["lstm"]
    IF = result["isolation_forest"]
    confidence = result.get("confidence")

    rul_hours = rul.get("rul_hours")
    maintenance_days = rul.get("maintenance_window_days")
    is_anomaly = lstm["is_anomaly"] or IF["is_anomaly"]

    overall_status = combined["overall_status"]
    recommendation = combined["recommendation"]

    # --- Circuit Breaker: confiança do modelo ---
    if confidence is not None and confidence < CONFIDENCE_THRESHOLD:
        overall_status = "retido"
        recommendation = (
            f"Diagnóstico retido — confiança do modelo em {confidence:.1f}% "
            f"(mínimo {CONFIDENCE_THRESHOLD:.0f}%). Revisão humana recomendada."
        )
        logger.info("componente=%d — confiança baixa (%.1f%%), diagnóstico rebaixado para retido.",
                    componente_id, confidence)

    db = SessionLocal()
    try:
        diagnostico_service.save_diagnostico(
            db,
            componente_id=componente_id,
            overall_status=overall_status,
            is_anomaly=is_anomaly,
            lstm_severity=lstm["severity"],
            risk_level=rul.get("risk_level", "unknown"),
            rul_hours=round(rul_hours, 1) if rul_hours is not None else None,
            maintenance_window_days=round(maintenance_days, 1) if maintenance_days is not None else None,
            health_score=round(combined["health_score"], 4),
            health_index=result.get("health_index"),
            recommendation=recommendation,
            confidence=confidence,
            threshold_status=threshold_status,
            threshold_message=threshold_message,
            breached_metrics=",".join(breached_metrics) if breached_metrics else None,
        )
        logger.info(
            "componente=%d — diagnóstico salvo: status=%s health=%.3f anomaly=%s confidence=%s limite=%s",
            componente_id, overall_status, combined["health_score"], is_anomaly, confidence, threshold_status,
        )
    except Exception as e:
        logger.error("componente=%d — erro ao salvar diagnostico: %s", componente_id, e)
        db.rollback()
    finally:
        db.close()


async def diagnostico_loop() -> None:
    logger.info("Diagnostico scheduler iniciado — intervalo: %ds | componentes: %s",
                DIAGNOSTIC_INTERVAL_SECONDS, COMPONENTE_IDS)
    while True:
        for componente_id in COMPONENTE_IDS:
            try:
                await _run_once(componente_id)
            except Exception as e:
                logger.error("componente=%d — erro no ciclo de diagnóstico: %s", componente_id, e)
        await asyncio.sleep(DIAGNOSTIC_INTERVAL_SECONDS)
