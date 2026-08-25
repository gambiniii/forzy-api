"""
Geração periódica de diagnóstico ML a partir do que já está no Postgres.

Desacoplado do WebSocket de dispositivo (src/routers/leituras.py) — roda
independente de qualquer cliente conectado, usando as leituras que o
forzy_poller já grava em leitura_sensor. Roda uma vez por motor cadastrado
em COMPONENTE_IDS (um motor não afeta o diagnóstico do outro).
"""

import asyncio
import logging

from src.services.forzy_poller import COMPONENTE_IDS

logger = logging.getLogger("app.diagnostico_scheduler")

DIAGNOSTIC_INTERVAL_SECONDS = 300  # 5 min
MIN_ROWS = 5                       # abaixo disso, pula o ciclo (dados insuficientes)
MAX_ROWS = 500                     # janela máxima puxada do banco para feature engineering


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

    if len(rows) < MIN_ROWS:
        logger.debug("componente=%d — apenas %d leitura(s) — mínimo %d, pulando ciclo.",
                      componente_id, len(rows), MIN_ROWS)
        return

    rows_dict = [
        {"timestamp": r.timestamp, "rpm": r.rpm, "vibracao": r.vibracao, "temperatura": r.temperatura}
        for r in rows
    ]

    from ml_module.inference.forzy_adapter import leituras_to_raw_df
    from ml_module.features.feature_engineering import build_features
    from ml_module.inference.predict import predict_single

    raw_df = leituras_to_raw_df(rows_dict)
    if raw_df.empty:
        logger.debug("componente=%d — raw_df vazio após conversão — pulando ciclo.", componente_id)
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

    rul_hours = rul.get("rul_hours")
    maintenance_days = rul.get("maintenance_window_days")
    is_anomaly = lstm["is_anomaly"] or IF["is_anomaly"]

    db = SessionLocal()
    try:
        diagnostico_service.save_diagnostico(
            db,
            componente_id=componente_id,
            overall_status=combined["overall_status"],
            is_anomaly=is_anomaly,
            lstm_severity=lstm["severity"],
            risk_level=rul.get("risk_level", "unknown"),
            rul_hours=round(rul_hours, 1) if rul_hours is not None else None,
            maintenance_window_days=round(maintenance_days, 1) if maintenance_days is not None else None,
            health_score=round(combined["health_score"], 4),
            health_index=result.get("health_index"),
            recommendation=combined["recommendation"],
        )
        logger.info(
            "componente=%d — diagnóstico salvo: status=%s health=%.3f anomaly=%s",
            componente_id, combined["overall_status"], combined["health_score"], is_anomaly,
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
