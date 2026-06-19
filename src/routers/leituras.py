import json
import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from src.ws.manager import manager

log = logging.getLogger("ws.leituras")

router = APIRouter()


class LeituraCreate(BaseModel):
    componente_id: int
    timestamp: Optional[datetime] = None
    temperatura: Optional[float] = None
    umidade: Optional[float] = None
    corrente: Optional[float] = None
    voltagem: Optional[float] = None
    rpm: Optional[float] = None
    vibracao: Optional[float] = None
    inclinacao: Optional[float] = None


def _register(r: APIRouter):
    from src.controllers.leitura_controller import (
        ctrl_ingest_leitura, ctrl_get_leituras, ctrl_get_ultima_leitura,
    )
    r.post("/",                                      response_model=None, status_code=201)(ctrl_ingest_leitura)
    r.get("/componente/{componente_id}",             response_model=None)(ctrl_get_leituras)
    r.get("/componente/{componente_id}/ultima",      response_model=None)(ctrl_get_ultima_leitura)


_register(router)


async def _run_prediction(componente_id: int) -> dict | None:
    """Roda inferência ML na janela acumulada e retorna payload para broadcast."""
    import asyncio
    try:
        from ml_module.features.feature_engineering import build_features
        from ml_module.inference.predict import predict_single
        from ml_module.inference.forzy_adapter import leituras_to_raw_df

        rows = manager.get_window(componente_id)
        raw_df = leituras_to_raw_df(rows)
        if raw_df.empty:
            return None

        loop = asyncio.get_event_loop()
        features_df = await loop.run_in_executor(None, build_features, raw_df)
        result = await loop.run_in_executor(None, predict_single, features_df, manager.ml_models)

        combined = result["combined"]
        rul = result["rul"]
        lstm = result["lstm"]

        log.info(
            "componente=%d — ML: status=%s  rul=%.1fh  severity=%s",
            componente_id,
            combined["overall_status"],
            rul["rul_hours"],
            lstm["severity"],
        )

        is_anomaly = lstm["is_anomaly"] or result["isolation_forest"]["is_anomaly"]

        if is_anomaly:
            from src.db.postgres import SessionLocal
            from src.services import anomalia_service
            db = SessionLocal()
            try:
                anomalia_service.save_anomalia(
                    db,
                    componente_id=componente_id,
                    overall_status=combined["overall_status"],
                    lstm_severity=lstm["severity"],
                    risk_level=rul["risk_level"],
                    rul_hours=round(rul["rul_hours"], 1),
                    maintenance_window_days=round(rul["maintenance_window_days"], 1),
                    health_score=round(combined["health_score"], 4),
                    recommendation=combined["recommendation"],
                )
            except Exception as e:
                log.error("componente=%d — erro ao salvar anomalia: %s", componente_id, e)
                db.rollback()
            finally:
                db.close()

        return {
            "type": "prediction",
            "componente_id": componente_id,
            "overall_status": combined["overall_status"],
            "health_score": round(combined["health_score"], 4),
            "recommendation": combined["recommendation"],
            "rul_hours": round(rul["rul_hours"], 1),
            "maintenance_window_days": round(rul["maintenance_window_days"], 1),
            "risk_level": rul["risk_level"],
            "lstm_severity": lstm["severity"],
            "is_anomaly": is_anomaly,
        }
    except Exception as e:
        log.error("componente=%d — erro na inferência ML: %s", componente_id, e)
        return None


@router.websocket("/ws/{componente_id}")
async def ws_leituras(componente_id: int, websocket: WebSocket):
    from src.db.postgres import SessionLocal
    from src.services import leitura_service

    await manager.connect(componente_id, websocket)
    # Envia o status atual do sensor para o novo cliente imediatamente
    current_online = manager.sensors.get(componente_id, False)
    await websocket.send_text(json.dumps({"type": "status", "online": current_online}))

    is_sensor = False
    try:
        while True:
            raw = await websocket.receive_text()

            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                log.warning("componente=%d — payload inválido (não é JSON)", componente_id)
                continue

            if data.get("type") == "sensor":
                if not is_sensor:
                    is_sensor = True
                    manager.sensors[componente_id] = True
                    await manager.broadcast(componente_id, {"type": "status", "online": True})
                    log.info("componente=%d — sensor ONLINE", componente_id)

                payload = {k: v for k, v in data.items() if k != "type"}
                payload["componente_id"] = componente_id

                db = SessionLocal()
                try:
                    leitura = leitura_service.ingest_leitura(db, **payload)
                    leitura_dict = {
                        "type": "leitura",
                        "id": leitura.id,
                        "componente_id": leitura.componente_id,
                        "timestamp": leitura.timestamp.isoformat(),
                        "temperatura": leitura.temperatura,
                        "umidade": leitura.umidade,
                        "corrente": leitura.corrente,
                        "voltagem": leitura.voltagem,
                        "rpm": leitura.rpm,
                        "vibracao": leitura.vibracao,
                        "inclinacao": leitura.inclinacao,
                    }
                except Exception as e:
                    log.error("componente=%d — erro ao salvar leitura: %s", componente_id, e)
                    db.rollback()
                    continue
                finally:
                    db.close()

                await manager.broadcast(componente_id, leitura_dict)

                # Acumula leitura na janela e roda ML quando cheia
                window_full = manager.push_leitura(componente_id, leitura_dict)
                if window_full and manager.ml_ready:
                    prediction = await _run_prediction(componente_id)
                    if prediction:
                        await manager.broadcast(componente_id, prediction)

    except WebSocketDisconnect:
        manager.disconnect(componente_id, websocket)
        if is_sensor:
            manager.sensors[componente_id] = False
            await manager.broadcast(componente_id, {"type": "status", "online": False})
            log.info("componente=%d — sensor OFFLINE", componente_id)
    except Exception as e:
        log.error("componente=%d — erro inesperado no WS: %s", componente_id, e)
        manager.disconnect(componente_id, websocket)
        if is_sensor:
            manager.sensors[componente_id] = False
