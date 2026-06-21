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
        result = await loop.run_in_executor(None, predict_single, features_df, manager.ml_models, raw_df)

        combined = result["combined"]
        rul = result["rul"]
        lstm = result["lstm"]
        IF = result["isolation_forest"]

        estado = result.get("estado_operacional", "operando")

        if estado == "desligado":
            log.info("componente=%d — motor DESLIGADO (gate operacional)", componente_id)
            return {
                "type": "prediction",
                "componente_id": componente_id,
                "estado_operacional": "desligado",
                "overall_status": "motor_desligado",
                "health_score": None,
                "recommendation": combined["recommendation"],
                "rul_hours": None,
                "maintenance_window_days": None,
                "risk_level": "unknown",
                "lstm_severity": "n/a",
                "is_anomaly": False,
            }

        rul_hours = rul.get("rul_hours")
        maintenance_days = rul.get("maintenance_window_days")
        risk_level = rul.get("risk_level", "unknown")

        log.info(
            "componente=%d — ML: status=%s  health=%.3f  rul=%s  risk=%s  "
            "lstm_err=%.6f  lstm_sev=%s  lstm_anom=%s  if_score=%.6f  if_anom=%s  "
            "vib_min=%.4f  vib_max=%.4f",
            componente_id,
            combined["overall_status"],
            combined["health_score"],
            f"{rul_hours:.1f}h" if rul_hours is not None else "n/a",
            risk_level,
            lstm["reconstruction_error"],
            lstm["severity"],
            lstm["is_anomaly"],
            IF["anomaly_score"],
            IF["is_anomaly"],
            raw_df["1.2. Aceleração"].min(),
            raw_df["1.2. Aceleração"].max(),
        )

        is_anomaly = lstm["is_anomaly"] or IF["is_anomaly"]

        from src.db.postgres import SessionLocal
        from src.services import diagnostico_service
        db = SessionLocal()
        try:
            diagnostico_service.save_diagnostico(
                db,
                componente_id=componente_id,
                overall_status=combined["overall_status"],
                is_anomaly=is_anomaly,
                lstm_severity=lstm["severity"],
                risk_level=risk_level,
                rul_hours=round(rul_hours, 1) if rul_hours is not None else None,
                maintenance_window_days=round(maintenance_days, 1) if maintenance_days is not None else None,
                health_score=round(combined["health_score"], 4),
                health_index=result.get("health_index"),
                recommendation=combined["recommendation"],
            )
        except Exception as e:
            log.error("componente=%d — erro ao salvar diagnostico: %s", componente_id, e)
            db.rollback()
        finally:
            db.close()

        return {
            "type": "prediction",
            "componente_id": componente_id,
            "estado_operacional": "operando",
            "overall_status": combined["overall_status"],
            "health_score": round(combined["health_score"], 4),
            "health_index": result.get("health_index"),
            "recommendation": combined["recommendation"],
            "rul_hours": round(rul_hours, 1) if rul_hours is not None else None,
            "maintenance_window_days": round(maintenance_days, 1) if maintenance_days is not None else None,
            "risk_level": risk_level,
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
    try:
        current_online = manager.sensors.get(componente_id, False)
        await websocket.send_text(json.dumps({"type": "status", "online": current_online}))
    except Exception:
        pass

    is_sensor = False
    pending: list[dict] = []  # buffer de persistência — bulk insert a cada 10
    BULK_SIZE = 10

    try:
        while True:
            raw = await websocket.receive_text()

            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                log.warning("componente=%d — payload inválido (não é JSON)", componente_id)
                continue

            if data.get("type") != "sensor":
                continue

            if not is_sensor:
                is_sensor = True
                manager.sensors[componente_id] = True
                try:
                    await manager.broadcast(componente_id, {"type": "status", "online": True})
                except Exception:
                    pass
                log.info("componente=%d — sensor ONLINE", componente_id)

            payload = {k: v for k, v in data.items() if k != "type"}
            payload["componente_id"] = componente_id

            # Acumula no buffer ML imediatamente (sem esperar o banco)
            window_full = manager.push_leitura(componente_id, payload)

            # Broadcast simples para o frontend
            try:
                await manager.broadcast(componente_id, {"type": "leitura", **payload})
            except Exception:
                pass

            # Acumula para bulk insert
            pending.append(payload)
            if len(pending) >= BULK_SIZE:
                batch = pending[:]
                pending.clear()
                db = SessionLocal()
                try:
                    leitura_service.bulk_insert_leituras(db, batch)
                except Exception as e:
                    log.error("componente=%d — erro no bulk insert: %s", componente_id, e)
                    db.rollback()
                finally:
                    db.close()

            # Roda ML quando buffer atingir 500
            if window_full and manager.ml_ready:
                log.info("componente=%d — buffer cheio, iniciando ML", componente_id)
                try:
                    prediction = await _run_prediction(componente_id)
                    manager.reset_window(componente_id)
                    if prediction:
                        await manager.broadcast(componente_id, prediction)
                except Exception as e:
                    log.error("componente=%d — erro no ML (conexão mantida): %s", componente_id, e)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        log.error("componente=%d — erro inesperado no WS: %s", componente_id, e)
    finally:
        # Persiste o que sobrou no buffer de persistência
        if pending:
            db = SessionLocal()
            try:
                leitura_service.bulk_insert_leituras(db, pending)
            except Exception as e:
                log.error("componente=%d — erro ao persistir leituras pendentes: %s", componente_id, e)
                db.rollback()
            finally:
                db.close()
        manager.disconnect(componente_id, websocket)
        if is_sensor:
            manager.sensors[componente_id] = False
            try:
                await manager.broadcast(componente_id, {"type": "status", "online": False})
            except Exception:
                pass
            log.info("componente=%d — sensor OFFLINE", componente_id)
