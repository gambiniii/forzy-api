import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError

from src.db.postgres import create_tables
from src.routers import auth, alerts, maintenance, ml, aneel
from src.routers import maquinas, componentes, atributos, leituras, plantas, diagnosticos
from rag_module.api.router import router as chat_router

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s  %(name)s — %(message)s",
)
logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
logging.getLogger("uvicorn.error").setLevel(logging.ERROR)  # mostra erros mas suprime WS disconnect noise

app = FastAPI(
    title="Forzy Digital Twin API",
    description="API para monitoramento de motores industriais — Projeto Promon/FIAP",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router,         prefix="/auth",        tags=["Auth"])
app.include_router(plantas.router,      prefix="/plantas",     tags=["Plantas"])
app.include_router(maquinas.router,     prefix="/maquinas",    tags=["Máquinas"])
app.include_router(componentes.router,  prefix="/componentes", tags=["Componentes"])
app.include_router(atributos.router,    prefix="/atributos",   tags=["Atributos EAV"])
app.include_router(leituras.router,     prefix="/leituras",    tags=["Leituras de Sensor"])
app.include_router(diagnosticos.router, prefix="/diagnosticos", tags=["Diagnósticos ML"])
app.include_router(alerts.router,       prefix="/alerts",      tags=["Alertas"])
app.include_router(maintenance.router,  prefix="/maintenance", tags=["Manutenção"])
app.include_router(ml.router,           prefix="/ml",          tags=["ML / Predição"])
app.include_router(aneel.router,        prefix="/aneel",       tags=["ANEEL / Tensão"])
app.include_router(chat_router,                                tags=["Chat / RAG"])


@app.exception_handler(SQLAlchemyError)
async def sqlalchemy_error_handler(request: Request, exc: SQLAlchemyError):
    cause = str(exc.__cause__ or exc).splitlines()[0]
    logging.getLogger("app").error("DB error on %s %s — %s", request.method, request.url.path, cause)
    return JSONResponse(status_code=500, content={"detail": f"Erro de banco de dados: {cause}"})


@app.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception):
    logging.getLogger("app").error("Unexpected error on %s %s — %s", request.method, request.url.path, exc)
    return JSONResponse(status_code=500, content={"detail": "Erro interno do servidor."})


@app.on_event("startup")
async def startup():
    create_tables()
    _load_ml_models()


def _load_ml_models():
    import sys
    from pathlib import Path
    log = logging.getLogger("app")
    try:
        ml_root = Path(__file__).resolve().parent
        if str(ml_root) not in sys.path:
            sys.path.insert(0, str(ml_root))
        from ml_module.inference.predict import load_all_models
        from src.ws.manager import manager
        models = load_all_models()
        manager.set_ml_models(models)
        if_nf   = models["isolation_forest"][1].n_features_in_
        lstm_nf = models["lstm"][1].n_features_in_
        lstm_th = models["lstm"][2]
        log.info("ML carregado — IF features=%d  LSTM features=%d  LSTM threshold=%.6f", if_nf, lstm_nf, lstm_th)
    except Exception as e:
        log.warning("ML não carregado: %s", e)


@app.get("/")
def root():
    return {"status": "Forzy API online", "version": "0.2.0"}
