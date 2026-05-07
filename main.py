from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.db.postgres import create_tables
from src.routers import auth, alerts, maintenance, ml, aneel
from src.routers import maquinas, componentes, atributos, leituras

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
app.include_router(maquinas.router,     prefix="/maquinas",    tags=["Máquinas"])
app.include_router(componentes.router,  prefix="/componentes", tags=["Componentes"])
app.include_router(atributos.router,    prefix="/atributos",   tags=["Atributos EAV"])
app.include_router(leituras.router,     prefix="/leituras",    tags=["Leituras de Sensor"])
app.include_router(alerts.router,       prefix="/alerts",      tags=["Alertas"])
app.include_router(maintenance.router,  prefix="/maintenance", tags=["Manutenção"])
app.include_router(ml.router,           prefix="/ml",          tags=["ML / Predição"])
app.include_router(aneel.router,        prefix="/aneel",       tags=["ANEEL / Tensão"])


@app.on_event("startup")
async def startup():
    create_tables()


@app.get("/")
def root():
    return {"status": "Forzy API online", "version": "0.2.0"}
