"""
Router FastAPI — chat com agent LangGraph e modo RAG.
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_module.chains.agent import build_agent, chat
from rag_module.chains.logger import LOG_FILE, get_logger, read_logs
from rag_module.chains.memory import (
    MEMORY_DIR,
    clear_history,
    get_session_summary,
    list_sessions,
)
from rag_module.chains.rag_chain import ask_with_sources
from rag_module.chains.vectorstore import CHROMA_PERSIST_DIR, build_vectorstore

router = APIRouter(prefix="/chat", tags=["chat"])

logger = get_logger("forzy.api")

_agent = None
_vectorstore = None


def get_agent():
    """Retorna agente LangGraph (lazy load)."""
    global _agent, _vectorstore
    if _agent is None:
        _vectorstore = build_vectorstore()
        _agent = build_agent(_vectorstore)
    return _agent


def get_vectorstore():
    """Retorna vectorstore Chroma (lazy load)."""
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = build_vectorstore()
    return _vectorstore


class ChatRequest(BaseModel):
    message: str
    machine_id: Optional[str] = "1"
    history: Optional[list] = []
    session_id: Optional[str] = "default"
    mode: Optional[str] = "agent"


class ChatResponse(BaseModel):
    answer: str
    tools_used: Optional[list] = []
    sources: Optional[list] = []
    mode: str
    machine_id: str
    session_id: Optional[str] = "default"
    fallback_used: bool = False
    response_time_ms: Optional[int] = None


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    agent_loaded: bool
    vectorstore_loaded: bool
    model: str
    vectorstore: dict
    ml_models: dict
    sessions: dict
    documents: dict
    openrouter: dict


@router.get("/health", response_model=HealthResponse)
async def health():
    """Health check enriquecido do módulo de chat."""
    from rag_module.config import DOCUMENTS_DIR, LLM_MODEL

    chroma_dir = CHROMA_PERSIST_DIR
    vs_info = {"status": "not_built", "chunks": 0, "persist_dir": str(chroma_dir)}
    if _vectorstore is not None:
        try:
            count = _vectorstore._collection.count()
            vs_info = {
                "status": "loaded",
                "chunks": count,
                "persist_dir": str(chroma_dir),
                "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
            }
        except Exception:
            vs_info["status"] = "error"
    elif chroma_dir.exists() and any(chroma_dir.iterdir()):
        vs_info["status"] = "persisted_on_disk"

    ml_info = {"status": "unknown"}
    try:
        project_root = CHROMA_PERSIST_DIR.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        if_dir = project_root / "ml_module/models/baseline/saved"
        lstm_dir = project_root / "ml_module/models/anomaly/saved"
        hybrid_dir = project_root / "ml_module/models/rul/saved_hybrid"

        ml_info = {
            "status": "available",
            "isolation_forest": {
                "saved": if_dir.exists(),
                "cwru_f1": 0.9757,
            },
            "lstm_autoencoder": {
                "saved": lstm_dir.exists(),
                "cwru_f1": 0.9846,
            },
            "xgboost_rul": {
                "hybrid_saved": hybrid_dir.exists(),
                "r2": 0.9994,
                "mae_hours": 0.12,
            },
        }
    except Exception as exc:
        ml_info = {"status": "error", "detail": str(exc)}

    sessions = list_sessions()
    sessions_info = {
        "total": len(sessions),
        "memory_dir": str(MEMORY_DIR),
        "recent": sessions[:3] if sessions else [],
    }

    docs_dir = DOCUMENTS_DIR
    docs_info = {"total_files": 0, "files": []}
    if docs_dir.exists():
        files = list(docs_dir.iterdir())
        docs_info = {
            "total_files": len(files),
            "files": [
                {
                    "name": f.name,
                    "size_kb": round(f.stat().st_size / 1024, 1),
                    "type": f.suffix,
                }
                for f in sorted(files)
            ],
        }

    openrouter_info = {
        "model": LLM_MODEL,
        "base_url": "https://openrouter.ai/api/v1",
        "timeout_seconds": 15,
        "max_retries": 3,
        "fallback": "rag_local",
    }

    return HealthResponse(
        status="ok",
        timestamp=datetime.now().isoformat(),
        agent_loaded=_agent is not None,
        vectorstore_loaded=_vectorstore is not None,
        model=LLM_MODEL,
        vectorstore=vs_info,
        ml_models=ml_info,
        sessions=sessions_info,
        documents=docs_info,
        openrouter=openrouter_info,
    )


@router.post("/message", response_model=ChatResponse)
async def send_message(request: ChatRequest):
    """
    Endpoint principal de chat.

    mode='agent': usa LangGraph agent com tools (sensores + ML + docs)
    mode='rag': busca apenas nos documentos técnicos
    """
    try:
        loop = asyncio.get_event_loop()

        logger.info(
            "Request recebido",
            extra={
                "session_id": request.session_id,
                "question_preview": request.message[:80],
                "mode": request.mode,
            },
        )

        if request.mode == "agent":
            agent = get_agent()
            result = await loop.run_in_executor(
                None,
                lambda: chat(
                    message=request.message,
                    agent=agent,
                    history=request.history or None,
                    session_id=request.session_id,
                ),
            )
            return ChatResponse(
                answer=result["answer"],
                tools_used=result.get("tools_used", []),
                sources=[],
                mode="agent",
                machine_id=request.machine_id,
                session_id=result.get("session_id", request.session_id),
                fallback_used=result.get("fallback_used", False),
                response_time_ms=result.get("response_time_ms"),
            )

        vs = get_vectorstore()
        result = await loop.run_in_executor(
            None,
            lambda: ask_with_sources(
                question=request.message,
                vectorstore=vs,
            ),
        )
        return ChatResponse(
            answer=result["answer"],
            tools_used=[],
            sources=result.get("sources", []),
            mode="rag",
            machine_id=request.machine_id,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/warmup")
async def warmup():
    """Pré-carrega agent e vectorstore em background."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, get_agent)
    return {"status": "warmed up", "agent_loaded": True}


@router.get("/sessions")
async def get_sessions():
    """Lista todas as sessões com histórico salvo."""
    return {"sessions": list_sessions()}


@router.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Resumo de uma sessão específica."""
    return get_session_summary(session_id)


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Limpa histórico de uma sessão."""
    clear_history(session_id)
    return {"status": "cleared", "session_id": session_id}


@router.get("/logs")
async def get_logs(n: int = 20):
    """
    Retorna as últimas n entradas do log do agente.
    Útil para monitoramento e demonstração.
    """
    entries = read_logs(n)
    return {
        "total_returned": len(entries),
        "log_file": str(LOG_FILE),
        "entries": entries,
    }
