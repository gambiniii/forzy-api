"""
Persistência de histórico de conversa entre sessões RAG.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

MEMORY_DIR = PROJECT_ROOT / "rag_module" / "memory"
MEMORY_DIR.mkdir(exist_ok=True)
MAX_HISTORY_TURNS = 10


def _sanitize_session_id(session_id: str) -> str:
    """Sanitiza session_id — apenas alfanumérico, hífen e underscore."""
    cleaned = "".join(c for c in session_id if c.isalnum() or c in "-_")
    return cleaned or "default"


def _session_path(session_id: str) -> Path:
    """Retorna caminho do arquivo JSON da sessão."""
    safe_id = _sanitize_session_id(session_id)
    return MEMORY_DIR / f"{safe_id}.json"


def load_history(session_id: str) -> list:
    """Carrega histórico da sessão do disco."""
    path = _session_path(session_id)
    if not path.exists():
        return []

    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("messages", [])
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Falha ao carregar histórico de %s: %s", session_id, exc)
        return []


def save_history(session_id: str, messages: list) -> None:
    """Persiste histórico convertido das mensagens LangChain."""
    safe_id = _sanitize_session_id(session_id)
    history = []

    for msg in messages:
        if hasattr(msg, "content") and msg.content:
            role = "user" if "Human" in type(msg).__name__ else "assistant"
            history.append({"role": role, "content": str(msg.content)})

    max_items = MAX_HISTORY_TURNS * 2
    history = history[-max_items:]

    payload = {
        "session_id": safe_id,
        "updated_at": datetime.now().isoformat(),
        "turn_count": len(history) // 2,
        "messages": history,
    }

    path = _session_path(safe_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def clear_history(session_id: str) -> None:
    """Remove arquivo de histórico da sessão."""
    path = _session_path(session_id)
    if path.exists():
        path.unlink()


def list_sessions() -> list:
    """Lista todas as sessões persistidas, mais recentes primeiro."""
    sessions = []

    for filepath in MEMORY_DIR.glob("*.json"):
        try:
            with open(filepath, encoding="utf-8") as f:
                data = json.load(f)
            sessions.append(
                {
                    "session_id": filepath.stem,
                    "updated_at": data.get("updated_at"),
                    "turn_count": data.get("turn_count", 0),
                }
            )
        except (json.JSONDecodeError, OSError):
            sessions.append(
                {
                    "session_id": filepath.stem,
                    "updated_at": None,
                    "turn_count": 0,
                }
            )

    sessions.sort(key=lambda s: s.get("updated_at") or "", reverse=True)
    return sessions


def get_session_summary(session_id: str) -> dict:
    """Retorna resumo de uma sessão."""
    path = _session_path(session_id)

    if not path.exists():
        return {
            "session_id": _sanitize_session_id(session_id),
            "exists": False,
            "turn_count": 0,
            "updated_at": None,
            "last_message": None,
        }

    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {
            "session_id": _sanitize_session_id(session_id),
            "exists": False,
            "turn_count": 0,
            "updated_at": None,
            "last_message": None,
        }

    messages = data.get("messages", [])
    last_user = None
    for msg in reversed(messages):
        if msg.get("role") == "user":
            last_user = msg.get("content")
            break

    return {
        "session_id": data.get("session_id", _sanitize_session_id(session_id)),
        "exists": True,
        "turn_count": data.get("turn_count", len(messages) // 2),
        "updated_at": data.get("updated_at"),
        "last_message": last_user,
    }
