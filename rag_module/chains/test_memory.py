"""Teste de persistência de histórico entre sessões."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from rag_module.chains.agent import build_agent, chat
from rag_module.chains.memory import clear_history, get_session_summary

SESSION_ID = "tecnico_01"

clear_history(SESSION_ID)

print("=" * 60)
print("RODADA 1 — primeira sessão")
print("=" * 60)

agent = build_agent()
r1 = chat("Como está o motor?", agent=agent, session_id=SESSION_ID)
print(f"\nTurno 1: {r1['answer'][:200]}")

r2 = chat("E o RUL?", agent=agent, session_id=SESSION_ID)
print(f"\nTurno 2: {r2['answer'][:200]}")

summary = get_session_summary(SESSION_ID)
print(f"\nSessão salva: {summary}")

print("\n" + "=" * 60)
print("RODADA 2 — reabertura (agent novo, history=None)")
print("=" * 60)

agent2 = build_agent()
r3 = chat(
    "Baseado no que discutimos, devo me preocupar?",
    agent=agent2,
    history=None,
    session_id=SESSION_ID,
)
print(f"\nTurno 3 (nova sessão, histórico carregado): {r3['answer'][:300]}")
print(f"Tools usadas: {r3['tools_used']}")
