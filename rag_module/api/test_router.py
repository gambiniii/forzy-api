"""Teste standalone do router — agent e RAG sem subir FastAPI."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rag_module.chains.agent import build_agent, chat
from rag_module.chains.rag_chain import ask_with_sources
from rag_module.chains.vectorstore import build_vectorstore


def test_agent_mode():
    print("=== TESTE MODE: AGENT ===")
    agent = build_agent()

    perguntas = [
        "Como está o motor agora? Tem alguma anomalia?",
        "Baseado no RUL atual, quando devo fazer manutenção?",
        "O que significa severidade low no LSTM e devo me preocupar?",
    ]

    history = []
    for p in perguntas:
        print(f"\nUSER: {p}")
        result = chat(p, agent=agent, history=history)
        print(f"AGENT: {result['answer'][:400]}...")
        print(f"Tools: {result['tools_used']}")
        history = result["messages"]


def test_rag_mode():
    print("\n=== TESTE MODE: RAG ===")
    vs = build_vectorstore()

    result = ask_with_sources(
        "Quais são os procedimentos de instalação do sensor de vibração?",
        vectorstore=vs,
    )
    print(f"RESPOSTA: {result['answer'][:400]}...")
    print(f"FONTES: {[s['file'] for s in result['sources']]}")


if __name__ == "__main__":
    test_agent_mode()
    test_rag_mode()
