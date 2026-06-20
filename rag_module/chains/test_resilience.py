"""Teste de retry e fallback do LLM."""

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.WARNING)


def safe_print(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", errors="replace").decode("ascii"))


import rag_module.chains.rag_chain as rag_chain
from rag_module.chains.rag_chain import ask
from rag_module.chains.vectorstore import build_vectorstore

ORIGINAL_URL = rag_chain.OPENROUTER_BASE_URL
INVALID_URL = "https://invalid-openrouter-url.example/api/v1"
QUESTION = "Como está o motor?"

print("=" * 60)
print("TESTE 1 — URL inválida (esperado: FALLBACK LOCAL)")
print("=" * 60)

rag_chain.OPENROUTER_BASE_URL = INVALID_URL
vectorstore = build_vectorstore()

try:
    result1 = ask(QUESTION, vectorstore=vectorstore)
    is_fallback = "indisponível" in result1 or "⚠️" in result1
    path1 = "FALLBACK LOCAL" if is_fallback else "LLM NORMAL"
    print(f"Caminho tomado: {path1}")
    safe_print(f"Resposta ({len(result1)} chars):\n{result1[:500]}")
except Exception as exc:
    print(f"ERRO inesperado: {exc}")

rag_chain.OPENROUTER_BASE_URL = ORIGINAL_URL

print("\n" + "=" * 60)
print("TESTE 2 — URL restaurada (esperado: LLM NORMAL)")
print("=" * 60)

try:
    result2 = ask(QUESTION, vectorstore=vectorstore)
    is_fallback = "indisponível" in result2 or "⚠️" in result2
    path2 = "FALLBACK LOCAL" if is_fallback else "LLM NORMAL"
    print(f"Caminho tomado: {path2}")
    safe_print(f"Resposta ({len(result2)} chars):\n{result2[:500]}")
except Exception as exc:
    print(f"ERRO: {exc}")
