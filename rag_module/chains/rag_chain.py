"""
RAG Chain — LangChain LCEL + OpenRouter (Gemini) para assistente técnico Forzy.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI

from rag_module.chains.llm_resilience import resilient_llm_call
from rag_module.chains.vectorstore import build_vectorstore, get_retriever
from rag_module.config import (
    LLM_MODEL,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    RETRIEVER_K,
    SYSTEM_PROMPT,
)


def get_llm(temperature: float = 0.2) -> ChatOpenAI:
    """Retorna LLM configurado para OpenRouter."""
    return ChatOpenAI(
        model=LLM_MODEL,
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        temperature=temperature,
        max_tokens=2048,
        default_headers={
            "HTTP-Referer": "https://forzy-digital-twin.com",
            "X-Title": "Forzy Digital Twin",
        },
    )


def format_docs(docs) -> str:
    """Formata documentos recuperados para o prompt."""
    return "\n\n---\n\n".join(
        [
            f"[{doc.metadata.get('source_file', 'doc')}]\n{doc.page_content}"
            for doc in docs
        ]
    )


def build_rag_chain(vectorstore=None):
    """Constrói chain RAG completo com LCEL."""
    retriever = get_retriever(vectorstore, k=RETRIEVER_K)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                SYSTEM_PROMPT
                + """

Contexto dos documentos técnicos:
{context}
""",
            ),
            ("human", "{question}"),
        ]
    )

    rag_chain = (
        RunnableParallel(
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough(),
            }
        )
        | prompt
        | get_llm()
        | StrOutputParser()
    )

    return rag_chain


def ask(question: str, vectorstore=None) -> str:
    """Executa pergunta no RAG chain e retorna resposta."""
    chain = build_rag_chain(vectorstore)

    def call_chain():
        return chain.invoke(question)

    def fallback():
        retriever = get_retriever(vectorstore)
        docs = retriever.invoke(question)
        if not docs:
            return "Não encontrei informações relevantes nos documentos técnicos."
        chunks = [
            f"[{d.metadata.get('source_file', 'doc')}]\n{d.page_content}"
            for d in docs[:3]
        ]
        return (
            "⚠️ LLM temporariamente indisponível. "
            "Informações dos documentos técnicos:\n\n"
            + "\n\n---\n\n".join(chunks)
        )

    return resilient_llm_call(
        func=call_chain,
        fallback_func=fallback,
        max_retries=3,
        timeout=15.0,
    )


def ask_with_sources(question: str, vectorstore=None) -> dict:
    """Executa pergunta e retorna resposta com fontes recuperadas."""
    if vectorstore is None:
        vectorstore = build_vectorstore()

    retriever = get_retriever(vectorstore, k=RETRIEVER_K)
    chain = build_rag_chain(vectorstore)

    def call_chain():
        answer = chain.invoke(question)
        docs = retriever.invoke(question)
        return {
            "answer": answer,
            "sources": [
                {
                    "file": doc.metadata.get("source_file"),
                    "preview": doc.page_content[:150],
                }
                for doc in docs
            ],
            "question": question,
        }

    def fallback():
        docs = retriever.invoke(question)
        if not docs:
            return {
                "answer": "Não encontrei informações relevantes nos documentos técnicos.",
                "sources": [],
                "question": question,
            }
        chunks = [
            f"[{d.metadata.get('source_file', 'doc')}]\n{d.page_content}"
            for d in docs[:3]
        ]
        return {
            "answer": (
                "⚠️ LLM temporariamente indisponível. "
                "Informações dos documentos técnicos:\n\n"
                + "\n\n---\n\n".join(chunks)
            ),
            "sources": [
                {
                    "file": doc.metadata.get("source_file"),
                    "preview": doc.page_content[:150],
                }
                for doc in docs[:3]
            ],
            "question": question,
        }

    return resilient_llm_call(
        func=call_chain,
        fallback_func=fallback,
        max_retries=3,
        timeout=15.0,
    )


if __name__ == "__main__":
    vectorstore = build_vectorstore()

    perguntas = [
        "A vibração do motor está em 3.2 mm/s. Isso é preocupante?",
        (
            "O sensor Pepperl+Fuchs está com temperatura de 72°C. "
            "Qual é o limite máximo e o que devo fazer?"
        ),
        (
            "O modelo de ML detectou anomalia às 13:44 com severidade medium. "
            "O que isso significa e quais são os próximos passos?"
        ),
        (
            "O RUL do motor está estimado em 113 horas. Quando devo "
            "agendar a manutenção?"
        ),
    ]

    for pergunta in perguntas:
        print(f"\n{'=' * 60}")
        print(f"PERGUNTA: {pergunta}")
        print("=" * 60)
        resposta = ask(pergunta, vectorstore=vectorstore)
        print(resposta)
