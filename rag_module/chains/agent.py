"""
Agente LangGraph — assistente Forzy com tools de sensores, ML e RAG.
"""

from __future__ import annotations

import operator
import sys
from pathlib import Path
from typing import Annotated, TypedDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from rag_module.chains.llm_resilience import resilient_llm_call
from rag_module.chains.logger import CallTimer, estimate_tokens, get_logger
from rag_module.chains.memory import load_history, save_history
from rag_module.chains.rag_chain import get_llm
from rag_module.chains.vectorstore import build_vectorstore, get_retriever
from rag_module.config import SYSTEM_PROMPT
from rag_module.tools.sensor_tool import (
    get_active_alerts,
    get_maintenance_history,
    get_ml_analysis,
    get_sensor_status,
)
from rag_module.tools.db_tool import get_db_leituras, get_db_diagnosticos
from rag_module.tools.report_tool import generate_motor_report

logger = get_logger("forzy.agent")


class AgentState(TypedDict):
    messages: Annotated[list, operator.add]


def create_rag_tool(vectorstore=None):
    """Cria tool de busca nos documentos técnicos."""
    vs = vectorstore if vectorstore is not None else build_vectorstore()

    @tool
    def search_technical_docs(query: str) -> str:
        """Busca informações nos manuais técnicos do motor WEG W22, sensor Pepperl+Fuchs e norma ISO 10816. Use para responder dúvidas sobre especificações técnicas, limites de operação, procedimentos de manutenção e interpretação de normas."""
        retriever = get_retriever(vs)
        docs = retriever.invoke(query)
        if not docs:
            return "Nenhuma informação encontrada nos documentos."
        return "\n\n---\n\n".join(
            [
                f"[{d.metadata.get('source_file', 'doc')}]\n{d.page_content}"
                for d in docs
            ]
        )

    return search_technical_docs


def build_agent(vectorstore=None):
    """Constrói e compila o grafo LangGraph do agente."""
    tools = [
        get_sensor_status,
        get_ml_analysis,
        get_maintenance_history,
        get_active_alerts,
        get_db_leituras,
        get_db_diagnosticos,
        generate_motor_report,
        create_rag_tool(vectorstore),
    ]

    llm = get_llm(temperature=0.1)
    llm_with_tools = llm.bind_tools(tools)

    def agent_node(state: AgentState):
        messages = state["messages"]
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def should_continue(state: AgentState):
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return END

    tool_node = ToolNode(tools)

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue)
    graph.add_edge("tools", "agent")

    return graph.compile()


def _extract_tools_used(messages: list) -> list[str]:
    """Extrai nomes das tools invocadas na conversa."""
    used: list[str] = []
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
                if name and name not in used:
                    used.append(name)
    return used


def _run_agent(agent, messages):
    return agent.invoke({"messages": messages})


def _rag_fallback(message, vectorstore=None):
    """Fallback: responde só com RAG quando agent falha."""
    from rag_module.chains.rag_chain import ask_with_sources

    vs = vectorstore or build_vectorstore()
    result = ask_with_sources(message, vectorstore=vs)
    return {
        "answer": "⚠️ Agente temporariamente indisponível. "
        "Resposta baseada apenas nos documentos:\n\n"
        + result["answer"],
        "tools_used": [],
        "messages": [],
        "session_id": "fallback",
        "fallback_used": True,
    }


def chat(
    message: str,
    agent=None,
    history: list | None = None,
    session_id: str = "default",
    vectorstore=None,
) -> dict:
    """Envia mensagem ao agente e retorna resposta com tools utilizadas."""
    if agent is None:
        agent = build_agent(vectorstore)

    persisted = load_history(session_id) if not history else history

    messages = []
    for h in persisted:
        if h["role"] == "user":
            messages.append(HumanMessage(content=h["content"]))
        else:
            messages.append(AIMessage(content=h["content"]))
    messages.append(HumanMessage(content=message))

    logger.info(
        "Chat iniciado",
        extra={
            "session_id": session_id,
            "question_preview": message[:80],
            "history_turns": len(persisted) // 2,
        },
    )

    try:
        with CallTimer() as timer:
            raw = resilient_llm_call(
                func=lambda: _run_agent(agent, messages),
                fallback_func=None,
                max_retries=3,
                timeout=20.0,
            )
        result_messages = raw["messages"]

        last_msg = result_messages[-1]
        answer = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

        tools_used = []
        for msg in result_messages:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                tools_used.extend([tc["name"] for tc in msg.tool_calls])
        tools_used = list(set(tools_used))

        tokens_in = sum(
            estimate_tokens(m.content) for m in messages if hasattr(m, "content")
        )
        tokens_out = estimate_tokens(answer)

        logger.info(
            "Chat concluído",
            extra={
                "session_id": session_id,
                "tools_used": tools_used,
                "response_time_ms": round(timer.elapsed_ms),
                "tokens_estimated": tokens_in + tokens_out,
                "fallback_used": False,
            },
        )

        save_history(session_id, result_messages)

        return {
            "answer": answer,
            "tools_used": tools_used,
            "messages": result_messages,
            "session_id": session_id,
            "response_time_ms": round(timer.elapsed_ms),
            "fallback_used": False,
        }

    except Exception as exc:
        logger.error(
            f"Chat falhou: {exc}",
            extra={
                "session_id": session_id,
                "fallback_used": True,
                "response_time_ms": 0,
            },
        )
        return _rag_fallback(message, vectorstore=vectorstore)


if __name__ == "__main__":
    print("Inicializando agente Forzy Digital Twin...")
    vectorstore = build_vectorstore()
    agent = build_agent(vectorstore)

    conversa = [
        "Como está o motor agora?",
        "Tem alguma anomalia detectada?",
        (
            "Com base no RUL e no histórico de manutenção, "
            "quando devo agendar a próxima parada?"
        ),
    ]

    for turno, pergunta in enumerate(conversa, start=1):
        print(f"\n{'=' * 60}")
        print(f"TURNO {turno}")
        print(f"PERGUNTA: {pergunta}")
        print("=" * 60)

        resultado = chat(pergunta, agent=agent, session_id="demo")

        print(f"\nTools utilizadas: {resultado['tools_used'] or ['nenhuma']}")
        print(f"\nRESPOSTA:\n{resultado['answer']}")
