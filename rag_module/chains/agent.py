"""
Agente LangGraph — Forzy Digital Twin.
Tools: sensores, ML, DB completo, relatórios PDF/Excel/Word, RAG técnico.
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
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from rag_module.chains.llm_resilience import resilient_llm_call
from rag_module.chains.logger import CallTimer, estimate_tokens, get_logger
from rag_module.chains.memory import load_history, save_history
from rag_module.chains.rag_chain import get_llm
from rag_module.chains.vectorstore import build_vectorstore, get_retriever
from rag_module.config import SYSTEM_PROMPT
from rag_module.tools.sensor_tool import (
    get_sensor_status,
    get_ml_analysis,
    get_maintenance_history,
    get_active_alerts,
)
from rag_module.tools.db_tool import (
    get_db_leituras,
    get_db_diagnosticos,
    get_system_overview,
    get_motor_details,
    get_sensor_trends,
    get_alerts_history,
    get_maintenance_records,
    compare_motors,
)
from rag_module.tools.report_tool import generate_report, generate_motor_report
from rag_module.tools.web_search_tool import web_search
from rag_module.tools.motor_parts_tool import (
    get_motor_part_info,
    diagnose_by_signals,
    get_motor_baselines,
)

logger = get_logger("forzy.agent")


class AgentState(TypedDict):
    messages: Annotated[list, operator.add]


def create_rag_tool(vectorstore=None):
    from langchain_core.tools import tool

    vs = vectorstore if vectorstore is not None else build_vectorstore()

    @tool
    def search_technical_docs(query: str) -> str:
        """Busca nos manuais técnicos do motor WEG W22, sensor Pepperl+Fuchs e norma ISO 10816.
        Use para especificações técnicas, limites de operação, procedimentos e normas."""
        retriever = get_retriever(vs)
        docs = retriever.invoke(query)
        if not docs:
            return "Nenhuma informação encontrada nos documentos técnicos."
        return "\n\n---\n\n".join(
            f"[{d.metadata.get('source_file', 'doc')}]\n{d.page_content}" for d in docs
        )

    return search_technical_docs


def build_agent(vectorstore=None):
    """Constrói e compila o grafo LangGraph com todas as tools."""
    tools = [
        # Tempo real via API
        get_sensor_status,
        get_ml_analysis,
        get_active_alerts,
        get_maintenance_history,
        # Banco de dados completo
        get_db_leituras,
        get_db_diagnosticos,
        get_system_overview,
        get_motor_details,
        get_sensor_trends,
        get_alerts_history,
        get_maintenance_records,
        compare_motors,
        # Relatórios
        generate_report,
        generate_motor_report,
        # Documentação técnica
        create_rag_tool(vectorstore),
        # Pesquisa web
        web_search,
        # Peças do motor e diagnóstico por componente (modelo 3D)
        get_motor_part_info,
        diagnose_by_signals,
        get_motor_baselines,
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

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue)
    graph.add_edge("tools", "agent")

    return graph.compile()


def _rag_fallback(message: str, vectorstore=None) -> dict:
    from rag_module.chains.rag_chain import ask_with_sources
    vs = vectorstore or build_vectorstore()
    result = ask_with_sources(message, vectorstore=vs)
    return {
        "answer": "⚠️ Agente temporariamente indisponível. Resposta baseada nos documentos:\n\n" + result["answer"],
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
    """Envia mensagem ao agente LangGraph e retorna resposta com tools utilizadas."""
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

    logger.info("Chat iniciado", extra={
        "session_id": session_id,
        "question_preview": message[:80],
        "history_turns": len(persisted) // 2,
    })

    try:
        with CallTimer() as timer:
            raw = resilient_llm_call(
                func=lambda: agent.invoke({"messages": messages}),
                fallback_func=None,
                max_retries=3,
                timeout=30.0,
            )

        result_messages = raw["messages"]
        last_msg = result_messages[-1]
        answer = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

        tools_used = list({
            tc["name"] if isinstance(tc, dict) else getattr(tc, "name", "?")
            for msg in result_messages
            if hasattr(msg, "tool_calls") and msg.tool_calls
            for tc in msg.tool_calls
        })

        # Extrai token REPORT:: das respostas das tools (o LLM parafraseia o conteúdo)
        import re as _re
        report_token: str | None = None
        for msg in result_messages:
            content = msg.content if hasattr(msg, "content") else ""
            if isinstance(content, str):
                m = _re.search(r"REPORT::([\w\-]+\.(?:pdf|xlsx|docx))", content)
                if m:
                    report_token = m.group(1)
                    break

        logger.info("Chat concluído", extra={
            "session_id": session_id,
            "tools_used": tools_used,
            "response_time_ms": round(timer.elapsed_ms),
            "fallback_used": False,
        })

        save_history(session_id, result_messages)

        return {
            "answer": answer,
            "tools_used": tools_used,
            "messages": result_messages,
            "session_id": session_id,
            "response_time_ms": round(timer.elapsed_ms),
            "fallback_used": False,
            "report_token": report_token,
        }

    except Exception as exc:
        logger.error(f"Chat falhou: {exc}", extra={"session_id": session_id, "fallback_used": True})
        return _rag_fallback(message, vectorstore=vectorstore)
