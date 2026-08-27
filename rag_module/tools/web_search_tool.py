"""
Tool LangChain — pesquisa web via DuckDuckGo.
Usada quando o usuário pede informações externas: normas técnicas, datasheets,
especificações de fabricante, preços, artigos, perguntas gerais fora do escopo do DB.
"""

from __future__ import annotations

from langchain_core.tools import tool


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """Faz pesquisa na web sobre qualquer assunto usando DuckDuckGo.
    Use quando o usuário pedir informações que não estão no banco de dados ou nos
    documentos técnicos: normas externas, datasheets de componentes, artigos técnicos,
    preços de peças, tutoriais de manutenção, especificações de fabricantes, etc.

    Parâmetros:
      query: o que pesquisar (em português ou inglês)
      max_results: número de resultados (padrão 5, máximo 10)

    Exemplos de uso:
      - "datasheet sensor Pepperl+Fuchs VIM32PL"
      - "norma ISO 10816 vibração motores"
      - "motor WEG W22 manual manutenção"
      - "como calcular RUL de motor elétrico"
    """
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS

        n = min(int(max_results), 10)
        results = []

        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=n):
                title = r.get("title", "Sem título")
                href  = r.get("href", "")
                body  = r.get("body", "")[:300]
                results.append(f"**{title}**\n{href}\n{body}")

        if not results:
            return f"Nenhum resultado encontrado para: {query}"

        header = f"Resultados da web para '{query}' ({len(results)} encontrados):\n\n"
        return header + "\n\n---\n\n".join(results)

    except Exception as exc:
        return f"Erro na pesquisa web: {exc}"
