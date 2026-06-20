import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rag_module.chains.agent import build_agent, chat
from rag_module.chains.logger import get_logger, read_logs


def main():
    logger = get_logger("forzy.test")
    logger.info("Iniciando teste de logging")

    agent = build_agent()

    r1 = chat("Como está o motor?", agent=agent, session_id="log_test")
    print(
        f"Turno 1 — tempo: {r1.get('response_time_ms')}ms | "
        f"tools: {r1.get('tools_used')}"
    )

    r2 = chat("E o RUL?", agent=agent, session_id="log_test")
    print(
        f"Turno 2 — tempo: {r2.get('response_time_ms')}ms | "
        f"tools: {r2.get('tools_used')}"
    )

    print("\n--- Últimas 6 entradas do log ---")
    for entry in read_logs(6):
        print(json.dumps(entry, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
