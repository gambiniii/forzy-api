import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


async def main():
    from rag_module.chains.agent import build_agent
    from rag_module.chains.vectorstore import build_vectorstore
    import rag_module.api.router as r

    print("Carregando vectorstore e agent...")
    r._vectorstore = build_vectorstore()
    r._agent = build_agent(r._vectorstore)

    print("\nChamando /chat/health...")
    response = await r.health()

    response_dict = response.model_dump()
    print(json.dumps(response_dict, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    asyncio.run(main())
