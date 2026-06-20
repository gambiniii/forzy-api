"""
Vectorstore Chroma — carregamento, indexação e retrieval de documentos RAG.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_module.config import (
    CHROMA_PERSIST_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DOCUMENTS_DIR,
    EMBEDDING_MODEL,
    MOTOR_WEG_SPECS,
)

_last_build_stats: dict = {
    "documents_loaded": 0,
    "chunks_created": 0,
    "embeddings_count": 0,
}


def get_embeddings() -> HuggingFaceEmbeddings:
    """Retorna modelo de embeddings HuggingFace para CPU."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def load_documents(documents_dir: Path) -> list:
    """Carrega PDFs e TXTs do diretório de documentos."""
    documents = []

    for filepath in sorted(documents_dir.iterdir()):
        if filepath.suffix.lower() == ".pdf":
            loaded = PyPDFLoader(str(filepath)).load()
        elif filepath.suffix.lower() == ".txt":
            loaded = TextLoader(str(filepath), encoding="utf-8").load()
        else:
            continue

        for doc in loaded:
            doc.metadata["source_file"] = filepath.name
        documents.extend(loaded)

    return documents


def split_documents(documents: list) -> list:
    """Divide documentos em chunks com overlap."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def _vectorstore_has_data(persist_dir: Path) -> bool:
    """Verifica se o diretório de persistência contém dados do Chroma."""
    if not persist_dir.exists():
        return False
    entries = [p for p in persist_dir.iterdir() if p.name != ".gitkeep"]
    return len(entries) > 0


def build_vectorstore(force_rebuild: bool = False) -> Chroma:
    """Constrói ou carrega vectorstore Chroma persistido."""
    global _last_build_stats

    embeddings = get_embeddings()

    if _vectorstore_has_data(CHROMA_PERSIST_DIR) and not force_rebuild:
        vectorstore = Chroma(
            persist_directory=str(CHROMA_PERSIST_DIR),
            embedding_function=embeddings,
        )
        count = vectorstore._collection.count()
        _last_build_stats = {
            "documents_loaded": 0,
            "chunks_created": count,
            "embeddings_count": count,
        }
        print("[OK] Vectorstore carregado do disco")
        return vectorstore

    documents = load_documents(DOCUMENTS_DIR)
    chunks = split_documents(documents)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_PERSIST_DIR),
    )

    count = vectorstore._collection.count()
    _last_build_stats = {
        "documents_loaded": len(documents),
        "chunks_created": len(chunks),
        "embeddings_count": count,
    }
    print("[OK] Vectorstore criado e persistido")
    return vectorstore


def get_retriever(vectorstore: Chroma | None = None, k: int = 4):
    """Retorna retriever de similaridade sobre o vectorstore."""
    if vectorstore is None:
        vectorstore = build_vectorstore()
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )


def test_retrieval(query: str | None = None) -> None:
    """Testa retrieval e imprime top 3 resultados."""
    query = query or "vibração alta no motor, o que pode ser?"
    retriever = get_retriever()
    results = retriever.invoke(query)

    print(f"\nQuery: {query}")
    print("-" * 60)
    for idx, doc in enumerate(results[:3], start=1):
        source = doc.metadata.get("source_file", "desconhecido")
        preview = doc.page_content[:200].replace("\n", " ")
        print(f"[{idx}] source: {source}")
        print(f"    preview: {preview}...")
    print("-" * 60)


def get_build_stats() -> dict:
    """Retorna estatísticas da última construção/carregamento."""
    return _last_build_stats.copy()


if __name__ == "__main__":
    import sys as _sys

    force_rebuild = "--rebuild" in _sys.argv

    if _vectorstore_has_data(CHROMA_PERSIST_DIR) and not force_rebuild:
        print(f"Vectorstore existente encontrado em: {CHROMA_PERSIST_DIR}")
        print("Use --rebuild para recriar o índice.")
    elif force_rebuild:
        print("Modo --rebuild: recriando vectorstore...")

    _ = MOTOR_WEG_SPECS  # importado conforme especificação do prompt

    vectorstore = build_vectorstore(force_rebuild=force_rebuild)
    stats = get_build_stats()

    print(f"\nTotal de documentos carregados: {stats['documents_loaded']}")
    print(f"Total de chunks criados: {stats['chunks_created']}")
    print(f"Tamanho do vectorstore (embeddings): {stats['embeddings_count']}")

    test_retrieval("vibração alta no motor, o que pode ser?")
    test_retrieval("qual o limite de temperatura do sensor?")
    test_retrieval("como interpretar o RUL do motor?")
