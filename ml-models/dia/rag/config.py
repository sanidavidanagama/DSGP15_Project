from dataclasses import dataclass
from pathlib import Path
import os

@dataclass(frozen=True)
class RagConfig:
    rag_dir: Path
    data_dir: Path
    chroma_dir: Path

    llm_model: str
    embedding_model: str

    top_k: int

    @staticmethod
    def from_env() -> "RagConfig":
        rag_dir = Path(__file__).resolve().parent
        data_dir = rag_dir / "data"
        chroma_dir = data_dir / "chroma_db"

        llm_model = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")
        embedding_model = os.getenv("GEMINI_MODEL", "text-embedding-004")
        top_k = int(os.getenv("RAG_TOP_K", "6"))

        return RagConfig(
            rag_dir=rag_dir,
            data_dir=data_dir,
            chroma_dir=chroma_dir,
            llm_model=llm_model,
            embedding_model=embedding_model,
            top_k=top_k,
        )