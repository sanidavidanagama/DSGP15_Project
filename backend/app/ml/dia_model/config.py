# app/ml/dia_model/config.py
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class RagConfig:
    rag_dir: Path
    data_dir: Path
    chroma_dir: Path
    llm_model: str
    top_k: int
    api_key: str

    @staticmethod
    def from_settings():
        """Build RagConfig using FastAPI Pydantic settings"""
        from app.core.config import settings

        base_dir = Path(__file__).resolve().parent
        dirs = {
            "rag_dir": base_dir,
            "data_dir": base_dir / "data",
            "chroma_dir": base_dir / "chroma_db",
        }

        return RagConfig(
            rag_dir=dirs["rag_dir"],
            data_dir=dirs["data_dir"],
            chroma_dir=dirs["chroma_dir"],
            llm_model=settings.GEMINI_MODEL,
            top_k=settings.RAG_TOP_K,
            api_key=settings.GOOGLE_API_KEY,
        )