from dataclasses import dataclass
from pathlib import Path
import os

@dataclass(frozen=True)
class RagConfig:
    rag_dir: Path
    data_dir: Path
    chroma_dir: Path
    llm_model: str
    top_k: int

    @staticmethod
    def default_dirs() -> dict:
        rag_dir = Path(__file__).resolve().parent
        data_dir = rag_dir / "data"
        chroma_dir = rag_dir / "chroma_db"
        return {
            "rag_dir": rag_dir,
            "data_dir": data_dir,
            "chroma_dir": chroma_dir,
        }
