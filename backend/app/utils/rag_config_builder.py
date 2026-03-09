from app.ml.dia_model.config import RagConfig
from app.core.config import settings

def build_rag_config_from_settings():
    dirs = RagConfig.default_dirs()
    return RagConfig(
        rag_dir=dirs["rag_dir"],
        data_dir=dirs["data_dir"],
        chroma_dir=dirs["chroma_dir"],
        llm_model=settings.GEMINI_MODEL,
        top_k=settings.RAG_TOP_K or 6,
    )