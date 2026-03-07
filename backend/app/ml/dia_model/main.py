import sys
from dia_rag_pipeline import DIARagPipeline
from config import RagConfig
from app.core.config import settings

def main() -> None:
    if len(sys.argv) < 3:
        print('Usage: python main.py "<image_path>" "<child_text_description>"')
        sys.exit(1)

    image_path = sys.argv[1]
    child_description = sys.argv[2]

    dirs = RagConfig.default_dirs()
    rag_config = RagConfig(
        rag_dir=dirs["rag_dir"],
        data_dir=dirs["data_dir"],
        chroma_dir=dirs["chroma_dir"],
        llm_model=settings.GEMINI_MODEL,
        top_k=settings.RAG_TOP_K or 6,
    )
    pipeline = DIARagPipeline(rag_config, settings.GOOGLE_API_KEY)
    json_output = pipeline.run(image_path=image_path, child_description=child_description)
    print(json_output)

if __name__ == "__main__":
    main()