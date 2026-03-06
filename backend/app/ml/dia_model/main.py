from __future__ import annotations
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from config import RagConfig
from dia_rag_pipeline import DIARagPipeline


def main() -> None:
    # Load root backend .env
    root_env = Path(__file__).resolve().parents[2] / ".env"
    load_dotenv(root_env)

    if len(sys.argv) < 3:
        print('Usage: python main.py "<image_path>" "<child_text_description>"')
        sys.exit(1)

    image_path = sys.argv[1]
    child_description = sys.argv[2]

    cfg = RagConfig.from_env()
    pipeline = DIARagPipeline(cfg)

    json_output = pipeline.run(image_path=image_path, child_description=child_description)
    print(json_output)


if __name__ == "__main__":
    main()