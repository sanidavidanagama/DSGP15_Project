from __future__ import annotations
import os
import sys

from dotenv import load_dotenv

from config import RagConfig
from dia_rag_pipeline import DIARagPipeline


def main() -> None:
    load_dotenv()
    # uv run ml-models/dia/rag/main.py C:\Users\sanid\PycharmProjects\DSGP\DSGP15_Project\ml-models\dataset\Dataset\Images\Emotion\test\Happiness\101-1B-267-F-H.jpg "I love my friends very, very much. I love my teacher too"
    # to run: python ml-models/dia/rag/main.py /path/to/image.jpg "child description here"
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
