import sys
from dia_rag_pipeline import DIARagPipeline


def main() -> None:
    if len(sys.argv) < 4:
        print('Usage: python main.py "<GOOGLE_API_KEY>" "<image_path>" "<child_text_description>"')
        sys.exit(1)

    api_key = sys.argv[1]
    image_path = sys.argv[2]
    child_description = sys.argv[3]

    pipeline = DIARagPipeline(api_key)
    json_output = pipeline.run(image_path=image_path, child_description=child_description)
    print(json_output)

if __name__ == "__main__":
    main()