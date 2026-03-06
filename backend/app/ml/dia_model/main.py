import sys
from dia_rag_pipeline import DIARagPipeline

def main() -> None:
    if len(sys.argv) < 3:
        print('Usage: python main.py "<image_path>" "<child_text_description>"')
        sys.exit(1)

    image_path = sys.argv[1]
    child_description = sys.argv[2]

    pipeline = DIARagPipeline()
    json_output = pipeline.run(image_path=image_path, child_description=child_description)
    print(json_output)

if __name__ == "__main__":
    main()