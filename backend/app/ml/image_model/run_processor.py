"""
Run the Child Drawing Preprocessor

Usage:
python run_preprocessor.py path/to/image.jpg
"""

import sys
from PIL import Image
from backend.app.ml.image_model.processor import ChildDrawingPreprocessor


def main(image_path):
    preprocessor = ChildDrawingPreprocessor()

    try:
        result = preprocessor.process(image_path)

        output_path = "processed_output.jpg"
        Image.fromarray(result).save(output_path)

    except Exception as e:
        print("Processing failed.")
        print(str(e))


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python run_preprocessor.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    main(image_path)