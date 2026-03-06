"""
Test Output Format Functions

Tests:
1. process_to_bytes()
2. process_to_pil()
"""

from backend.app.ml.image_model.processor import ChildDrawingPreprocessor
from PIL import Image


def test_process_to_bytes(image_path):

    print("\nTEST: process_to_bytes()")

    preprocessor = ChildDrawingPreprocessor()

    output_bytes = preprocessor.process_to_bytes(image_path, format="JPEG")

    print(f"Bytes length: {len(output_bytes)}")

    with open("bytes_output.jpg", "wb") as f:
        f.write(output_bytes)

    print("Saved bytes_output.jpg")


def test_process_to_pil(image_path):

    print("\nTEST: process_to_pil()")

    preprocessor = ChildDrawingPreprocessor()

    pil_img = preprocessor.process_to_pil(image_path)

    print(f"PIL image size: {pil_img.size}")
    print(f"PIL mode: {pil_img.mode}")

    pil_img.save("pil_output.jpg")

    print("Saved pil_output.jpg")


if __name__ == "__main__":

    image_path = "../data/black bg2.jpeg" 

    test_process_to_bytes(image_path)
    test_process_to_pil(image_path)