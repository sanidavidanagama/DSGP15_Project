"""
Paper Validity Checker

Checks whether the uploaded image contains a valid sheet of paper.

Usage:
python validate_paper.py image.jpg
"""

import sys
import cv2
import numpy as np
from backend.app.ml.image_model.processor import ChildDrawingPreprocessor


def validate_image(image_path):

    preprocessor = ChildDrawingPreprocessor()

    try:
        img = preprocessor._load_image(image_path)
        img = preprocessor._resize_if_large(img)

        mask = preprocessor._detect_paper_fast(img)

        valid = preprocessor._validate_paper_mask(mask, img.shape[:2])

        if valid:
            print("Paper detected successfully.")
            return True

        else:
            print("Paper mask failed validation.")
            return False

    except Exception as e:
        print("IMAGE REJECTED")
        print(str(e))
        return False


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python validate_paper.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    validate_image(image_path)