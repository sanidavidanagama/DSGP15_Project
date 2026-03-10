import os
from uuid import uuid4
from fastapi import UploadFile
from datetime import datetime
from PIL import Image
from app.ml.image_model.processor import ChildDrawingPreprocessor
import os
from app.core.config import settings

RAW_IMAGE_DIR = settings.RAW_IMAGE_DIR

os.makedirs(RAW_IMAGE_DIR, exist_ok=True)

def save_upload_image(upload_file: UploadFile) -> str:
    ext = os.path.splitext(upload_file.filename)[1]
    filename = f"{uuid4().hex}{ext}"
    file_path = os.path.join(RAW_IMAGE_DIR, filename)
    with open(file_path, "wb") as f:
        f.write(upload_file.file.read())
    return file_path


def run_image_processor(image_path: str):
    """
    Runs the ChildDrawingPreprocessor on the given image_path and saves the processed image.
    Returns the output path if successful, None otherwise.
    """

    image_processor = ChildDrawingPreprocessor()
    try:
        result = image_processor.process(image_path)
        # Save processed image
        processed_dir = settings.PROCESSED_IMAGE_DIR
        os.makedirs(processed_dir, exist_ok=True)
        output_path = os.path.join(
            processed_dir,
            f"processed_{os.path.basename(image_path)}"
        )
        Image.fromarray(result).save(output_path)
        return output_path
    except Exception as e:
        print(f"Image processing failed: {e}")
        return None
    
def build_image_metadata(processed_image_path):
    file_size = os.path.getsize(processed_image_path)

    with Image.open(processed_image_path) as img:
        width, height = img.size

    return {
        "processed_image_path": processed_image_path,
        "created_at": datetime.utcnow().isoformat(),
        "file_type": processed_image_path.split(".")[-1],
        "size": file_size,
        "dimensions": f"{width}x{height}"
    }

