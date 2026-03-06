import os
from uuid import uuid4
from fastapi import UploadFile

RAW_IMAGE_DIR = "backend/uploads/raw/"

os.makedirs(RAW_IMAGE_DIR, exist_ok=True)

def save_upload_image(upload_file: UploadFile) -> str:
    ext = os.path.splitext(upload_file.filename)[1]
    filename = f"{uuid4().hex}{ext}"
    file_path = os.path.join(RAW_IMAGE_DIR, filename)
    with open(file_path, "wb") as f:
        f.write(upload_file.file.read())
    return file_path
