from fastapi import APIRouter, UploadFile, File, HTTPException
from app.schemas.image import ImageValidationResponse
from app.ml.image_model.processor import ChildDrawingPreprocessor
from PIL import Image
import io

router = APIRouter()

# Initialised once at module load (SAM model is expensive to load)
preprocessor = ChildDrawingPreprocessor()

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
MAX_FILE_SIZE_MB = 20
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


@router.post("/validate_image", response_model=ImageValidationResponse)
async def validate_image(image: UploadFile = File(...)):


    #  MIME type check 
    if image.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{image.content_type}'. "
                   f"Allowed types: {', '.join(ALLOWED_CONTENT_TYPES)}."
        )

    #  File size check 
    image_bytes = await image.read()

    if len(image_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"File size exceeds the {MAX_FILE_SIZE_MB} MB limit."
        )

    if len(image_bytes) == 0:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty."
        )

    # Integrity check (can it actually be decoded as an image?) 
    try:
        pil_image = Image.open(io.BytesIO(image_bytes))
        pil_image.verify()  # Catches truncated / corrupt files
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="File appears to be corrupt or is not a valid image."
        )

    # Preprocessor: paper detection + full processing pipeline 
    # Re-open after verify() since verify() exhausts the file handle
    try:
        processed_image = preprocessor.process(image_bytes)
    except RuntimeError as e:
        # RuntimeError is raised by the preprocessor when paper cannot be detected
        # or the mask fails the quality-gate checks
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Image could not be processed: {str(e)}"
        )

    return ImageValidationResponse(
        valid=True,
        message="Image validated and preprocessed successfully."
    )