
from sqlalchemy.orm import Session
from app.database.crud_job import update_job_status_and_result
from PIL import Image
from app.ml.image_model.processor import ChildDrawingPreprocessor
import os
from app.core.config import settings

def process_job(job_id: str, image_path: str, description: str, db: Session):
    """
    Main job processing function. To be called in background after job creation.
    Args:
        job_id (str): The job's unique ID
        image_path (str): Path to the uploaded image
        description (str): User-provided description
        db (Session): SQLAlchemy DB session
    """
    # Update job status to 'processing', clear processed_image_path and result
    update_job_status_and_result(db, job_id, status="processing", result=None, processed_image_path=None)

    # Run the image processor, save processed image
    processed_image_path = _run_image_processor(image_path)
    if processed_image_path != None:
        update_job_status_and_result(db, job_id, status="image_processed", processed_image_path=processed_image_path)
    
    # TODO: Run the mood and dia ML pipelines in parallel (using threading or concurrency)
    # TODO: Wait for both results, then call the recommendations model.
    # TODO: Update job status to "done" and save the result in the database.

    pass


def _run_image_processor(image_path: str):
    """
    Runs the ChildDrawingPreprocessor on the given image_path and saves the processed image.
    Returns the output path if successful, None otherwise.
    """

    preprocessor = ChildDrawingPreprocessor()
    try:
        result = preprocessor.process(image_path)
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
    


