from sqlalchemy.orm import Session
from app.database.crud_job import update_job_status_and_result
from app.services.image_service import run_image_processor

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
    processed_image_path = run_image_processor(image_path)
    if processed_image_path != None:
        update_job_status_and_result(db, job_id, status="image_processed", processed_image_path=processed_image_path)
    
    # TODO: Run the mood and dia ML pipelines in parallel (using threading or concurrency)
    # TODO: Wait for both results, then call the recommendations model.
    # TODO: Update job status to "done" and save the result in the database.

    pass
