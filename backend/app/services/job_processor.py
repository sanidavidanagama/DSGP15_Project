from sqlalchemy.orm import Session
from app.database.crud_job import update_job_status_and_result
from app.services.image_service import run_image_processor
from app.services.emotion_service import run_emotion_pipeline
import threading

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

        # Run the emotion model in a thread (TODO: Add dia model threading and aggregation)
        emotion_result = {}
        def emotion_task():
            nonlocal emotion_result
            emotion_result = run_emotion_pipeline(processed_image_path, description)

        emotion_thread = threading.Thread(target=emotion_task)
        emotion_thread.start()
        emotion_thread.join()  # Wait for emotion model to finish
        update_job_status_and_result(db, job_id, status="emotion_processed", result=emotion_result)
        print("Predicted emotion: ", emotion_result)
        pass
    else:
        update_job_status_and_result(db, job_id, status="failed", result={"error": "Image processing failed"})
    
    # TODO: Update job status/result in DB after both models are run
    # TODO: Add dia model threading and aggregation
