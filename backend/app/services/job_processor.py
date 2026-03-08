from sqlalchemy.orm import Session
from app.database.crud_job import update_job_status_and_result
from app.services.image_service import run_image_processor

from app.services.emotion_service import run_emotion_pipeline
from app.core.config import settings
import threading
import json

# Import DIARagPipeline from the correct path

from app.ml.dia_model.dia_rag_pipeline import DIARagPipeline
from app.ml.dia_model.config import RagConfig
from app.utils.rag_config_builder import build_rag_config_from_settings

def process_job(job_id: str, image_path: str, description: str, db: Session):
    update_job_status_and_result(db, job_id, status="processing", result=None, processed_image_path=None)

    processed_image_path = run_image_processor(image_path)
    if not processed_image_path:
        update_job_status_and_result(db, job_id, status="failed", result={"error": "Image processing failed"})
        return

    update_job_status_and_result(db, job_id, status="image_processed", processed_image_path=processed_image_path)

    emotion_result = {}
    dia_result = {}

    def emotion_task():
        nonlocal emotion_result
        emotion_result = run_emotion_pipeline(processed_image_path, description)

    def dia_task():
        nonlocal dia_result
        rag_config = RagConfig.from_settings()  # automatically includes API key
        pipeline = DIARagPipeline(rag_config)
        dia_result = pipeline.run(processed_image_path, description)

    # run in parallel
    threads = [
        threading.Thread(target=emotion_task),
        threading.Thread(target=dia_task)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    result = {"emotion": emotion_result, "dia": dia_result}
    update_job_status_and_result(db, job_id, status="emotions_and_dia_processed", result=result)
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
    if processed_image_path is not None:
        update_job_status_and_result(db, job_id, status="image_processed", processed_image_path=processed_image_path)

        # Prepare threading for emotion and dia models
        emotion_result = {}
        dia_result = {}

        def emotion_task():
            nonlocal emotion_result
            emotion_result = run_emotion_pipeline(processed_image_path, description)

        def dia_task():
            nonlocal dia_result
            rag_config = RagConfig.from_settings()
            pipeline = DIARagPipeline(rag_config)
            raw_result = pipeline.run(processed_image_path, description)

            # Convert JSON string -> dict
            try:
                dia_result = json.loads(raw_result)
            except json.JSONDecodeError:
                dia_result = {"error": "Invalid JSON returned by DIA", "raw": raw_result}

        # Start both threads
        emotion_thread = threading.Thread(target=emotion_task)
        dia_thread = threading.Thread(target=dia_task)
        emotion_thread.start()
        dia_thread.start()
        emotion_thread.join()
        dia_thread.join()

        # Aggregate results
        result = {
            "emotion": emotion_result,
            "dia": dia_result
        }
        update_job_status_and_result(db, job_id, status="emotions_and_dia_processed", result=result)
        
    else:
        update_job_status_and_result(db, job_id, status="failed", result={"error": "Image processing failed"})
