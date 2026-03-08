from sqlalchemy.orm import Session
from app.database.crud_job import update_job_status_and_result
from app.services.image_service import run_image_processor, build_image_metadata

from app.services.emotion_service import run_emotion_pipeline
from app.core.config import settings
import threading
import json

# Import DIARagPipeline from the correct path


from app.ml.dia_model.dia_rag_pipeline import DIARagPipeline
from app.ml.dia_model.config import RagConfig

# Recommendation engine imports
from app.ml.recommendation_model.recommendations_engine import RecommendationEngine
from app.utils.recommendation_input_builder import RecommendationInputBuilder

def process_job(job_id: str, image_path: str, description: str, db: Session):
    # Clear status at start
    update_job_status_and_result(db, job_id, status="processing", result=None, processed_image_path=None)

    # Run image processor
    processed_image_path = run_image_processor(image_path)
    if not processed_image_path:
        update_job_status_and_result(db, job_id, status="failed", result={"error": "Image processing failed"})
        return

    update_job_status_and_result(db, job_id, status="image_processed", processed_image_path=processed_image_path)

    # Prepare results containers
    emotion_result = {}
    dia_result = {}
    recommendation_result = {}

    # Threaded tasks
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

    # Run emotion and DIA in parallel first
    threads = [
        threading.Thread(target=emotion_task),
        threading.Thread(target=dia_task)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Now run recommendation engine (can be threaded if needed, but depends on emotion/dia)
    def recommendation_task():
        nonlocal recommendation_result
        engine = RecommendationEngine()
        mood, data = RecommendationInputBuilder.build(emotion_result, dia_result)
        recommendation_result = engine.generate_recommendation(mood, data)

    rec_thread = threading.Thread(target=recommendation_task)
    rec_thread.start()
    rec_thread.join()

    # Aggregate results
    image_metadata = build_image_metadata(processed_image_path)

    result = {
        "image": image_metadata,
        "emotion": emotion_result,
        "dia": dia_result,
        "recommendation": recommendation_result
    }
    update_job_status_and_result(db, job_id, status="done", result=result)
    