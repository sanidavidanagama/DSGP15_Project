
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from app.schemas.job import JobResult
from app.database.database import SessionLocal
from app.database.crud_job import create_job, get_job_by_job_id
from app.services.image_service import save_upload_image
from app.services.job_processor import process_job
import threading
import uuid

router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()



@router.post("/upload")
async def upload_image_with_description(
    image: UploadFile = File(...),
    description: str = Form(...),
    db: Session = Depends(get_db)
):
    job_id = str(uuid.uuid4())
    image_path = save_upload_image(image)
    job = create_job(db, job_id=job_id, image_path=image_path, description=description)
    # Start background processing in a new thread
    threading.Thread(target=process_job, args=(job_id, image_path, description, SessionLocal())).start()
    return {"job_id": job.job_id, "message": "Job created and processing started."}



@router.get("/job_status/{job_id}")
async def get_job_status(job_id: str, db: Session = Depends(get_db)):
    job = get_job_by_job_id(db, job_id)
    if not job:
        return {"status": "not_found", "detail": "Job not found"}
    # Return job status and details directly from the database
    return {
        "job_id": job.job_id,
        "status": job.status,
        "processed_image_path": job.processed_image_path,
        "result": job.result,
    }
