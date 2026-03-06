
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from app.schemas.job import JobResult
from app.database.database import SessionLocal
from app.database.crud_job import create_job, get_job_by_job_id
from app.services.image_service import save_upload_image
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
    return {"job_id": job.job_id, "message": "Job created."}


@router.get("/job_status/{job_id}", response_model=JobResult)
async def get_job_status(job_id: str, db: Session = Depends(get_db)):
    job = get_job_by_job_id(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != "done" or not job.result:
        raise HTTPException(status_code=202, detail="Job is still processing")
    # Unpack job.result dict to JobResult
    return JobResult(**job.result)
