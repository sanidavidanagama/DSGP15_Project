from sqlalchemy.orm import Session
from app.models.job import Job
from typing import Optional

# Create a new job
def create_job(db: Session, job_id: str, image_path: str, description: str) -> Job:
    job = Job(job_id=job_id, image_path=image_path, description=description)
    db.add(job)
    db.commit()
    db.refresh(job)
    return job

# Get a job by job_id
def get_job_by_job_id(db: Session, job_id: str) -> Optional[Job]:
    return db.query(Job).filter(Job.job_id == job_id).first()

# Update job status and result
def update_job_status_and_result(db: Session, job_id: str, status: str, result: dict = None, processed_image_path: str = None):
    job = get_job_by_job_id(db, job_id)
    if job:
        job.status = status
        if result is not None:
            job.result = result
        if processed_image_path is not None:
            job.processed_image_path = processed_image_path
        db.commit()
        db.refresh(job)
    return job
