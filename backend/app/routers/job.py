from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse
import uuid

router = APIRouter()

@router.post("/upload")
async def upload_image_with_description(
    image: UploadFile = File(...),
    description: str = Form(...)
):
    job_id = str(uuid.uuid4())
    # TODO: Save image, create job in DB, start processing (to be implemented)
    return JSONResponse(content={"job_id": job_id, "message": "Job created (placeholder)."})

@router.get("/job_status/{job_id}")
async def get_job_status(job_id: str):
    # TODO: Query job status/result from DB
    # Placeholder response
    return JSONResponse(content={"job_id": job_id, "status": "pending", "result": None})
