from datetime import datetime

from pydantic import BaseModel


class StudentSavedAnalysisCreate(BaseModel):
    job_id: str


class StudentSavedAnalysisResponse(BaseModel):
    id: int
    student_id: int
    job_id: str
    mood: str | None = None
    confidence: str | None = None
    summary: str | None = None
    saved_at: datetime

    model_config = {"from_attributes": True}
