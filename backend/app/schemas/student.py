from datetime import datetime
from typing import Optional

from pydantic import BaseModel

from app.schemas.student_saved_analysis import StudentSavedAnalysisResponse


class StudentCreate(BaseModel):
    name: str
    gender: str


class StudentUpdate(BaseModel):
    name: Optional[str] = None
    gender: Optional[str] = None


class StudentResponse(BaseModel):
    id: int
    class_id: int
    name: str
    gender: str
    joined_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class StudentWithStatsResponse(StudentResponse):
    last_predicted_mood: Optional[str] = None
    last_predicted_at: Optional[datetime] = None
    total_analyses: int = 0


class StudentDetailResponse(StudentWithStatsResponse):
    history: list[StudentSavedAnalysisResponse] = []
