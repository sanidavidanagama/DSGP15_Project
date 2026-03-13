from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class StudentCreate(BaseModel):
    name: str
    age_group: str


class StudentUpdate(BaseModel):
    name: Optional[str] = None
    age_group: Optional[str] = None


class StudentResponse(BaseModel):
    id: int
    class_id: int
    name: str
    age_group: str
    joined_at: Optional[datetime] = None

    model_config = {"from_attributes": True}
