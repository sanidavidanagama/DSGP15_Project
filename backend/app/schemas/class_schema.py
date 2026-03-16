from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, field_validator

VALID_DAYS = {"Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"}


class ClassCreate(BaseModel):
    class_name: str
    grade_age_group: str
    schedule_days: List[str]
    description: Optional[str] = None

    @field_validator("schedule_days")
    @classmethod
    def validate_schedule_days(cls, value: List[str]) -> List[str]:
        if not value:
            raise ValueError("At least one schedule day must be selected.")
        invalid_days = set(value) - VALID_DAYS
        if invalid_days:
            raise ValueError("Invalid schedule days provided.")
        return value


class ClassUpdate(BaseModel):
    class_name: Optional[str] = None
    grade_age_group: Optional[str] = None
    schedule_days: Optional[List[str]] = None
    description: Optional[str] = None

    @field_validator("schedule_days")
    @classmethod
    def validate_schedule_days(cls, value: Optional[List[str]]) -> Optional[List[str]]:
        if value is None:
            return value
        if not value:
            raise ValueError("At least one schedule day must be selected.")
        invalid_days = set(value) - VALID_DAYS
        if invalid_days:
            raise ValueError("Invalid schedule days provided.")
        return value


class ClassResponse(BaseModel):
    id: int
    teacher_id: str
    class_name: str
    grade_age_group: str
    schedule_days: List[str]
    description: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    model_config = {"from_attributes": True}
