from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, field_validator

DAY_ALIASES = {
    "mon": "Monday",
    "monday": "Monday",
    "tue": "Tuesday",
    "tues": "Tuesday",
    "tuesday": "Tuesday",
    "wed": "Wednesday",
    "weds": "Wednesday",
    "wednesday": "Wednesday",
    "thu": "Thursday",
    "thur": "Thursday",
    "thurs": "Thursday",
    "thursday": "Thursday",
    "fri": "Friday",
    "friday": "Friday",
    "sat": "Saturday",
    "saturday": "Saturday",
    "sun": "Sunday",
    "sunday": "Sunday",
}
VALID_DAYS = set(DAY_ALIASES.values())


def _normalize_schedule_days(value: List[str]) -> List[str]:
    normalized_days: List[str] = []

    for day in value:
        raw = str(day).strip()
        mapped = DAY_ALIASES.get(raw.lower())
        if not mapped:
            raise ValueError("Invalid schedule days provided.")
        if mapped not in normalized_days:
            normalized_days.append(mapped)

    return normalized_days


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
        normalized = _normalize_schedule_days(value)
        invalid_days = set(normalized) - VALID_DAYS
        if invalid_days:
            raise ValueError("Invalid schedule days provided.")
        return normalized


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
        normalized = _normalize_schedule_days(value)
        invalid_days = set(normalized) - VALID_DAYS
        if invalid_days:
            raise ValueError("Invalid schedule days provided.")
        return normalized


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
