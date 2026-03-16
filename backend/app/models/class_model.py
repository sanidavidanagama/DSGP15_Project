from sqlalchemy import Boolean, Column, DateTime, Integer, String, Text
from sqlalchemy.dialects.sqlite import JSON as SQLiteJSON
from sqlalchemy.sql import func

from app.database.database import Base


class Classroom(Base):
    __tablename__ = "classes"

    id = Column(Integer, primary_key=True, index=True)
    teacher_id = Column(String, nullable=False, index=True)
    class_name = Column(String, nullable=False)
    grade_age_group = Column(String, nullable=False)
    schedule_days = Column(SQLiteJSON, nullable=False)
    description = Column(Text, nullable=True)
    is_deleted = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
