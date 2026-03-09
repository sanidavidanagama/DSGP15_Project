from sqlalchemy import Column, String, Integer, DateTime, Text
from sqlalchemy.dialects.sqlite import JSON as SQLiteJSON
from sqlalchemy.sql import func
from app.database.database import Base
import uuid

class Job(Base):
    __tablename__ = "jobs"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String, unique=True, index=True, default=lambda: str(uuid.uuid4()))
    status = Column(String, default="pending")
    image_path = Column(String, nullable=False)
    processed_image_path = Column(String, nullable=True)
    description = Column(Text, nullable=True)
    result = Column(SQLiteJSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
