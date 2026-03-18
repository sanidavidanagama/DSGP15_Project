from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.sql import func

from app.database.database import Base


class StudentSavedAnalysis(Base):
    __tablename__ = "student_saved_analyses"
    __table_args__ = (UniqueConstraint("student_id", "job_id", name="uq_student_job"),)

    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(Integer, ForeignKey("students.id"), nullable=False, index=True)
    job_id = Column(String, nullable=False, index=True)
    mood = Column(String, nullable=True)
    confidence = Column(String, nullable=True)
    summary = Column(String, nullable=True)
    saved_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
