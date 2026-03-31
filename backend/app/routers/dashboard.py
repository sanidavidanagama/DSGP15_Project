from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.database.database import SessionLocal
from app.models.class_model import Classroom
from app.models.student import Student
from app.models.student_saved_analysis import StudentSavedAnalysis
from app.schemas.dashboard import DashboardOverviewResponse, DashboardRecentActivityItem
from app.services.auth_service import get_current_teacher

router = APIRouter(prefix="/dashboard", tags=["Dashboard"])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_teacher_id(teacher_id: str = Depends(get_current_teacher)) -> str:
    return teacher_id


def _relative_time_label(value: datetime) -> str:
    now = datetime.now(timezone.utc)
    ts = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    delta = now - ts
    seconds = max(0, int(delta.total_seconds()))

    if seconds < 60:
        return "just now"
    if seconds < 3600:
        minutes = seconds // 60
        unit = "minute" if minutes == 1 else "minutes"
        return f"{minutes} {unit} ago"
    if seconds < 86400:
        hours = seconds // 3600
        unit = "hour" if hours == 1 else "hours"
        return f"{hours} {unit} ago"

    days = seconds // 86400
    unit = "day" if days == 1 else "days"
    return f"{days} {unit} ago"


@router.get("/overview", response_model=DashboardOverviewResponse)
def get_dashboard_overview(
    teacher_id: str = Depends(get_teacher_id),
    db: Session = Depends(get_db),
):
    total_students = (
        db.query(func.count(Student.id))
        .join(Classroom, Student.class_id == Classroom.id)
        .filter(
            Classroom.teacher_id == teacher_id,
            Classroom.is_deleted == False,
            Student.is_deleted == False,
        )
        .scalar()
        or 0
    )

    active_classes = (
        db.query(func.count(Classroom.id))
        .filter(Classroom.teacher_id == teacher_id, Classroom.is_deleted == False)
        .scalar()
        or 0
    )

    analyses_base_query = (
        db.query(StudentSavedAnalysis)
        .join(Student, StudentSavedAnalysis.student_id == Student.id)
        .join(Classroom, Student.class_id == Classroom.id)
        .filter(
            Classroom.teacher_id == teacher_id,
            Classroom.is_deleted == False,
            Student.is_deleted == False,
        )
    )

    total_analyses = analyses_base_query.count()

    now = datetime.now(timezone.utc)
    week_start = (now - timedelta(days=now.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
    analyses_this_week = analyses_base_query.filter(StudentSavedAnalysis.saved_at >= week_start).count()

    recent_rows = (
        db.query(StudentSavedAnalysis, Student.name)
        .join(Student, StudentSavedAnalysis.student_id == Student.id)
        .join(Classroom, Student.class_id == Classroom.id)
        .filter(
            Classroom.teacher_id == teacher_id,
            Classroom.is_deleted == False,
            Student.is_deleted == False,
        )
        .order_by(StudentSavedAnalysis.saved_at.desc())
        .limit(5)
        .all()
    )

    recent_activity = [
        DashboardRecentActivityItem(
            student_name=str(student_name),
            emotion=str(item.mood) if item.mood else "Unknown",
            time_ago=_relative_time_label(item.saved_at),
            saved_at=item.saved_at,
        )
        for item, student_name in recent_rows
    ]

    return DashboardOverviewResponse(
        total_students=int(total_students),
        total_analyses=int(total_analyses),
        active_classes=int(active_classes),
        analyses_this_week=int(analyses_this_week),
        recent_activity=recent_activity,
    )
