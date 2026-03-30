from typing import List
from datetime import timezone
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database.crud_class import get_class_by_id
from app.services.auth_service import get_current_teacher
from app.database.crud_student import (
    create_student,
    get_student_by_id,
    get_students_by_class,
    soft_delete_student,
    update_student,
)
from app.database.crud_student_saved_analysis import (
    create_saved_analysis,
    list_saved_analyses,
    list_saved_analyses_with_job_context,
)
from app.database.crud_job import get_job_by_job_id
from app.database.database import SessionLocal
from app.schemas.student import (
    StudentCreate,
    StudentDetailResponse,
    StudentResponse,
    StudentUpdate,
    StudentWithStatsResponse,
)
from app.schemas.student_saved_analysis import StudentSavedAnalysisCreate, StudentSavedAnalysisResponse
from app.schemas.job import JobStatusResponse



router = APIRouter(tags=["Students"])

BACKEND_ROOT = Path(__file__).resolve().parents[2]


def _to_absolute_path(path_value: str | None) -> str | None:
    if not path_value:
        return None

    path_obj = Path(path_value)
    if path_obj.is_absolute():
        return str(path_obj)

    return str((BACKEND_ROOT / path_obj).resolve())


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_teacher_id(teacher_id: str = Depends(get_current_teacher)) -> str:
    return teacher_id


def resolve_class(class_id: int, teacher_id: str, db: Session):
    classroom = get_class_by_id(db, class_id, teacher_id)
    if not classroom:
        raise HTTPException(status_code=404, detail="Class not found")
    return classroom


def resolve_student_with_ownership(student_id: int, teacher_id: str, db: Session):
    student = get_student_by_id(db, student_id)
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    resolve_class(student.class_id, teacher_id, db)
    return student


@router.post("/classes/{class_id}/students", response_model=StudentResponse, status_code=201)
def add_student(
    class_id: int,
    payload: StudentCreate,
    teacher_id: str = Depends(get_teacher_id),
    db: Session = Depends(get_db),
):
    resolve_class(class_id, teacher_id, db)
    return create_student(db, class_id, payload)


@router.get("/classes/{class_id}/students", response_model=List[StudentResponse])
def list_students(
    class_id: int,
    teacher_id: str = Depends(get_teacher_id),
    db: Session = Depends(get_db),
):
    resolve_class(class_id, teacher_id, db)
    return get_students_by_class(db, class_id)


@router.get("/students/{student_id}", response_model=StudentResponse)
def get_student(
    student_id: int,
    teacher_id: str = Depends(get_teacher_id),
    db: Session = Depends(get_db),
):
    return resolve_student_with_ownership(student_id, teacher_id, db)


@router.get("/classes/{class_id}/students/{student_id}", response_model=StudentDetailResponse)
def get_student_in_class(
    class_id: int,
    student_id: int,
    teacher_id: str = Depends(get_teacher_id),
    db: Session = Depends(get_db),
):
    classroom = resolve_class(class_id, teacher_id, db)

    student = get_student_by_id(db, student_id)
    if not student or student.class_id != classroom.id:
        raise HTTPException(status_code=404, detail="Student not found")

    history = list_saved_analyses_with_job_context(db, student_id)
    total_analyses = len(history)

    last_mood = None
    last_saved_at = None
    if history:
        latest = history[0]
        last_mood = latest.get("mood")
        last_saved_at = latest.get("saved_at")

    return StudentDetailResponse(
        id=student.id,
        class_id=student.class_id,
        name=student.name,
        gender=student.gender,
        joined_at=student.joined_at,
        last_predicted_mood=last_mood,
        last_predicted_at=last_saved_at,
        total_analyses=total_analyses,
        history=history,
    )


@router.patch("/students/{student_id}", response_model=StudentResponse)
def edit_student(
    student_id: int,
    payload: StudentUpdate,
    teacher_id: str = Depends(get_teacher_id),
    db: Session = Depends(get_db),
):
    resolve_student_with_ownership(student_id, teacher_id, db)
    return update_student(db, student_id, payload)


@router.delete("/students/{student_id}")
def delete_student(
    student_id: int,
    teacher_id: str = Depends(get_teacher_id),
    db: Session = Depends(get_db),
):
    resolve_student_with_ownership(student_id, teacher_id, db)
    soft_delete_student(db, student_id)
    return {"message": "Student deleted"}


@router.post("/students/{student_id}/saved-analyses", response_model=StudentSavedAnalysisResponse, status_code=201)
def save_analysis_for_student(
    student_id: int,
    payload: StudentSavedAnalysisCreate,
    teacher_id: str = Depends(get_teacher_id),
    db: Session = Depends(get_db),
):
    resolve_student_with_ownership(student_id, teacher_id, db)

    job = get_job_by_job_id(db, payload.job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status != "done" or not isinstance(job.result, dict):
        raise HTTPException(status_code=400, detail="Job is not ready to be saved")

    return create_saved_analysis(db, student_id, job)


@router.get("/students/{student_id}/saved-analyses", response_model=List[StudentSavedAnalysisResponse])
def list_saved_analyses_for_student(
    student_id: int,
    teacher_id: str = Depends(get_teacher_id),
    db: Session = Depends(get_db),
):
    resolve_student_with_ownership(student_id, teacher_id, db)
    return list_saved_analyses_with_job_context(db, student_id)


@router.get(
    "/classes/{class_id}/students/{student_id}/report/{job_id}",
    response_model=JobStatusResponse,
)
def get_saved_report_for_student(
    class_id: int,
    student_id: int,
    job_id: str,
    teacher_id: str = Depends(get_teacher_id),
    db: Session = Depends(get_db),
):
    classroom = resolve_class(class_id, teacher_id, db)

    student = get_student_by_id(db, student_id)
    if not student or student.class_id != classroom.id:
        raise HTTPException(status_code=404, detail="Student not found")

    history = list_saved_analyses_with_job_context(db, student_id)
    if not any(item.get("job_id") == job_id for item in history):
        raise HTTPException(status_code=404, detail="Report not found for this student")

    job = get_job_by_job_id(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    result = job.result or None

    if isinstance(result, dict):
        image_result = result.get("image")
        if isinstance(image_result, dict):
            image_result["processed_image_path"] = _to_absolute_path(
                image_result.get("processed_image_path")
            )

    started_at = job.created_at
    finished_at = job.updated_at if job.status == "done" else None

    def _to_iso(value):
        if value is None:
            return None
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat()

    analysis_duration_seconds = None
    if started_at is not None and finished_at is not None:
        analysis_duration_seconds = max(0.0, (finished_at - started_at).total_seconds())

    return {
        "job_id": job.job_id,
        "status": job.status,
        "raw_image_path": _to_absolute_path(job.image_path),
        "analysis_started_at": _to_iso(started_at),
        "analysis_finished_at": _to_iso(finished_at),
        "analysis_duration_seconds": analysis_duration_seconds,
        "result": result,
    }
