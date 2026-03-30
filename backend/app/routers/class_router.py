from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database.crud_class import (
    create_class,
    get_class_by_id,
    get_classes,
    soft_delete_class,
    update_class,
)
from app.database.crud_student import get_students_by_class
from app.services.auth_service import get_current_teacher
from app.database.database import SessionLocal
from app.schemas.class_schema import ClassCreate, ClassResponse, ClassUpdate, ClassWithStatsResponse

router = APIRouter(prefix="/classes", tags=["Classes"])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_teacher_id(teacher_id: str = Depends(get_current_teacher)) -> str:
    return teacher_id


@router.post("", response_model=ClassResponse, status_code=201)
def create_class_route(
    payload: ClassCreate,
    teacher_id: str = Depends(get_teacher_id),
    db: Session = Depends(get_db),
):
    return create_class(db, teacher_id, payload)


@router.get("", response_model=List[ClassWithStatsResponse])
def get_classes_route(
    teacher_id: str = Depends(get_teacher_id),
    db: Session = Depends(get_db),
):
    classrooms = get_classes(db, teacher_id)

    classes_with_stats: List[ClassWithStatsResponse] = []
    for classroom in classrooms:
        student_count = len(get_students_by_class(db, classroom.id))

        classes_with_stats.append(
            ClassWithStatsResponse(
                id=classroom.id,
                teacher_id=classroom.teacher_id,
                class_name=classroom.class_name,
                grade_age_group=classroom.grade_age_group,
                schedule_days=classroom.schedule_days,
                description=classroom.description,
                created_at=classroom.created_at,
                updated_at=classroom.updated_at,
                student_count=student_count,
            )
        )

    return classes_with_stats


@router.get("/{class_id}", response_model=ClassResponse)
def get_class_route(
    class_id: int,
    teacher_id: str = Depends(get_teacher_id),
    db: Session = Depends(get_db),
):
    classroom = get_class_by_id(db, class_id, teacher_id)
    if not classroom:
        raise HTTPException(status_code=404, detail="Class not found")
    return classroom


@router.patch("/{class_id}", response_model=ClassResponse)
def update_class_route(
    class_id: int,
    payload: ClassUpdate,
    teacher_id: str = Depends(get_teacher_id),
    db: Session = Depends(get_db),
):
    classroom = update_class(db, class_id, teacher_id, payload)
    if not classroom:
        raise HTTPException(status_code=404, detail="Class not found")
    return classroom


@router.delete("/{class_id}")
def delete_class_route(
    class_id: int,
    teacher_id: str = Depends(get_teacher_id),
    db: Session = Depends(get_db),
):
    deleted = soft_delete_class(db, class_id, teacher_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Class not found")
    return {"message": "Class deleted"}
