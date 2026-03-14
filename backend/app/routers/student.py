from typing import List

from fastapi import APIRouter, Depends, Header, HTTPException
from sqlalchemy.orm import Session

from app.database.crud_class import get_class_by_id
from app.database.crud_student import (
    create_student,
    get_student_by_id,
    get_students_by_class,
    soft_delete_student,
    update_student,
)
from app.database.database import SessionLocal
from app.schemas.student import StudentCreate, StudentResponse, StudentUpdate



router = APIRouter(tags=["Students"])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_teacher_id(x_teacher_id: str = Header(default="dev-teacher", alias="X-Teacher-Id")) -> str:
    return x_teacher_id


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
