from typing import List

from fastapi import APIRouter, Depends, Header, HTTPException
from sqlalchemy.orm import Session

from app.database.crud_class import (
    create_class,
    get_class_by_id,
    get_classes,
    soft_delete_class,
    update_class,
)
from app.database.database import SessionLocal
from app.schemas.class_schema import ClassCreate, ClassResponse, ClassUpdate

router = APIRouter(prefix="/classes", tags=["Classes"])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_teacher_id(x_teacher_id: str = Header(default="dev-teacher", alias="X-Teacher-Id")) -> str:
    return x_teacher_id


@router.post("", response_model=ClassResponse, status_code=201)
def create_class_route(
    payload: ClassCreate,
    teacher_id: str = Depends(get_teacher_id),
    db: Session = Depends(get_db),
):
    return create_class(db, teacher_id, payload)


@router.get("", response_model=List[ClassResponse])
def get_classes_route(
    teacher_id: str = Depends(get_teacher_id),
    db: Session = Depends(get_db),
):
    return get_classes(db, teacher_id)


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
