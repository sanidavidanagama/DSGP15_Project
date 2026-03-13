from typing import List, Optional

from sqlalchemy.orm import Session

from app.models.student import Student
from backend.app.schemas.student import StudentCreate, StudentUpdate


def create_student(db: Session, class_id: int, data: StudentCreate) -> Student:
    student = Student(class_id=class_id, name=data.name, age_group=data.age_group)
    db.add(student)
    db.commit()
    db.refresh(student)
    return student


def get_students_by_class(db: Session, class_id: int) -> List[Student]:
    return (
        db.query(Student)
        .filter(Student.class_id == class_id, Student.is_deleted == False)
        .all()
    )


def get_student_by_id(db: Session, student_id: int) -> Optional[Student]:
    return (
        db.query(Student)
        .filter(Student.id == student_id, Student.is_deleted == False)
        .first()
    )


def update_student(db: Session, student_id: int, data: StudentUpdate) -> Optional[Student]:
    student = get_student_by_id(db, student_id)
    if not student:
        return None
    for field, value in data.model_dump(exclude_unset=True).items():
        setattr(student, field, value)
    db.commit()
    db.refresh(student)
    return student


def soft_delete_student(db: Session, student_id: int) -> bool:
    student = get_student_by_id(db, student_id)
    if not student:
        return False
    student.is_deleted = True
    db.commit()
    return True
