from typing import List, Optional

from sqlalchemy.orm import Session

from app.models.class_model import Classroom
from app.schemas.class_schema import ClassCreate, ClassUpdate


def create_class(db: Session, teacher_id: str, payload: ClassCreate) -> Classroom:
    classroom = Classroom(
        teacher_id=teacher_id,
        class_name=payload.class_name,
        grade_age_group=payload.grade_age_group,
        schedule_days=payload.schedule_days,
        description=payload.description,
    )
    db.add(classroom)
    db.commit()
    db.refresh(classroom)
    return classroom


def get_classes(db: Session, teacher_id: str) -> List[Classroom]:
    return (
        db.query(Classroom)
        .filter(Classroom.teacher_id == teacher_id, Classroom.is_deleted == False)
        .order_by(Classroom.id.desc())
        .all()
    )


def get_class_by_id(db: Session, class_id: int, teacher_id: str) -> Optional[Classroom]:
    return (
        db.query(Classroom)
        .filter(
            Classroom.id == class_id,
            Classroom.teacher_id == teacher_id,
            Classroom.is_deleted == False,
        )
        .first()
    )


def update_class(
    db: Session,
    class_id: int,
    teacher_id: str,
    payload: ClassUpdate,
) -> Optional[Classroom]:
    classroom = get_class_by_id(db, class_id, teacher_id)
    if not classroom:
        return None

    update_data = payload.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(classroom, key, value)

    db.commit()
    db.refresh(classroom)
    return classroom


def soft_delete_class(db: Session, class_id: int, teacher_id: str) -> bool:
    classroom = get_class_by_id(db, class_id, teacher_id)
    if not classroom:
        return False

    classroom.is_deleted = True
    db.commit()
    return True
