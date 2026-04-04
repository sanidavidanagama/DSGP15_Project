from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import select

from app.database.database import SessionLocal
from app.database.crud_auth import get_user_by_email, get_user_by_username
from app.models.class_model import Classroom
from app.models.student import Student
from app.models.student_saved_analysis import StudentSavedAnalysis
from app.schemas.settings import (
    ChangePasswordRequest,
    DeleteDataRequest,
    DeleteDataResponse,
    MessageResponse,
    SettingsProfileResponse,
    UpdateProfileRequest,
)
from app.services.auth_service import get_current_teacher, get_password_hash, verify_password

router = APIRouter(prefix="/settings", tags=["Settings"])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_current_user(teacher_id: str = Depends(get_current_teacher), db: Session = Depends(get_db)):
    user = get_user_by_email(db, teacher_id)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user


def _delete_teacher_owned_data(db: Session, teacher_email: str) -> DeleteDataResponse:
    class_ids = db.scalars(select(Classroom.id).where(Classroom.teacher_id == teacher_email)).all()

    if not class_ids:
        return DeleteDataResponse(deleted_classes=0, deleted_students=0, deleted_saved_analyses=0)

    student_ids = db.scalars(select(Student.id).where(Student.class_id.in_(class_ids))).all()

    deleted_saved_analyses = 0
    if student_ids:
        deleted_saved_analyses = (
            db.query(StudentSavedAnalysis)
            .filter(StudentSavedAnalysis.student_id.in_(student_ids))
            .delete(synchronize_session=False)
        )

    deleted_students = (
        db.query(Student)
        .filter(Student.class_id.in_(class_ids))
        .delete(synchronize_session=False)
    )

    deleted_classes = (
        db.query(Classroom)
        .filter(Classroom.id.in_(class_ids))
        .delete(synchronize_session=False)
    )

    return DeleteDataResponse(
        deleted_classes=int(deleted_classes),
        deleted_students=int(deleted_students),
        deleted_saved_analyses=int(deleted_saved_analyses),
    )


@router.get("/me", response_model=SettingsProfileResponse)
def get_my_settings_profile(user=Depends(get_current_user)):
    return SettingsProfileResponse(email=user.email, username=user.username)


@router.patch("/profile", response_model=SettingsProfileResponse)
def update_my_profile(payload: UpdateProfileRequest, user=Depends(get_current_user), db: Session = Depends(get_db)):
    username = payload.username.strip()
    if not username:
        raise HTTPException(status_code=400, detail="Username is required")

    existing = get_user_by_username(db, username)
    if existing and existing.id != user.id:
        raise HTTPException(status_code=400, detail="Username already in use")

    user.username = username
    db.commit()
    db.refresh(user)

    return SettingsProfileResponse(email=user.email, username=user.username)


@router.patch("/password", response_model=MessageResponse)
def change_my_password(payload: ChangePasswordRequest, user=Depends(get_current_user), db: Session = Depends(get_db)):
    if not verify_password(payload.current_password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Current password is incorrect")

    if payload.current_password == payload.new_password:
        raise HTTPException(status_code=400, detail="New password must be different from the current password")

    user.hashed_password = get_password_hash(payload.new_password)
    db.commit()

    return MessageResponse(message="Password updated successfully")


@router.delete("/data", response_model=DeleteDataResponse)
def delete_my_data(payload: DeleteDataRequest, user=Depends(get_current_user), db: Session = Depends(get_db)):
    if not verify_password(payload.current_password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Current password is incorrect")

    try:
        result = _delete_teacher_owned_data(db, user.email)
        db.commit()
        return result
    except Exception:
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to delete data")


@router.delete("/account", response_model=MessageResponse)
def delete_my_account(payload: DeleteDataRequest, user=Depends(get_current_user), db: Session = Depends(get_db)):
    if not verify_password(payload.current_password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Current password is incorrect")

    try:
        _delete_teacher_owned_data(db, user.email)
        db.delete(user)
        db.commit()
        return MessageResponse(message="Account and all associated data deleted successfully")
    except Exception:
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to delete account")
