from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from starlette import status
from sqlalchemy.orm import Session

from app.database.crud_auth import create_user, get_user_by_email, get_user_by_identifier, get_user_by_username
from app.database.database import SessionLocal
from app.schemas.auth import Token, TokenData, UserCreate, UserDB
from app.services.auth_service import (
    create_access_token,
    get_current_teacher,
    get_password_hash,
    verify_password,
)

router = APIRouter(prefix="/auth", tags=["Auth"])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.post("/register", response_model=UserDB)
def register_user(payload: UserCreate, db: Session = Depends(get_db)):
    username = payload.username.strip()
    if not username:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username is required")

    existing = get_user_by_email(db, payload.email)
    if existing:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered")

    existing_username = get_user_by_username(db, username)
    if existing_username:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username already registered")

    hashed_password = get_password_hash(payload.password)
    user = create_user(db, email=payload.email, hashed_password=hashed_password, username=username)
    return user


@router.post("/token", response_model=Token)
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    identifier = form_data.username.strip()
    user = get_user_by_identifier(db, identifier)
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email/username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(data={"sub": user.email})
    return {"access_token": access_token, "token_type": "bearer"}


@router.get("/me")
def read_current_teacher(teacher_id: str = Depends(get_current_teacher), db: Session = Depends(get_db)):
    user = get_user_by_email(db, teacher_id)
    return {
        "teacher_id": teacher_id,
        "email": teacher_id,
        "username": user.username if user else None,
    }
