from sqlalchemy.orm import Session

from app.models.user import User


def get_user_by_email(db: Session, email: str) -> User | None:
    return db.query(User).filter(User.email == email).first()


def get_user_by_username(db: Session, username: str) -> User | None:
    return db.query(User).filter(User.username == username).first()


def get_user_by_identifier(db: Session, identifier: str) -> User | None:
    return db.query(User).filter((User.email == identifier) | (User.username == identifier)).first()


def create_user(db: Session, email: str, hashed_password: str, username: str | None = None) -> User:
    db_user = User(email=email, hashed_password=hashed_password, username=username)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user
