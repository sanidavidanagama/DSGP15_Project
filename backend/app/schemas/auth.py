from pydantic import BaseModel, EmailStr


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    teacher_id: str | None = None


class UserBase(BaseModel):
    email: EmailStr


class UserCreate(UserBase):
    username: str
    password: str


class UserDB(UserBase):
    id: int
    username: str | None = None
    is_active: bool

    class Config:
        from_attributes = True
