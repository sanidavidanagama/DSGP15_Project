from pydantic import BaseModel, Field


class SettingsProfileResponse(BaseModel):
    email: str
    username: str | None = None


class UpdateProfileRequest(BaseModel):
    username: str = Field(min_length=1, max_length=100)


class ChangePasswordRequest(BaseModel):
    current_password: str = Field(min_length=1)
    new_password: str = Field(min_length=8, max_length=256)


class DeleteDataRequest(BaseModel):
    current_password: str = Field(min_length=1)


class DeleteDataResponse(BaseModel):
    deleted_classes: int
    deleted_students: int
    deleted_saved_analyses: int


class MessageResponse(BaseModel):
    message: str
