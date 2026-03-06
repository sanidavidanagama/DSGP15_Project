from pydantic import BaseModel

class ImageValidationRequest(BaseModel):
    pass

class ImageValidationResponse(BaseModel):
    valid: bool
    message: str
