from typing import Optional, List
from pydantic import BaseModel

class UploadJobResponse(BaseModel):
    job_id: str
    message: str

class ImageMetadata(BaseModel):
    processed_image_path: str
    created_at: str
    file_type: str
    size: int
    dimensions: str


class EmotionResult(BaseModel):
    emotion: str
    predicted_mood: Optional[str] = None
    happy_score: Optional[float] = None
    probabilities: Optional[dict[str, float]] = None


class DiaResult(BaseModel):
    line_pressure: str
    shading_intensity: str
    overall_tone: str
    page_usage: str
    figure_size: str
    placement: str
    human_figure_present: str
    missing_body_parts: str
    facial_features: str
    number_of_figures: str
    distance_between_figures: str
    self_positioning: str
    interpretation: List[str]


class DetectedPatterns(BaseModel):
    emotional: str
    spatial: str


class RecommendationResult(BaseModel):
    DetectedPatterns: DetectedPatterns
    RecommendationCategory: str
    RecommendationText: str


class ResultBundle(BaseModel):
    image: Optional[ImageMetadata] = None
    emotion: Optional[EmotionResult] = None
    dia: Optional[DiaResult] = None
    recommendation: Optional[RecommendationResult] = None


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    raw_image_path: Optional[str] = None
    result: Optional[ResultBundle] = None