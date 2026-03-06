from typing import List, Optional
from pydantic import BaseModel

class ImageResult(BaseModel):
    raw: str  # file path or URL
    processed: str  # file path or URL
    caption: str

class MoodResult(BaseModel):
    mood: str
    mood_probability: float

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
    intepretaton: List[str]

class DetectedPatterns(BaseModel):
    emotional: str
    spatial: str
    structural: str

class RecommendationsResult(BaseModel):
    detected_patterns: DetectedPatterns
    recommendation_category: str
    recommendation_text: str

class JobResult(BaseModel):
    image: ImageResult
    mood: MoodResult
    dia: DiaResult
    recommendations: RecommendationsResult
