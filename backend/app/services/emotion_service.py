from app.ml.mood_model.emotion_pipeline import EmotionPipeline
from app.core.config import settings
from pathlib import Path

def run_emotion_pipeline(image_path: str, text: str) -> dict:
    """
    Runs the emotion pipeline on the given image and text.
    Returns a dict with thresholded mood and class probabilities.
    """
    pipeline = EmotionPipeline(
        bert_model_dir=Path(settings.EMOTION_BERT_MODEL_DIR),
        resnet_model_path=Path(settings.EMOTION_RESNET_MODEL_PATH),
        fusion_model_path=Path(settings.EMOTION_FUSION_MODEL_PATH)
    )
    return pipeline.predict_with_thresholds(image_path, text)
