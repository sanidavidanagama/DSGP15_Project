from app.ml.mood_model.emotion_pipeline import EmotionPipeline
from app.core.config import settings
from pathlib import Path

def run_emotion_pipeline(image_path: str, text: str) -> dict:
    """
    Runs the emotion pipeline on the given image and text.
    Returns a dict with the prediction and any other relevant info.
    """
    pipeline = EmotionPipeline(
        bert_model_dir=Path(settings.EMOTION_BERT_MODEL_DIR),
        resnet_model_path=Path(settings.EMOTION_RESNET_MODEL_PATH),
        fusion_model_path=Path(settings.EMOTION_FUSION_MODEL_PATH)
    )
    prediction = pipeline.predict(image_path, text)
    return {"emotion": prediction}
