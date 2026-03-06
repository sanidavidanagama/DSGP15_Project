from emotion_pipeline import EmotionPipeline
from pathlib import Path

# Current script folder
CURRENT_DIR = Path(__file__).parent

# Paths relative to the current file
BERT_MODEL_DIR = CURRENT_DIR / "data" / "saved_emotion_bert"
RESNET_MODEL_PATH = CURRENT_DIR / "data" / "resnet50_emotion_model_cpu_optimized.pth"
FUSION_MODEL_PATH = CURRENT_DIR / "data" / "fusion_model.pth"
IMAGE_PATH = Path(__file__).parent.parent.parent.parent / "dataset" / "Dataset" / "Images" / "Emotion" / "test" / "example1.jpg"

# Initialize pipeline
pipeline = EmotionPipeline(
    bert_model_dir=BERT_MODEL_DIR,
    resnet_model_path=RESNET_MODEL_PATH,
    fusion_model_path=FUSION_MODEL_PATH
)

text = "The child is smiling"
prediction = pipeline.predict(IMAGE_PATH, text)
print("Predicted Emotion:", prediction)