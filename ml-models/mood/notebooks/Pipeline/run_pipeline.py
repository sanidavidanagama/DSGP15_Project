from emotion_pipeline import EmotionPipeline

BERT_MODEL_DIR = r"C:\Users\USER\PycharmProjects\DSGP15_Project\ml-models\mood\notebooks\DistilBERT\saved_emotion_bert"

RESNET_MODEL_PATH = r"C:\Users\USER\PycharmProjects\DSGP15_Project\ml-models\mood\notebooks\ResNet50\resnet50_emotion_model_cpu_optimized.pth"

FUSION_MODEL_PATH = r"C:\Users\USER\PycharmProjects\DSGP15_Project\ml-models\mood\notebooks\Fusion\fusion_model.pth"


pipeline = EmotionPipeline(
    bert_model_dir=BERT_MODEL_DIR,
    resnet_model_path=RESNET_MODEL_PATH,
    fusion_model_path=FUSION_MODEL_PATH
)

image_path = r"C:\Users\USER\Desktop\test.jpg"
text = "The child looks very happy"

prediction = pipeline.predict(image_path, text)

print("Predicted Emotion:", prediction)
