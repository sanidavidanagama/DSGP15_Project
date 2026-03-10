import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from transformers import DistilBertTokenizer, DistilBertModel


class MultimodalEmotionClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(2048 + 768, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2)
        )

    def forward(self, image_feat, text_feat):
        fused = torch.cat((image_feat, text_feat), dim=1)
        return self.fc(fused)


class EmotionPipeline:

    def __init__(self,
                 bert_model_dir,
                 resnet_model_path,
                 fusion_model_path,
                 device=None):

        self.device = device if device else (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        print("Using device:", self.device)

        # -------------------------
        # Load DistilBERT
        # -------------------------
        self.tokenizer = DistilBertTokenizer.from_pretrained(bert_model_dir)
        self.bert = DistilBertModel.from_pretrained(bert_model_dir)
        self.bert.to(self.device)
        self.bert.eval()

        # -------------------------
        # Load ResNet50
        # -------------------------
        self.resnet = models.resnet50(weights=None)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 2)

        self.resnet.load_state_dict(
            torch.load(resnet_model_path, map_location=self.device)
        )

        # Remove classifier for feature extraction
        self.resnet.fc = nn.Identity()
        self.resnet.to(self.device)
        self.resnet.eval()

        # -------------------------
        # Load Fusion Model
        # -------------------------
        self.fusion_model = MultimodalEmotionClassifier()
        self.fusion_model.load_state_dict(
            torch.load(fusion_model_path, map_location=self.device)
        )
        self.fusion_model.to(self.device)
        self.fusion_model.eval()

        # -------------------------
        # Image Transform
        # -------------------------
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])

        self.label_map = {0: "happy", 1: "sad"}

    @staticmethod
    def _score_to_mood(happy_score: float) -> str:
        # Thresholds are based on a 0-100 happy score.
        if happy_score < 40.0:
            return "sad"
        if happy_score < 60.0:
            return "neutral"
        return "happy"

    # -------------------------
    # Extract Text Features
    # -------------------------
    def extract_text_features(self, text):

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=64
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.bert(**inputs)

        return outputs.last_hidden_state[:, 0, :]  # CLS token

    # -------------------------
    # Extract Image Features
    # -------------------------
    def extract_image_features(self, image_path):

        image = Image.open(image_path).convert("RGB")
        image = self.image_transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.resnet(image)

        return features

    # -------------------------
    # Predict Emotion
    # -------------------------
    def predict(self, image_path, text):

        img_feat = self.extract_image_features(image_path)
        txt_feat = self.extract_text_features(text)

        with torch.no_grad():
            output = self.fusion_model(img_feat, txt_feat)
            pred_idx = torch.argmax(output, dim=1).item()

        return self.label_map[pred_idx]

    def predict_proba(self, image_path, text):

        img_feat = self.extract_image_features(image_path)
        txt_feat = self.extract_text_features(text)

        with torch.no_grad():
            logits = self.fusion_model(img_feat, txt_feat)
            probs = torch.softmax(logits, dim=1).squeeze(0)

        return {
            "happy": float(probs[0].item()),
            "sad": float(probs[1].item())
        }

    def predict_with_thresholds(self, image_path, text):
        class_probabilities = self.predict_proba(image_path, text)
        happy_probability = class_probabilities["happy"]
        happy_score = happy_probability * 100.0
        mood = self._score_to_mood(happy_score)

        return {
            "emotion": mood,
            "predicted_mood": mood,
            "happy_score": round(happy_score, 2),
            "probabilities": {
                "happy": round(class_probabilities["happy"], 4),
                "sad": round(class_probabilities["sad"], 4)
            }
        }
