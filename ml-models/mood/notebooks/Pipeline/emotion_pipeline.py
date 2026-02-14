import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
from transformers import DistilBertTokenizer, DistilBertModel


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

        # -----------------------
        # Load BERT
        # -----------------------
        self.tokenizer = DistilBertTokenizer.from_pretrained(bert_model_dir)
        self.bert = DistilBertModel.from_pretrained(bert_model_dir)
        self.bert.to(self.device)
        self.bert.eval()

        # -----------------------
        # Load ResNet
        # -----------------------
        self.resnet = models.resnet50(pretrained=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 2)

        self.resnet.load_state_dict(
            torch.load(resnet_model_path, map_location=self.device)
        )

        # Remove classifier for feature extraction
        self.resnet.fc = nn.Identity()
        self.resnet.to(self.device)
        self.resnet.eval()

        # -----------------------
        # Load Fusion Model
        # -----------------------
        self.fusion_model = nn.Sequential(
            nn.Linear(2048 + 768, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2)
        )

        self.fusion_model.load_state_dict(
            torch.load(fusion_model_path, map_location=self.device)
        )

        self.fusion_model.to(self.device)
        self.fusion_model.eval()

        # -----------------------
        # Image Transform
        # -----------------------
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])

        self.label_map = {0: "happy", 1: "sad"}

    # -----------------------
    # Text Feature Extraction
    # -----------------------
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

        return outputs.last_hidden_state[:, 0, :]

    # -----------------------
    # Image Feature Extraction
    # -----------------------
    def extract_image_features(self, image_path):

        image = Image.open(image_path).convert("RGB")
        image = self.image_transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.resnet(image)

        return features

    # -----------------------
    # Predict
    # -----------------------
    def predict(self, image_path, text):

        img_feat = self.extract_image_features(image_path)
        txt_feat = self.extract_text_features(text)

        fused = torch.cat((img_feat, txt_feat), dim=1)

        with torch.no_grad():
            output = self.fusion_model(fused)
            pred_idx = torch.argmax(output, dim=1).item()

        return self.label_map[pred_idx]
