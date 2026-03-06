from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

@dataclass
class GeminiClient:
    model_name: str

    def __post_init__(self) -> None:
        import os
        try:
            from google import genai
        except Exception as e:
            raise ImportError(
                "Missing dependency 'google-genai'. Install it with: pip install google-genai"
            ) from e

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set in environment.")
        self._genai = genai
        self._client = genai.Client(api_key=api_key)

    def generate_json(self, system_prompt: str, user_prompt: str, image_bytes: bytes, image_mime: str) -> str:
        resp = self._client.models.generate_content(
            model=self.model_name,
            contents=[
                system_prompt,
                self._genai.types.Part.from_bytes(data=image_bytes, mime_type=image_mime),
                user_prompt,
            ],
        )

        text = (resp.text or "").strip()
        return text
