from __future__ import annotations
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class GeminiClient:
    model_name: str
    api_key: str

    def __post_init__(self) -> None:
        try:
            from google import genai
        except Exception as e:
            raise ImportError(
                "Missing dependency 'google-genai'. Install it with: pip install google-genai"
            ) from e

        self._genai = genai
        self._client = genai.Client(api_key=self.api_key)

    def generate_json(self, system_prompt: str, user_prompt: str, image_bytes: bytes, image_mime: str) -> str:
        compact_user_prompt = (
            user_prompt
            + "\n\nReturn compact JSON in one object only. Do not add extra keys, comments, or markdown."
        )
        parts = [
            system_prompt,
            self._genai.types.Part.from_bytes(data=image_bytes, mime_type=image_mime),
            compact_user_prompt,
        ]

        def _call(max_tokens: int, force_json: bool) -> str:
            config = self._genai.types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=max_tokens,
                response_mime_type="application/json" if force_json else None,
            )
            resp = self._client.models.generate_content(
                model=self.model_name,
                contents=parts,
                config=config,
            )
            return (resp.text or "").strip()

        # First pass: strict JSON, bounded tokens for speed.
        try:
            text = _call(max_tokens=900, force_json=True)
        except Exception as exc:
            logger.warning("Gemini request failed on first attempt: %s", exc)
            text = ""

        # Single retry only to avoid quota exhaustion.
        if not text or not text.endswith("}"):
            logger.warning("Gemini JSON output appears incomplete; retrying once with larger token budget.")
            try:
                text = _call(max_tokens=1600, force_json=True)
            except Exception as exc:
                logger.warning("Gemini retry failed: %s", exc)
                text = ""

        return text