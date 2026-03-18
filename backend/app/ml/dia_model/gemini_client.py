from __future__ import annotations
from dataclasses import dataclass
import logging
import json

logger = logging.getLogger(__name__)


def _extract_balanced_json_object(text: str) -> str:
    """Return the first balanced JSON object from text, or empty string."""
    if not text:
        return ""

    start = text.find("{")
    if start == -1:
        return ""

    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1].strip()

    return ""


def _normalize_response_text(text: str) -> str:
    """Strip wrappers and keep only parseable JSON object when possible."""
    cleaned = (text or "").strip()
    if not cleaned:
        return ""

    # Remove accidental markdown fences.
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()

    obj_text = _extract_balanced_json_object(cleaned)
    if obj_text:
        try:
            json.loads(obj_text)
            return obj_text
        except json.JSONDecodeError:
            pass

    return cleaned

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

        def _call(max_tokens: int, force_json: bool) -> tuple[str, str]:
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
            text = (resp.text or "").strip()
            return _normalize_response_text(text), text

        # First pass: strict JSON with a moderate token budget for reliability.
        try:
            text, raw_text = _call(max_tokens=1400, force_json=True)
            if raw_text and not text:
                logger.warning("Gemini returned content but no JSON object could be extracted on first attempt.")
        except Exception as exc:
            logger.warning("Gemini request failed on first attempt: %s", exc)
            text = ""
            raw_text = ""

        # Single retry only to avoid quota exhaustion.
        if not text:
            logger.warning("Gemini JSON output appears incomplete; retrying once with larger token budget.")
            try:
                text, retry_raw_text = _call(max_tokens=3600, force_json=True)
                if retry_raw_text and not text:
                    logger.warning("Gemini retry returned content but still no valid JSON object could be extracted.")
            except Exception as exc:
                logger.warning("Gemini retry failed: %s", exc)
                text = ""

        return text