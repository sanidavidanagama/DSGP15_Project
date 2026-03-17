from __future__ import annotations
from dataclasses import dataclass
import time
import logging

from app.ml.dia_model.config import RagConfig
from app.ml.dia_model.rag_retriever import RagRetriever
from app.ml.dia_model.gemini_client import GeminiClient
from app.ml.dia_model.utils import read_image_bytes


from app.ml.dia_model.prompts import SYSTEM_PROMPT, json_structure

logger = logging.getLogger(__name__)
MAX_CONTEXT_CHARS = 7000


def _format_context(chunks) -> str:
    if not chunks:
        return ""

    # Defensive check: ensure we received RetrievedChunk objects
    for i, c in enumerate(chunks):
        if not hasattr(c, "text") or not hasattr(c, "source") or not hasattr(c, "page"):
            raise TypeError(f"Retriever returned unexpected item at index {i}: {type(c)} -> {c}")

    lines = []
    for c in chunks:
        lines.append(f"[Source: {c.source} p.{c.page}] {c.text}")
    context = "\n\n".join(lines)
    if len(context) > MAX_CONTEXT_CHARS:
        return context[:MAX_CONTEXT_CHARS] + "\n\n[Context truncated for response reliability.]"
    return context


@dataclass
class DrawingIndicatorAnalyser:
    config: RagConfig

    def __post_init__(self):
        self.retriever = RagRetriever(self.config)
        self.llm = GeminiClient(self.config.llm_model, api_key=self.config.api_key)
        self._cached_context: str | None = None
        self._retrieval_query = (
            "Drawing Indicator Analysis methods for interpreting children's drawings using observable features; "
            "rules for cautious interpretation; linking child text description to features; non-clinical phrasing."
        )

    def run(self, image_path: str, child_description: str) -> str:
        t0 = time.perf_counter()
        self.retriever.build_or_update_index()
        t1 = time.perf_counter()

        if self._cached_context is None:
            chunks = self.retriever.retrieve(query=self._retrieval_query)
            self._cached_context = _format_context(chunks)
        context = self._cached_context
        t2 = time.perf_counter()

        image_bytes, image_mime = read_image_bytes(image_path)
        t3 = time.perf_counter()

        user_prompt = f"""
        ...
        Return EXACTLY one JSON object that matches this JSON structure (same keys, no extra keys):
        {json_structure}

        Retrieved literature context:
        {context}

        Child text description:
        {child_description}

        Output rules:
        - Output JSON only (no markdown, no backticks, no extra text).
        - Use only enumerated values for the categorical fields.
        - Interpretation must be 3–5 short lines (fill unused lines with empty strings if needed).
        """.strip()

        out = self.llm.generate_json(
            system_prompt=SYSTEM_PROMPT.strip(),
            user_prompt=user_prompt,
            image_bytes=image_bytes,
            image_mime=image_mime,
        )
        t4 = time.perf_counter()
        logger.info(
            "DIA run timings index=%.2fs retrieve=%.2fs image=%.2fs llm=%.2fs total=%.2fs",
            (t1 - t0),
            (t2 - t1),
            (t3 - t2),
            (t4 - t3),
            (t4 - t0),
        )
        return out
