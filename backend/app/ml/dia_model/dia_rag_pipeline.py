from __future__ import annotations
from dataclasses import dataclass
from typing import List
import os
from pathlib import Path
from dotenv import load_dotenv
from config import RagConfig
from rag_retriever import RagRetriever
from gemini_client import GeminiClient
from utils import read_image_bytes


from prompts import SYSTEM_PROMPT, json_structure


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
    return "\n\n".join(lines)



class DIARagPipeline:
    def __init__(self, config: RagConfig, api_key: str) -> None:
        self.config = config
        self.retriever = RagRetriever(self.config)
        self.llm = GeminiClient(self.config.llm_model, api_key=api_key)

    def run(self, image_path: str, child_description: str) -> str:
        self.retriever.build_or_update_index()

        query = (
            "Drawing Indicator Analysis methods for interpreting children's drawings using observable features; "
            "rules for cautious interpretation; linking child text description to features; non-clinical phrasing."
        )
        chunks = self.retriever.retrieve(query=query)
        context = _format_context(chunks)

        image_bytes, image_mime = read_image_bytes(image_path)

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

        return self.llm.generate_json(
            system_prompt=SYSTEM_PROMPT.strip(),
            user_prompt=user_prompt,
            image_bytes=image_bytes,
            image_mime=image_mime,
        )
