from __future__ import annotations
from dataclasses import dataclass
import re
from typing import List
from app.ml.dia_model.pdf_loader import DocChunk

@dataclass
class SplitChunk:
    text: str
    source: str
    page: int
    chunk_id: str

class SimpleTextSplitter:
    """Header-aware splitter that keeps structured sections intact when possible."""

    _SECTION_HEADER_RE = re.compile(
        r"^(?:PART\s+\d+\b.*|\d+(?:\.\d+)+\s+.+)$",
        re.IGNORECASE,
    )

    def __init__(self, chunk_size: int = 900, overlap: int = 150):
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, docs: List[DocChunk]) -> List[SplitChunk]:
        out: List[SplitChunk] = []
        for d in docs:
            raw = self._normalize_document_text(d.text)
            if not raw:
                continue

            idx = 0
            sections = self._split_into_sections(raw)
            for section in sections:
                for chunk_text in self._chunk_section(section):
                    out.append(
                        SplitChunk(
                            text=chunk_text,
                            source=d.source,
                            page=d.page,
                            chunk_id=f"{d.source}:p{d.page}:c{idx}",
                        )
                    )
                    idx += 1
        return out

    def _normalize_document_text(self, text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
        normalized = "\n".join(lines)
        normalized = re.sub(r"\n{3,}", "\n\n", normalized)
        return normalized.strip()

    def _is_section_header(self, line: str) -> bool:
        if self._SECTION_HEADER_RE.match(line):
            return True

        # Heuristic for short all-caps section labels.
        if len(line) <= 100 and line.upper() == line and any(ch.isalpha() for ch in line):
            return True

        return False

    def _split_into_sections(self, text: str) -> List[str]:
        lines = text.split("\n")
        sections: List[str] = []
        current: List[str] = []

        for raw_line in lines:
            line = raw_line.strip()

            if not line:
                if current and current[-1] != "":
                    current.append("")
                continue

            if self._is_section_header(line) and current:
                section_text = "\n".join(current).strip()
                if section_text:
                    sections.append(section_text)
                current = [line]
                continue

            current.append(line)

        tail = "\n".join(current).strip()
        if tail:
            sections.append(tail)

        return sections or [text]

    def _chunk_section(self, section_text: str) -> List[str]:
        if len(section_text) <= self.chunk_size:
            return [section_text]

        paragraphs = [p.strip() for p in re.split(r"\n{2,}", section_text) if p.strip()]
        if len(paragraphs) <= 1:
            return self._split_long_text(section_text)

        chunks: List[str] = []
        current = ""
        for para in paragraphs:
            candidate = para if not current else f"{current}\n\n{para}"
            if len(candidate) <= self.chunk_size:
                current = candidate
                continue

            if current:
                chunks.extend(self._enforce_chunk_limit(current))
            current = para

        if current:
            chunks.extend(self._enforce_chunk_limit(current))

        return chunks

    def _enforce_chunk_limit(self, text: str) -> List[str]:
        text = text.strip()
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]
        return self._split_long_text(text)

    def _split_long_text(self, text: str) -> List[str]:
        compact = re.sub(r"\s+", " ", text).strip()
        if len(compact) <= self.chunk_size:
            return [compact]

        out: List[str] = []
        start = 0
        while start < len(compact):
            end = min(len(compact), start + self.chunk_size)
            chunk_text = compact[start:end].strip()
            if chunk_text:
                out.append(chunk_text)
            if end == len(compact):
                break
            start = max(0, end - self.overlap)

        return out