from __future__ import annotations
from dataclasses import dataclass
from typing import List
from pdf_loader import DocChunk
from utils import normalize_whitespaces

@dataclass
class SplitChunk:
    text: str
    source: str
    page: int
    chunk_id: str

class SimpleTextSplitter:
    def __init__(self, chunk_size: int = 900, overlap: int = 150):
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, docs: List[DocChunk]) -> List[SplitChunk]:
        out: List[SplitChunk] = []
        for d in docs:
            raw = normalize_whitespaces(d.text)
            start = 0
            idx = 0
            while start < len(raw):
                end = min(len(raw), start + self.chunk_size)
                chunk_text = raw[start:end].strip()
                if chunk_text:
                    out.append(
                        SplitChunk(
                            text=chunk_text,
                            source=d.source,
                            page=d.page,
                            chunk_id=f"{d.source}:p{d.page}:c{idx}",
                        )
                    )
                idx += 1
                if end == len(raw):
                    break
                start = max(0, end - self.overlap)
        return out