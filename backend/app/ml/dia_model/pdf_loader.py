from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List

@dataclass
class DocChunk:
    text: str
    source: str
    page: int

def load_pdfs_from_folder(pdf_dir: Path) -> List[DocChunk]:
    if not pdf_dir.exists():
        raise FileNotFoundError(f"PDF folder not found: {pdf_dir}")

    chunks: List[DocChunk] = []
    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in: {pdf_dir}")

    try:
        from pypdf import PdfReader
    except Exception as e:
        raise ImportError(
            "Missing dependency 'pypdf'. Install it with: uv add pypdf"
        ) from e

    for pdf_path in pdf_files:
        reader = PdfReader(str(pdf_path))
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            text = text.strip()
            if text:
                chunks.append(DocChunk(text=text, source=pdf_path.name, page=i + 1))

    if not chunks:
        raise RuntimeError("PDFs loaded but no extractable text was found.")

    return chunks
