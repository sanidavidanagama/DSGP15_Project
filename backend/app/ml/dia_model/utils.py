from __future__ import annotations
from pathlib import Path
import mimetypes

def read_image_bytes(image_path: str) -> tuple[bytes, str]:
    p = Path(image_path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p}")

    mime, _ = mimetypes.guess_type(str(p))
    if mime is None:
        mime = "image/jpeg"

    return p.read_bytes(), mime

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def normalize_whitespaces(s: str) -> str:
    return " ".join(s.split())