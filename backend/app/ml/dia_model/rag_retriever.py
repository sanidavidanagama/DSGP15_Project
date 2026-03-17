from __future__ import annotations
from dataclasses import dataclass
import json
import threading
import time
from pathlib import Path
from typing import List
import logging

from app.ml.dia_model.config import RagConfig
from app.ml.dia_model.pdf_loader import load_pdfs_from_folder
from app.ml.dia_model.text_splitter import SimpleTextSplitter
from app.ml.dia_model.vector_store import ChromaVectorStore, LocalEmbedder, RetrievedChunk

logger = logging.getLogger(__name__)

@dataclass
class RagRetriever:
    config: RagConfig
    _build_lock = threading.Lock()

    def __post_init__(self) -> None:
        model_name = self.config.st_embed_model or "sentence-transformers/all-MiniLM-L6-v2"
        self._embedder = LocalEmbedder(model_name=model_name)
        self._store = ChromaVectorStore(
            persist_dir=self.config.chroma_dir,
            collection_name="dia_literature",
            embedder=self._embedder,
        )
        self._meta_path = self.config.chroma_dir / "index_meta.json"

    def _pdf_fingerprint(self) -> dict:
        pdf_files = sorted(self.config.data_dir.glob("*.pdf"))
        return {
            "files": [
                {
                    "name": p.name,
                    "size": p.stat().st_size,
                    "mtime_ns": p.stat().st_mtime_ns,
                }
                for p in pdf_files
            ]
        }

    def _current_index_signature(self) -> dict:
        return {
            "pdfs": self._pdf_fingerprint(),
            "chunk_size": 900,
            "overlap": 150,
            "embed_model": self.config.st_embed_model or "sentence-transformers/all-MiniLM-L6-v2",
        }

    def _is_index_fresh(self) -> bool:
        if self._store.count() == 0:
            return False
        if not self._meta_path.exists():
            return False
        try:
            saved = json.loads(self._meta_path.read_text(encoding="utf-8"))
        except Exception:
            return False
        return saved == self._current_index_signature()

    def _write_meta(self, meta: dict) -> None:
        self._meta_path.write_text(json.dumps(meta, ensure_ascii=True, indent=2), encoding="utf-8")

    def build_or_update_index(self) -> None:
        if self._is_index_fresh():
            logger.info("DIA RAG index is fresh. Skipping rebuild.")
            return

        with self._build_lock:
            if self._is_index_fresh():
                logger.info("DIA RAG index became fresh while waiting for lock.")
                return

            t0 = time.perf_counter()
            docs = load_pdfs_from_folder(self.config.data_dir)
            t1 = time.perf_counter()

            splitter = SimpleTextSplitter(chunk_size=900, overlap=150)
            chunks = splitter.split(docs)
            t2 = time.perf_counter()

            self._store.reset_collection()
            self._store.add_chunks(chunks)
            t3 = time.perf_counter()

            meta = self._current_index_signature()
            self._write_meta(meta)
            logger.info(
                "DIA RAG index rebuilt. docs=%d chunks=%d load=%.2fs split=%.2fs embed_upsert=%.2fs total=%.2fs",
                len(docs),
                len(chunks),
                (t1 - t0),
                (t2 - t1),
                (t3 - t2),
                (t3 - t0),
            )

    def retrieve(self, query: str) -> List[RetrievedChunk]:
        return self._store.similarity_search(query=query, top_k=self.config.top_k)
