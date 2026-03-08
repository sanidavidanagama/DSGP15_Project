from __future__ import annotations
from dataclasses import dataclass
from typing import List

from app.ml.dia_model.config import RagConfig
from app.ml.dia_model.pdf_loader import load_pdfs_from_folder
from app.ml.dia_model.text_splitter import SimpleTextSplitter
from app.ml.dia_model.vector_store import ChromaVectorStore, LocalEmbedder, RetrievedChunk

@dataclass
class RagRetriever:
    config: RagConfig

    def __post_init__(self) -> None:
        self._embedder = LocalEmbedder()
        self._store = ChromaVectorStore(
            persist_dir=self.config.chroma_dir,
            collection_name="dia_literature",
            embedder=self._embedder,
        )

    def build_or_update_index(self) -> None:
        # For simplicity, we just upsert everything each run.
        docs = load_pdfs_from_folder(self.config.data_dir)
        splitter = SimpleTextSplitter(chunk_size=900, overlap=150)
        chunks = splitter.split(docs)
        self._store.add_chunks(chunks)

    def retrieve(self, query: str) -> List[RetrievedChunk]:
        return self._store.similarity_search(query=query, top_k=self.config.top_k)
