from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any
from pathlib import Path

from sympy.multipledispatch.dispatcher import source
from torch.nn.functional import embedding
from sentence_transformers import SentenceTransformer

from utils import ensure_dir
from text_splitter import SplitChunk

@dataclass
class RetrievedChunk:
    text: str
    source: str
    page: int
    chunk_id: str
    score: float

class LocalEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.tolist()


class ChromaVectorStore:
    """
    Minimal Chroma wrapper: add documents + similarity search.
    Requires: uv add chromadb
    """
    def __init__(self, persist_dir: Path, collection_name: str, embedder: LocalEmbedder()): # type: ignore
        ensure_dir(persist_dir)
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.embedder = embedder

        try:
            import chromadb
        except Exception as e:
            raise ImportError(
                "Missing dependency 'chromadb'. Install it with: uv add chromadb"
            ) from e

        self._chroma = chromadb.PersistentClient(path=str(persist_dir))
        self._col = self._chroma.get_or_create_collection(name=collection_name)

    def count(self) -> int:
        return self._col.count()

    def add_chunks(self, chunks: List[SplitChunk]) -> None:
        if not chunks:
            return

        ids = [c.chunk_id for c in chunks]
        metadatas = [{"source": c.source, "page": c.page} for c in chunks]
        documents = [c.text for c in chunks]
        embeddings = self.embedder.embed_texts(documents)

        self._col.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    def similarity_search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        q_emb = self.embedder.embed_texts([query])[0]

        res = self._col.query(
            query_embeddings=[q_emb],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],  # no "ids" here
        )

        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        ids = (res.get("ids") or [[]])[0]  # ids still returned by Chroma

        out: list[RetrievedChunk] = []
        for doc, meta, dist, cid in zip(docs, metas, dists, ids):
            out.append(
                RetrievedChunk(
                    text=doc,
                    source=str(meta.get("source", "")),
                    page=int(meta.get("page", -1)),
                    chunk_id=str(cid),
                    score=float(dist),  # distance; smaller = better
                )
            )
        return out
