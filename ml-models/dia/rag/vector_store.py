from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any
from pathlib import Path

from sympy.multipledispatch.dispatcher import source
from torch.nn.functional import embedding

from utils import ensure_dir
from text_splitter import SplitChunk

@dataclass
class RetrievedChunk:
    text: str
    source: str
    page: int
    chunk_id: str
    score: float

class GoogleEmbedder:
    """
    Minimal embedder using the 'google-genai' SDK.
    Requires: uv add google-genai
    Env: GOOGLE_API_KEY=...
    """
    def __init__(self, embedding_model: str):
        self.embedding_model = embedding_model
        try:
            from google import genai
        except Exception as e:
            raise ImportError(
                "Missing dependency 'google-genai'. Install it with: uv add google-genai"
            ) from e
        self._client = genai.Client()

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        resp = self._client.models.embed_content(
            model=self.embedding_model,
            contents=texts,
        )
        return [e.values for e in resp.embeddings]

class ChromaVectorStore:
    """
    Minimal Chroma wrapper: add documents + similarity search.
    Requires: uv add chromadb
    """
    def __init__(self, persist_dir: Path, collection_name: str, embedder: GoogleEmbedder):
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

    def similarity_search(self, query: str, top_k: int) -> List[RetrievedChunk]:
        q_emb = self.embedder.embed_texts([query])[0]
        res = self._col.query(
            query_embeddings=[q_emb],
            n_results=top_k,
            include=["documents", "metadatas", "distances", "ids"]
        )

        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        ids = res.get("ids", [[]])[0]

        out: List[RetrievedChunk] = []
        for doc, meta, dist, cid in zip(docs, metas, dists, ids):
            score = float(dist)
            out.append(dist)
            out.append(
                RetrievedChunk(
                    text=doc,
                    source=str(meta.get("source", "")),
                    page=int(meta.get("page", -1)),
                    chunk_id=str(cid),
                    score=score,
                )
            )
        return out
