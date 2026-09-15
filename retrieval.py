from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class RetrievedDocument:
    """Raw, attributable evidence returned by a retriever."""

    document_id: str
    text: str
    score: float
    metadata: dict[str, object]

try:
    from torch._higher_order_ops.associative_scan import associative_scan as _associative_scan
except ImportError:  # Older PyTorch builds do not expose the prototype API.
    _associative_scan = None

class SQLiteFTSRetriever:
    """Persistent lexical retrieval with raw text and provenance.

    SQLite FTS5 is a real, dependency-free retrieval backend suitable for a
    demo or a small private corpus. It performs exact lexical matching, unlike
    the vector-only demo retriever. For a large corpus, keep this API and swap
    the implementation for a managed BM25/ANN service plus a reranker.
    """

    def __init__(self, database_path: str | Path) -> None:
        path = Path(database_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("CREATE TABLE IF NOT EXISTS documents (document_id TEXT PRIMARY KEY, text TEXT NOT NULL, metadata_json TEXT NOT NULL)")
        self.connection.execute("CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(document_id UNINDEXED, text, tokenize='unicode61')")

    def close(self) -> None:
        self.connection.close()

    def upsert(self, document_id: str, text: str, metadata: dict[str, object] | None = None) -> None:
        if not document_id or not text:
            raise ValueError("document_id and text must be non-empty")
        metadata_json = json.dumps(metadata or {}, sort_keys=True)
        with self.connection:
            self.connection.execute("DELETE FROM documents_fts WHERE document_id = ?", (document_id,))
            self.connection.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))
            self.connection.execute("INSERT INTO documents(document_id, text, metadata_json) VALUES (?, ?, ?)", (document_id, text, metadata_json))
            self.connection.execute("INSERT INTO documents_fts(document_id, text) VALUES (?, ?)", (document_id, text))

    def search(self, query: str, top_k: int = 4) -> list[RetrievedDocument]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        # Quote individual terms so ordinary user punctuation cannot alter FTS
        # query syntax. This is lexical retrieval, not semantic retrieval.
        terms = re.findall(r"\w+", query, flags=re.UNICODE)
        if not terms:
            return []
        fts_query = " AND ".join(f'"{term}"' for term in terms)
        rows = self.connection.execute(
            "SELECT d.document_id, d.text, d.metadata_json, bm25(documents_fts) "
            "FROM documents_fts JOIN documents d USING(document_id) "
            "WHERE documents_fts MATCH ? ORDER BY bm25(documents_fts) LIMIT ?",
            (fts_query, top_k),
        ).fetchall()
        return [RetrievedDocument(row[0], row[1], float(row[3]), json.loads(row[2])) for row in rows]


class FAISSVectorRetriever:
    """Semantic retrieval backed by a FAISS CPU L2 index.

    Documents are embedded with Sentence Transformers and stored alongside
    their FAISS vector positions so search results retain their source text
    and metadata.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        *,
        embedder: object | None = None,
    ) -> None:
        try:
            import faiss
        except ImportError as error:
            raise ImportError("Install faiss-cpu to use FAISSVectorRetriever") from error

        if embedder is None:
            from sentence_transformers import SentenceTransformer

            embedder = SentenceTransformer(model_name)
        self.embedder = embedder
        self._faiss = faiss
        self._documents: list[RetrievedDocument] = []
        self._vectors: list[object] = []
        self.index = None

    def _encode(self, texts: list[str]) -> object:
        import numpy as np

        vectors = self.embedder.encode(texts, convert_to_numpy=True)
        return np.asarray(vectors, dtype="float32")

    def _rebuild_index(self) -> None:
        if not self._vectors:
            self.index = None
            return
        import numpy as np

        vectors = np.vstack(self._vectors).astype("float32", copy=False)
        self.index = self._faiss.IndexFlatL2(vectors.shape[1])
        self.index.add(vectors)

    def upsert(
        self,
        document_id: str,
        text: str,
        metadata: dict[str, object] | None = None,
    ) -> None:
        if not document_id or not text:
            raise ValueError("document_id and text must be non-empty")

        document = RetrievedDocument(document_id, text, 0.0, metadata or {})
        vector = self._encode([text])[0:1]
        existing = next(
            (position for position, item in enumerate(self._documents) if item.document_id == document_id),
            None,
        )
        if existing is None:
            self._documents.append(document)
            self._vectors.append(vector)
        else:
            self._documents[existing] = document
            self._vectors[existing] = vector
        self._rebuild_index()

    def search(self, query: str, top_k: int = 4) -> list[RetrievedDocument]:
        if not query:
            raise ValueError("query must be non-empty")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.index is None:
            return []

        query_vector = self._encode([query])
        distances, indices = self.index.search(query_vector, min(top_k, len(self._documents)))
        return [
            RetrievedDocument(
                self._documents[index].document_id,
                self._documents[index].text,
                -float(distance),
                self._documents[index].metadata,
            )
            for distance, index in zip(distances[0], indices[0])
        ]


