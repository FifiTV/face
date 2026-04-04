"""
Embedding database backed by ChromaDB (persistent vector store).
Raw images are NOT stored — only embeddings (project spec requirement).

Collection uses cosine distance space so ChromaDB's ANN index can be used
directly for 1:N nearest-neighbour lookup.

Distance convention (ChromaDB cosine space):
    distance = 1 - cosine_similarity
    => similarity = 1 - distance   (0 = identical, 2 = opposite)
"""
from __future__ import annotations

from pathlib import Path

import chromadb
import numpy as np

DEFAULT_DB_PATH = Path(__file__).parent.parent / "embeddings" / "chromadb"
COLLECTION_NAME = "faces"


class EmbeddingDB:
    def __init__(self, path: str | Path = DEFAULT_DB_PATH) -> None:
        self._path = Path(path)
        self._path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self._path))
        self._col = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    # ── Write ─────────────────────────────────────────────────────────────────

    def add_user(self, user_id: str, embedding: np.ndarray) -> None:
        """Store one embedding for user_id. Multiple calls append more embeddings."""
        existing = self._col.get(where={"user_id": user_id}, include=[])
        idx = len(existing["ids"])
        self._col.add(
            ids=[f"{user_id}__{idx}"],
            embeddings=[embedding.tolist()],
            metadatas=[{"user_id": user_id}],
        )

    def remove_user(self, user_id: str) -> None:
        """Delete all embeddings for user_id."""
        self._col.delete(where={"user_id": user_id})

    # ── Read ──────────────────────────────────────────────────────────────────

    def get_user_embeddings(self, user_id: str) -> list[np.ndarray]:
        """Return all stored embeddings for user_id."""
        result = self._col.get(
            where={"user_id": user_id},
            include=["embeddings"],
        )
        return [np.array(e) for e in result["embeddings"]]

    def get_all_users(self) -> list[str]:
        all_meta = self._col.get(include=["metadatas"])["metadatas"]
        return list({m["user_id"] for m in all_meta})

    def __len__(self) -> int:
        return len(self.get_all_users())

    def count_embeddings(self) -> int:
        """Total number of stored embedding vectors (across all users)."""
        return self._col.count()

    # ── Nearest-neighbour queries (used by authorization.py) ──────────────────

    def query_top1(self, embedding: np.ndarray) -> tuple[str, float]:
        """
        Find the closest stored embedding across all users.

        Returns:
            (user_id, cosine_similarity)  — similarity in [-1, 1]
        Raises:
            ValueError if the database is empty.
        """
        if self._col.count() == 0:
            raise ValueError("Database is empty — enroll users first.")
        result = self._col.query(
            query_embeddings=[embedding.tolist()],
            n_results=1,
            include=["metadatas", "distances"],
        )
        user_id = result["metadatas"][0][0]["user_id"]
        similarity = 1.0 - result["distances"][0][0]
        return user_id, similarity

    def query_user(self, user_id: str, embedding: np.ndarray) -> float:
        """
        Find maximum cosine similarity between embedding and all stored
        embeddings of the given user (used for 1:1 verification).

        Returns:
            max cosine_similarity in [-1, 1]
        """
        stored = self.get_user_embeddings(user_id)
        if not stored:
            return 0.0
        sims = [float(np.dot(embedding, ref)) for ref in stored]
        return max(sims)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self) -> None:
        """No-op: ChromaDB writes to disk automatically on every mutation."""

    def load(self) -> "EmbeddingDB":
        """No-op: ChromaDB loads from disk automatically on client creation."""
        return self

    @classmethod
    def from_file(cls, path: str | Path = DEFAULT_DB_PATH) -> "EmbeddingDB":
        return cls(path)
