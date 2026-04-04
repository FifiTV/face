"""
Authorization: 1:1 verification and 1:N identification.
Both operations delegate nearest-neighbour search to ChromaDB.
"""
import time

import numpy as np

from .database import EmbeddingDB
from .model import get_embedding

DEFAULT_THRESHOLD = 0.4  # cosine similarity threshold — tune with Task 6


def verify(
    img_bgr: np.ndarray,
    user_id: str,
    app,
    db: EmbeddingDB,
    threshold: float = DEFAULT_THRESHOLD,
) -> tuple[bool, float]:
    """
    1:1 verification: does this image belong to the claimed user?

    Uses ChromaDB to fetch the user's stored embeddings and computes
    max cosine similarity against the probe.

    Returns:
        (authorized, score)  — score is max cosine similarity in [-1, 1]
    """
    probe = get_embedding(app, img_bgr)
    if probe is None:
        return False, 0.0

    score = db.query_user(user_id, probe)
    return score >= threshold, score


def identify(
    img_bgr: np.ndarray,
    app,
    db: EmbeddingDB,
    threshold: float = DEFAULT_THRESHOLD,
) -> tuple[str | None, float]:
    """
    1:N identification: who is this person?

    Uses ChromaDB ANN query (HNSW index, cosine space) for fast lookup.

    Returns:
        (user_id, score)  — user_id is None when best score < threshold
    """
    probe = get_embedding(app, img_bgr)
    if probe is None:
        return None, 0.0

    user_id, score = db.query_top1(probe)
    if score >= threshold:
        return user_id, score
    return None, score


def measure_verify_time(
    img_bgr: np.ndarray,
    user_id: str,
    app,
    db: EmbeddingDB,
    n: int = 20,
) -> float:
    """Return mean 1:1 verification time (seconds) over n runs."""
    times = []
    for _ in range(n):
        t = time.perf_counter()
        verify(img_bgr, user_id, app, db)
        times.append(time.perf_counter() - t)
    return float(np.mean(times))


def measure_identify_time(
    img_bgr: np.ndarray,
    app,
    db: EmbeddingDB,
    n: int = 20,
) -> float:
    """Return mean 1:N identification time (seconds) over n runs."""
    times = []
    for _ in range(n):
        t = time.perf_counter()
        identify(img_bgr, app, db)
        times.append(time.perf_counter() - t)
    return float(np.mean(times))
