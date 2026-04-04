"""
User enrollment: build embedding database from a folder of images.

Expected folder structure:
    data/enrolled/
        alice/
            img1.jpg
            img2.jpg
        bob/
            img1.jpg

Each user is represented by a single *averaged* embedding (mean of all
per-image embeddings, re-normalised to unit length).
"""
import time
from pathlib import Path

import cv2
import numpy as np

from .database import EmbeddingDB
from .model import get_embedding
from .utils import list_images


def _average_embeddings(embeddings: list[np.ndarray]) -> np.ndarray:
    """Mean-pool embeddings and re-normalise to unit length."""
    stack = np.stack(embeddings, axis=0)          # (N, 512)
    mean = stack.mean(axis=0)                      # (512,)
    norm = np.linalg.norm(mean)
    return mean / norm if norm > 0 else mean


def enroll_user_averaged(
    user_id: str,
    images: list[np.ndarray],
    app,
    db: EmbeddingDB,
    detect: bool = False,
) -> int:
    """
    Enroll a user with a single averaged embedding.

    Extracts one embedding per image, averages them, L2-normalises the
    result, and stores it as a single vector in ChromaDB.

    Args:
        detect: If True, run face detection first; if no face is found the
                image is skipped.  If False (default), treat each image as a
                pre-cropped face and feed it directly to ArcFace — suitable
                for datasets like FaceScrub where images are already aligned
                face crops.

    Returns:
        Number of images that contributed to the average (0 = none usable).
    """
    embeddings = []
    for img in images:
        # detect=False  → fallback=True  (skip detector, use full image)
        # detect=True   → fallback=False (detector must find a face)
        emb = get_embedding(app, img, fallback=not detect)
        if emb is not None:
            embeddings.append(emb)

    if not embeddings:
        return 0

    avg_emb = _average_embeddings(embeddings)
    db.add_user(user_id, avg_emb)
    return len(embeddings)


def enroll_from_folder(
    enrolled_dir: str | Path,
    app,
    db: EmbeddingDB,
    detect: bool = False,
    skip_existing: bool = True,
) -> dict[str, int]:
    """
    Batch enroll all users from enrolled_dir/<user_id>/ structure.

    Each user ends up with exactly one averaged embedding in ChromaDB.

    Args:
        detect:        Whether to run face detection (see enroll_user_averaged).
        skip_existing: If True, users already present in ChromaDB are skipped.
                       If False, their embedding is removed and re-computed.

    Returns:
        dict {user_id: n_images_used}  (0 means no usable face found / skipped)
    """
    enrolled_dir = Path(enrolled_dir)
    existing = set(db.get_all_users())
    results = {}

    user_dirs = sorted(d for d in enrolled_dir.iterdir() if d.is_dir())
    for user_dir in user_dirs:
        user_id = user_dir.name

        if user_id in existing:
            if skip_existing:
                print(f"  {user_id}: already in DB, skipping.")
                results[user_id] = 0
                continue
            else:
                db.remove_user(user_id)

        images = [cv2.imread(str(p)) for p in list_images(user_dir)]
        images = [img for img in images if img is not None]

        if not images:
            print(f"  WARNING: no images found for {user_id}, skipping.")
            continue

        n = enroll_user_averaged(user_id, images, app, db, detect=detect)
        results[user_id] = n
        print(f"  {user_id}: averaged {n}/{len(images)} embeddings -> 1 vector stored")

    return results


def measure_enrollment_time(
    img_bgr: np.ndarray,
    app,
    db: EmbeddingDB,
    user_id: str = "_timing_test",
) -> float:
    """Return wall-clock time (seconds) to enroll a single image."""
    start = time.perf_counter()
    emb = get_embedding(app, img_bgr)
    if emb is not None:
        db.add_user(user_id, emb)
    elapsed = time.perf_counter() - start
    db.remove_user(user_id)
    return elapsed
