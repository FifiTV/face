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

Also provides split_enrolled_images() which separates each person's photos
into an enrollment set (70%) and a held-out test set (30%), as required by
the project spec ("tests must use photos not used for enrollment").
"""
import random
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

from .database import EmbeddingDB
from .model import get_embedding
from .utils import list_images


# ── Enrolled image split ──────────────────────────────────────────────────────

def split_enrolled_images(
    enrolled_dir: Path,
    test_dir: Path,
    test_ratio: float = 0.30,
    seed: int = 42,
    reset: bool = False,
) -> dict[str, tuple[int, int]]:
    """
    Split each person's images in enrolled_dir into train and test sets.

    Per project spec: test images must be different from enrollment images.

        enrolled_dir/<name>/  ->  keeps (1 - test_ratio) fraction
        test_dir/<name>/      ->  receives test_ratio fraction

    The split is idempotent: persons already present in test_dir are skipped
    unless reset=True (which moves images back and re-splits).

    Returns:
        dict {person_name: (n_train, n_test)}
    """
    test_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    results: dict[str, tuple[int, int]] = {}

    person_dirs = sorted(d for d in enrolled_dir.iterdir() if d.is_dir())
    for person_dir in person_dirs:
        person_test_dir = test_dir / person_dir.name
        existing_test = list_images(person_test_dir) if person_test_dir.exists() else []

        if existing_test and not reset:
            results[person_dir.name] = (len(list_images(person_dir)), len(existing_test))
            continue

        if reset and existing_test:
            for p in existing_test:
                shutil.move(str(p), person_dir / p.name)
            if person_test_dir.exists() and not list_images(person_test_dir):
                person_test_dir.rmdir()

        all_imgs = list_images(person_dir)
        if not all_imgs:
            results[person_dir.name] = (0, 0)
            continue

        shuffled = all_imgs[:]
        rng.shuffle(shuffled)
        n_test = max(1, round(len(shuffled) * test_ratio))
        test_imgs = shuffled[:n_test]

        person_test_dir.mkdir(parents=True, exist_ok=True)
        for p in test_imgs:
            shutil.move(str(p), person_test_dir / p.name)

        results[person_dir.name] = (len(shuffled) - n_test, n_test)

    return results


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
