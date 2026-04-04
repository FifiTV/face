"""
User enrollment: build embedding database from a folder of images.

Expected folder structure:
    data/enrolled/
        alice/
            img1.jpg
            img2.jpg
        bob/
            img1.jpg
"""
import time
from pathlib import Path
import numpy as np

from .database import EmbeddingDB
from .utils import list_images


def enroll_user(user_id: str, images: list[np.ndarray], app, db: EmbeddingDB) -> int:
    """
    Enroll a single user from a list of BGR images.
    Returns number of successfully enrolled embeddings.
    """
    from .model import get_embedding

    count = 0
    for img in images:
        emb = get_embedding(app, img)
        if emb is not None:
            db.add_user(user_id, emb)
            count += 1
    return count


def enroll_from_folder(
    enrolled_dir: str | Path,
    app,
    db: EmbeddingDB,
    save: bool = True,
) -> dict[str, int]:
    """
    Batch enroll all users from enrolled_dir/<user_id>/*.jpg structure.
    Returns dict {user_id: enrolled_count}.
    """
    import cv2

    enrolled_dir = Path(enrolled_dir)
    results = {}

    for user_dir in sorted(enrolled_dir.iterdir()):
        if not user_dir.is_dir():
            continue
        user_id = user_dir.name
        images = [cv2.imread(str(p)) for p in list_images(user_dir)]
        images = [img for img in images if img is not None]
        count = enroll_user(user_id, images, app, db)
        results[user_id] = count
        print(f"  Enrolled {user_id}: {count}/{len(images)} images")

    if save:
        db.save()

    return results


def measure_enrollment_time(img_bgr: np.ndarray, app, db: EmbeddingDB, user_id: str = "_timing_test") -> float:
    """Return time in seconds to enroll a single image."""
    from .model import get_embedding

    start = time.perf_counter()
    emb = get_embedding(app, img_bgr)
    if emb is not None:
        db.add_user(user_id, emb)
    elapsed = time.perf_counter() - start
    db.remove_user(user_id)
    return elapsed
