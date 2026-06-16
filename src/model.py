"""
Face embedding model using InsightFace (ArcFace backend, ONNX runtime — no TensorFlow).
For custom PyTorch fine-tuning use the FaceEmbedder class with a torch backbone.
"""
import os
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np

# Set INSIGHTFACE_HOME before importing insightface
# Models will be stored at ~/.insightface/ (default insightface cache location)
if 'INSIGHTFACE_HOME' not in os.environ:
    os.environ['INSIGHTFACE_HOME'] = str(Path.home() / '.insightface')


@contextmanager
def _quiet():
    """Suppress verbose stdout/stderr from InsightFace and ONNX runtime."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        # Redirect C-level stdout/stderr (ONNX runtime prints via printf)
        devnull = open(os.devnull, "w")
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr
            devnull.close()


def get_insightface_model(model_name: str = "buffalo_l", ctx_id: int = 0):
    """
    Load InsightFace ArcFace model from ~/.insightface/models/.

    Args:
        model_name: InsightFace model pack ('buffalo_l', 'buffalo_sc', etc.)
        ctx_id: GPU id (0) or -1 for CPU
    Returns:
        insightface.app.FaceAnalysis instance ready for inference
    """
    from insightface.app import FaceAnalysis

    with _quiet():
        app = FaceAnalysis(name=model_name)
        app.prepare(ctx_id=ctx_id, det_size=(640, 640))
    return app


def _embed_crop(app, img_bgr: np.ndarray) -> np.ndarray:
    """
    Feed a pre-cropped face image directly to ArcFace, bypassing detection.

    Used as a fallback for datasets where images are already aligned face crops
    (e.g. FaceScrub), where the detector often fails because the face fills
    the entire image and falls below the minimum detection size.
    """
    rec = app.models["recognition"]
    # get_feat expects a list of BGR uint8 images — it handles normalisation internally
    img = cv2.resize(img_bgr, (112, 112))
    feat = rec.get_feat([img])   # (1, 512)
    emb = feat[0]
    norm = np.linalg.norm(emb)
    return emb / norm if norm > 0 else emb


def get_embedding(app, img_bgr: np.ndarray, fallback: bool = True) -> np.ndarray | None:
    """
    Get ArcFace embedding for the face that best matches:
      - large face area
      - proximity to image center

    If no face is detected and fallback=True, use the entire image as a crop.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        faces = app.get(img_bgr)

    if faces:
        h, w = img_bgr.shape[:2]
        img_center_x = w / 2
        img_center_y = h / 2

        max_distance = np.sqrt(img_center_x**2 + img_center_y**2)

        def score(face):
            x1, y1, x2, y2 = face.bbox

            # Face area
            area = (x2 - x1) * (y2 - y1)
            area_norm = area / (w * h)

            # Face center
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            # Distance from image center
            dist = np.sqrt(
                (cx - img_center_x) ** 2 +
                (cy - img_center_y) ** 2
            )

            center_score = 1.0 - (dist / max_distance)

            # Tunable weights
            return 0.7 * area_norm + 0.3 * center_score

        best_face = max(faces, key=score)
        return best_face.normed_embedding

    if fallback:
        return _embed_crop(app, img_bgr)

    return None


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two L2-normalised embeddings."""
    return float(np.dot(a, b))
