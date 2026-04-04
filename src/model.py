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

# InsightFace appends "models/" to root automatically, so point one level up
# to get weights stored at lab/face/models/<model_name>/
DEFAULT_MODEL_ROOT = Path(__file__).parent.parent


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
    Load InsightFace ArcFace model from lab/face/models/.

    Args:
        model_name: InsightFace model pack ('buffalo_l', 'buffalo_sc', etc.)
        ctx_id: GPU id (0) or -1 for CPU
    Returns:
        insightface.app.FaceAnalysis instance ready for inference
    """
    from insightface.app import FaceAnalysis

    with _quiet():
        app = FaceAnalysis(
            name=model_name,
            root=str(DEFAULT_MODEL_ROOT),
            allowed_modules=["detection", "recognition"],
        )
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
    Get ArcFace embedding for the largest face in the image.

    If detection finds no face and fallback=True, the whole image is treated as
    a pre-cropped face (appropriate for FaceScrub and similar datasets).

    Args:
        app:      FaceAnalysis instance from get_insightface_model()
        img_bgr:  BGR image as numpy array
        fallback: Use full image as face crop when detection fails (default True)
    Returns:
        512-d L2-normalised embedding, or None only when fallback=False and no face found
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        faces = app.get(img_bgr)

    if faces:
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        return face.normed_embedding  # shape (512,), L2-normalised

    if fallback:
        return _embed_crop(app, img_bgr)

    return None


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two L2-normalised embeddings."""
    return float(np.dot(a, b))
