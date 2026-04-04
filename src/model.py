"""
Face embedding model using InsightFace (ArcFace backend, ONNX runtime — no TensorFlow).
For custom PyTorch fine-tuning use the FaceEmbedder class with a torch backbone.
"""
from pathlib import Path

import numpy as np

# InsightFace appends "models/" to root automatically, so point one level up
# to get weights stored at lab/face/models/<model_name>/
DEFAULT_MODEL_ROOT = Path(__file__).parent.parent


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

    app = FaceAnalysis(
        name=model_name,
        root=str(DEFAULT_MODEL_ROOT),
        allowed_modules=["detection", "recognition"],
    )
    app.prepare(ctx_id=ctx_id, det_size=(640, 640))
    return app


def get_embedding(app, img_bgr: np.ndarray) -> np.ndarray | None:
    """
    Get ArcFace embedding for the largest face in the image.

    Args:
        app: FaceAnalysis instance from get_insightface_model()
        img_bgr: BGR image as numpy array
    Returns:
        512-d normalized embedding or None if no face detected
    """
    faces = app.get(img_bgr)
    if not faces:
        return None
    # pick face with largest bounding box area
    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    return face.normed_embedding  # shape (512,), L2-normalized


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two L2-normalized embeddings."""
    return float(np.dot(a, b))
