"""
Image preprocessing: face detection, alignment, resizing, normalization.
Uses OpenCV and dlib (already in venv).
"""
import cv2
import numpy as np
from pathlib import Path


TARGET_SIZE = (112, 112)  # standard ArcFace input size


def load_and_preprocess(path: str | Path, target_size: tuple = TARGET_SIZE) -> np.ndarray | None:
    """Load image, detect face, align and resize. Returns BGR uint8 or None."""
    img = cv2.imread(str(path))
    if img is None:
        return None
    return detect_and_align(img, target_size)


def detect_and_align(img_bgr: np.ndarray, target_size: tuple = TARGET_SIZE) -> np.ndarray | None:
    """
    Detect the largest face and return aligned crop resized to target_size.
    Uses OpenCV Haar cascade for detection.
    Returns BGR uint8 array or None if no face found.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    detector = _get_face_detector()
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return None

    # pick largest face
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    # add small margin
    margin = int(min(w, h) * 0.1)
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(img_bgr.shape[1], x + w + margin)
    y2 = min(img_bgr.shape[0], y + h + margin)

    crop = img_bgr[y1:y2, x1:x2]
    return cv2.resize(crop, target_size)


def normalize(img_bgr: np.ndarray) -> np.ndarray:
    """Normalize to float32 in [-1, 1] range (standard ArcFace normalization)."""
    return (img_bgr.astype(np.float32) - 127.5) / 128.0


_face_detector_cache = None


def _get_face_detector():
    global _face_detector_cache
    if _face_detector_cache is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _face_detector_cache = cv2.CascadeClassifier(cascade_path)
    return _face_detector_cache
