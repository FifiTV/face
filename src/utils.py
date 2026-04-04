import cv2
import numpy as np
from pathlib import Path


def load_image(path: str | Path) -> np.ndarray:
    """Load image as BGR numpy array."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    return img


def psnr(original: np.ndarray, degraded: np.ndarray) -> float:
    """Compute Peak Signal-to-Noise Ratio (dB) between two images."""
    mse = np.mean((original.astype(np.float64) - degraded.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10(255.0 ** 2 / mse)


def list_images(folder: str | Path, extensions=(".jpg", ".jpeg", ".png")) -> list[Path]:
    """Recursively list all image files in a folder."""
    folder = Path(folder)
    return sorted(p for ext in extensions for p in folder.rglob(f"*{ext}"))
