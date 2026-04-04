"""
Image degradation functions for robustness experiments:
- Gaussian noise (controlled by target PSNR)
- Luminance changes in YCbCr space (quadratic, linear, offset)
- JPEG compression
"""
import io
import cv2
import numpy as np
from PIL import Image


# ── Noise (Task 3) ────────────────────────────────────────────────────────────

def add_noise_to_psnr(img: np.ndarray, target_psnr: float) -> np.ndarray:
    """
    Add Gaussian noise to reach approximately target_psnr (dB).
    Works on uint8 BGR images.
    """
    img_f = img.astype(np.float64)
    sigma = 255.0 / (10 ** (target_psnr / 20.0))
    noise = np.random.normal(0, sigma, img.shape)
    noisy = np.clip(img_f + noise, 0, 255).astype(np.uint8)
    return noisy


PSNR_BANDS = [
    (50, 80),
    (40, 50),
    (30, 40),
    (20, 30),
    (10, 20),
]


def psnr_band_midpoint(band: tuple[int, int]) -> float:
    return (band[0] + band[1]) / 2.0


# ── Luminance (Task 4) ────────────────────────────────────────────────────────

def _to_ycbcr(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)


def _from_ycbcr(img_ycrcb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_ycrcb, cv2.COLOR_YCrCb2BGR)


def _clip_y(y: np.ndarray) -> np.ndarray:
    return np.clip(y, 0, 255).astype(np.uint8)


def apply_luminance_quadratic(img_bgr: np.ndarray) -> np.ndarray:
    """Quadratic scaling of Y channel: Y_new = (Y/255)^2 * 255."""
    ycrcb = _to_ycbcr(img_bgr).astype(np.float32)
    ycrcb[:, :, 0] = (ycrcb[:, :, 0] / 255.0) ** 2 * 255.0
    ycrcb[:, :, 0] = _clip_y(ycrcb[:, :, 0])
    return _from_ycbcr(ycrcb.astype(np.uint8))


LUMINANCE_LINEAR_COEFFS = [1 / 2, 3 / 5, 3 / 4, 4 / 3, 3 / 2]


def apply_luminance_linear(img_bgr: np.ndarray, coeff: float) -> np.ndarray:
    """Linear scaling of Y channel: Y_new = coeff * Y."""
    ycrcb = _to_ycbcr(img_bgr).astype(np.float32)
    ycrcb[:, :, 0] = _clip_y(ycrcb[:, :, 0] * coeff)
    return _from_ycbcr(ycrcb.astype(np.uint8))


LUMINANCE_OFFSETS = [-100, -20, -10, 30]


def apply_luminance_offset(img_bgr: np.ndarray, offset: int) -> np.ndarray:
    """Constant offset to Y channel: Y_new = Y + offset."""
    ycrcb = _to_ycbcr(img_bgr).astype(np.float32)
    ycrcb[:, :, 0] = _clip_y(ycrcb[:, :, 0] + offset)
    return _from_ycbcr(ycrcb.astype(np.uint8))


# ── JPEG compression (Task 7) ─────────────────────────────────────────────────

JPEG_QUALITIES = [10, 30, 70]  # at least 3 levels as required


def apply_jpeg(img_bgr: np.ndarray, quality: int) -> np.ndarray:
    """Encode and decode with JPEG at given quality (1–95)."""
    buf = io.BytesIO()
    pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    pil_img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    result = Image.open(buf)
    return cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
