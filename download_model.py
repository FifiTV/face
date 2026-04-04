"""
Download ArcFace model (InsightFace buffalo_l) to lab/face/models/.
Run: venv/Scripts/python lab/face/download_model.py
"""
import cv2
import numpy as np
from pathlib import Path
from insightface.app import FaceAnalysis

MODEL_NAME = "buffalo_l"       # ArcFace R50 recognition + RetinaFace detector
# InsightFace appends "models/" to root automatically -> weights end up in lab/face/models/<name>/
MODEL_ROOT = Path(__file__).parent
(MODEL_ROOT / "models").mkdir(exist_ok=True)

print(f"Downloading model '{MODEL_NAME}' -> {MODEL_ROOT / 'models'} ...")
app = FaceAnalysis(name=MODEL_NAME, root=str(MODEL_ROOT), allowed_modules=["detection", "recognition"])
app.prepare(ctx_id=0, det_size=(640, 640))  # ctx_id=-1 for CPU-only
print("Model ready.")

# Quick sanity check
dummy = np.zeros((112, 112, 3), dtype=np.uint8)
faces = app.get(dummy)
print(f"Test on blank image: detected {len(faces)} face(s) (expected 0).")
print("All OK — you can start enrollment.")
