"""
Download ArcFace model (InsightFace buffalo_l) to ~/.insightface/.
Run: .venv/bin/python download_model.py
"""
import os
from pathlib import Path

# Set INSIGHTFACE_HOME before importing insightface
# Models will be stored at ~/.insightface/ (default insightface cache location)
if 'INSIGHTFACE_HOME' not in os.environ:
    os.environ['INSIGHTFACE_HOME'] = str(Path.home() / '.insightface')

import numpy as np
from insightface.app import FaceAnalysis

MODEL_NAME = "buffalo_l"       # ArcFace R50 recognition + RetinaFace detector

print(f"Downloading model '{MODEL_NAME}' to {os.environ['INSIGHTFACE_HOME']} ...")
app = FaceAnalysis(name=MODEL_NAME)
app.prepare(ctx_id=0, det_size=(640, 640))  # ctx_id=-1 for CPU-only
print("Model ready.")

# Quick sanity check
dummy = np.zeros((112, 112, 3), dtype=np.uint8)
faces = app.get(dummy)
print(f"Test on blank image: detected {len(faces)} face(s) (expected 0).")
print("All OK — you can start enrollment.")
