"""
Download FaceScrub dataset from Google Drive and split into train/val/test/enrolled sets.

All four sets are pairwise disjoint at the *person* level (no identity overlap).

Usage:
    venv/Scripts/python lab/face/download_data.py [--skip-download] [--skip-extract]

Flags:
    --skip-download   Skip Google Drive download (zips already present in data/raw/)
    --skip-extract    Skip extraction (images already present in data/raw/)
"""
import argparse
import random
import shutil
import tomllib
import zipfile
from pathlib import Path

import gdown

# ── Config ────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
CONFIG_PATH = BASE_DIR / "config.toml"

with open(CONFIG_PATH, "rb") as f:
    cfg = tomllib.load(f)

SEED         = cfg["data"]["seed"]
N_ENROLLED   = cfg["data"]["n_enrolled"]
TRAIN_RATIO  = cfg["data"]["train_ratio"]
VAL_RATIO    = cfg["data"]["val_ratio"]
TEST_RATIO   = cfg["data"]["test_ratio"]
FOLDER_ID    = cfg["data"]["gdrive_folder_id"]

RAW_DIR      = BASE_DIR / cfg["paths"]["raw_dir"]
TRAIN_DIR    = BASE_DIR / cfg["paths"]["train_dir"]
VAL_DIR      = BASE_DIR / cfg["paths"]["val_dir"]
TEST_DIR     = BASE_DIR / cfg["paths"]["test_dir"]
ENROLLED_DIR = BASE_DIR / cfg["paths"]["enrolled_dir"]

assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-6, \
    "train_ratio + val_ratio + test_ratio must equal 1.0"
assert N_ENROLLED >= 80, "n_enrolled must be >= 80 (project spec)"

# ── Step 1: Download ──────────────────────────────────────────────────────────

def download(skip: bool) -> None:
    if skip:
        print("[download] Skipped.")
        return
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[download] Downloading FaceScrub folder from Google Drive -> {RAW_DIR}")
    gdown.download_folder(
        id=FOLDER_ID,
        output=str(RAW_DIR),
        quiet=False,
        use_cookies=False,
    )
    print("[download] Done.")

# ── Step 2: Extract zips ──────────────────────────────────────────────────────

def extract(skip: bool) -> None:
    if skip:
        print("[extract] Skipped.")
        return
    zips = list(RAW_DIR.glob("*.zip"))
    if not zips:
        print("[extract] No zip files found in data/raw/ — skipping.")
        return
    for zip_path in sorted(zips):
        print(f"[extract] Extracting {zip_path.name} ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(RAW_DIR)
    print(f"[extract] Extracted {len(zips)} archive(s).")

# ── Step 3: Collect identities ────────────────────────────────────────────────

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

def collect_identities() -> list[Path]:
    """
    Return sorted list of person directories found under data/raw/.
    Supports both flat layout (raw/<person>/) and nested (raw/actors/<person>/).
    """
    candidates: list[Path] = []
    for d in sorted(RAW_DIR.rglob("*")):
        if not d.is_dir():
            continue
        images = [p for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS]
        if images:
            candidates.append(d)

    # Keep only leaf directories (actual person folders, not category roots)
    leaf = []
    for d in candidates:
        children_with_images = any(
            c.is_dir() and any(p.suffix.lower() in IMAGE_EXTS for p in c.iterdir())
            for c in d.iterdir() if c.is_dir()
        )
        if not children_with_images:
            leaf.append(d)

    print(f"[collect] Found {len(leaf)} identities.")
    return leaf

# ── Step 4: Split & copy ──────────────────────────────────────────────────────

def split_and_copy(identities: list[Path]) -> None:
    rng = random.Random(SEED)
    shuffled = identities[:]
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    assert n_total >= N_ENROLLED + 3, (
        f"Not enough identities ({n_total}) for {N_ENROLLED} enrolled + train/val/test."
    )

    enrolled   = shuffled[:N_ENROLLED]
    remaining  = shuffled[N_ENROLLED:]

    n_remaining = len(remaining)
    n_train = int(n_remaining * TRAIN_RATIO)
    n_val   = int(n_remaining * VAL_RATIO)
    # test gets the rest so rounding never loses anyone
    train_ids = remaining[:n_train]
    val_ids   = remaining[n_train : n_train + n_val]
    test_ids  = remaining[n_train + n_val :]

    splits = {
        ENROLLED_DIR: enrolled,
        TRAIN_DIR:    train_ids,
        VAL_DIR:      val_ids,
        TEST_DIR:     test_ids,
    }

    for dest_root, ids in splits.items():
        dest_root.mkdir(parents=True, exist_ok=True)
        for person_dir in ids:
            dest = dest_root / person_dir.name
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(person_dir, dest)

    print(
        f"[split]  seed={SEED}\n"
        f"         enrolled : {len(enrolled):4d} people\n"
        f"         train    : {len(train_ids):4d} people\n"
        f"         val      : {len(val_ids):4d} people\n"
        f"         test     : {len(test_ids):4d} people\n"
        f"         total    : {n_total:4d} people"
    )

# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Download and split FaceScrub dataset.")
    parser.add_argument("--skip-download", action="store_true", help="Skip Google Drive download")
    parser.add_argument("--skip-extract",  action="store_true", help="Skip zip extraction")
    args = parser.parse_args()

    download(skip=args.skip_download)
    extract(skip=args.skip_extract)

    identities = collect_identities()
    split_and_copy(identities)
    print("[done] Dataset ready.")


if __name__ == "__main__":
    main()
