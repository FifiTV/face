"""
Build the ChromaDB enrollment database from data/enrolled/.

By default, before enrolling, images in data/enrolled/ are split:
    70%  ->  data/enrolled/          (used to build embeddings)
    30%  ->  data/enrolled_test/     (held-out for testing — project spec)

The split is idempotent: persons already split are skipped unless --reset-split
or --reset is passed.

Usage:
    venv/Scripts/python lab/face/enroll.py               # split + enroll (default)
    venv/Scripts/python lab/face/enroll.py --no-split    # skip split step
    venv/Scripts/python lab/face/enroll.py --reset       # wipe DB + redo split + enroll
    venv/Scripts/python lab/face/enroll.py --update      # re-enroll all (keep split)
    venv/Scripts/python lab/face/enroll.py --detect      # use face detector
"""
import argparse
import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.database import EmbeddingDB
from src.enrollment import enroll_from_folder, split_enrolled_images
from src.model import get_insightface_model

BASE_DIR    = Path(__file__).parent
CONFIG_PATH = BASE_DIR / "config.toml"

with open(CONFIG_PATH, "rb") as f:
    _cfg = tomllib.load(f)

ENROLLED_DIR      = BASE_DIR / _cfg["paths"]["enrolled_dir"]
ENROLLED_TEST_DIR = BASE_DIR / _cfg["paths"]["enrolled_test_dir"]
TEST_RATIO        = _cfg["enrolled_split"]["test_ratio"]
SEED              = _cfg["data"]["seed"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split enrolled images and build ChromaDB.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── DB mode ───────────────────────────────────────────────────────────────
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--update",
        action="store_true",
        help="Re-enroll everyone, overwriting existing DB entries.",
    )
    mode.add_argument(
        "--reset",
        action="store_true",
        help="Wipe the entire DB, redo the enrolled split, enroll from scratch.",
    )

    # ── Split control ─────────────────────────────────────────────────────────
    parser.add_argument(
        "--no-split",
        action="store_true",
        help="Skip the train/test split step (use if already split).",
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--detect",
        action="store_true",
        help="Run face detection (slower). Default: treat images as pre-cropped.",
    )
    parser.add_argument(
        "--ctx",
        type=int,
        default=0,
        metavar="ID",
        help="GPU device id (default: 0). Use -1 for CPU.",
    )

    args = parser.parse_args()

    if not ENROLLED_DIR.exists() or not any(ENROLLED_DIR.iterdir()):
        print(f"ERROR: {ENROLLED_DIR} is empty or does not exist.")
        print("Run download_data.py first or place images in data/enrolled/<name>/")
        sys.exit(1)

    # ── Step 1: split ─────────────────────────────────────────────────────────
    if not args.no_split:
        reset_split = args.reset
        print(f"[split] {'Resetting and splitting' if reset_split else 'Splitting'} "
              f"enrolled images ({int((1-TEST_RATIO)*100)}/{int(TEST_RATIO*100)} train/test) ...")
        split_results = split_enrolled_images(
            enrolled_dir=ENROLLED_DIR,
            test_dir=ENROLLED_TEST_DIR,
            test_ratio=TEST_RATIO,
            seed=SEED,
            reset=reset_split,
        )
        total_train = sum(t for t, _ in split_results.values())
        total_test  = sum(e for _, e in split_results.values())
        print(f"[split] Done — {total_train} train images, {total_test} test images\n")

        if total_test < 500:
            print(f"  WARNING: only {total_test} test images — Task 1 requires >= 500.\n")
    else:
        print("[split] Skipped.\n")

    # ── Step 2: enroll ────────────────────────────────────────────────────────
    print("Loading model...")
    app = get_insightface_model(ctx_id=args.ctx)
    db  = EmbeddingDB()

    if args.reset:
        n_before = len(db)
        print(f"Resetting DB ({n_before} users)...")
        for uid in db.get_all_users():
            db.remove_user(uid)

    skip_existing = not (args.update or args.reset)
    mode_label    = "reset+full" if args.reset else ("update" if args.update else "incremental")
    detect_label  = "detect=on"  if args.detect else "detect=off (pre-cropped)"

    print(f"\nMode: {mode_label} | {detect_label}")
    print(f"Enrolling from: {ENROLLED_DIR}\n")

    results = enroll_from_folder(
        ENROLLED_DIR, app, db,
        detect=args.detect,
        skip_existing=skip_existing,
    )

    added   = sum(1 for n in results.values() if n > 0)
    skipped = sum(1 for n in results.values() if n == 0)

    print(
        f"\nDone.\n"
        f"  Added/updated : {added} users\n"
        f"  Skipped       : {skipped} users (already in DB or no face found)\n"
        f"  Total in DB   : {len(db)} users  ({db.count_embeddings()} embeddings)\n"
        f"  Test images   : {ENROLLED_TEST_DIR}"
    )

    if len(db) < 80:
        print(f"\nWARNING: only {len(db)} users in DB — project spec requires >= 80.")


if __name__ == "__main__":
    main()
