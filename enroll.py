"""
Build the ChromaDB enrollment database from data/enrolled/.

For each person subfolder, all face images are processed, embeddings are
averaged and L2-normalised, and a single vector is stored in ChromaDB.

Usage:
    venv/Scripts/python lab/face/enroll.py [options]

Modes (mutually exclusive, default = incremental):
    (default)   Only add people not yet in the DB. Existing entries untouched.
    --update    Re-compute and overwrite embeddings for everyone in enrolled/.
    --reset     Wipe the entire DB first, then enroll everything from scratch.

Detection:
    (default)   No face detection — treats every image as a pre-cropped face
                (correct for FaceScrub; fast and reliable).
    --detect    Run RetinaFace detector first; images where no face is found
                are excluded from the average (slower, better for raw photos).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.database import EmbeddingDB
from src.enrollment import enroll_from_folder
from src.model import get_insightface_model

ENROLLED_DIR = Path(__file__).parent / "data" / "enrolled"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enroll faces into ChromaDB.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--update",
        action="store_true",
        help="Re-enroll everyone, overwriting existing DB entries.",
    )
    mode.add_argument(
        "--reset",
        action="store_true",
        help="Wipe the entire DB, then enroll everything from scratch.",
    )

    parser.add_argument(
        "--detect",
        action="store_true",
        help="Run face detection (slower). Default: treat images as pre-cropped faces.",
    )
    parser.add_argument(
        "--ctx",
        type=int,
        default=0,
        metavar="ID",
        help="GPU device id for InsightFace (default: 0). Use -1 for CPU.",
    )

    args = parser.parse_args()

    if not ENROLLED_DIR.exists() or not any(ENROLLED_DIR.iterdir()):
        print(f"ERROR: {ENROLLED_DIR} is empty or does not exist.")
        print("Run download_data.py first or place images in data/enrolled/<name>/")
        sys.exit(1)

    print("Loading model...")
    app = get_insightface_model(ctx_id=args.ctx)
    db = EmbeddingDB()

    if args.reset:
        n_before = len(db)
        print(f"Resetting DB ({n_before} users)...")
        for uid in db.get_all_users():
            db.remove_user(uid)

    skip_existing = not (args.update or args.reset)
    mode_label = "reset+full" if args.reset else ("update" if args.update else "incremental")
    detect_label = "detect=on" if args.detect else "detect=off (pre-cropped)"

    print(f"\nMode: {mode_label} | {detect_label}")
    print(f"Enrolling from: {ENROLLED_DIR}\n")

    results = enroll_from_folder(
        ENROLLED_DIR,
        app,
        db,
        detect=args.detect,
        skip_existing=skip_existing,
    )

    added   = sum(1 for n in results.values() if n > 0)
    skipped = sum(1 for uid, n in results.items() if n == 0)

    print(
        f"\nDone.\n"
        f"  Added/updated : {added} users\n"
        f"  Skipped       : {skipped} users (already in DB or no face found)\n"
        f"  Total in DB   : {len(db)} users  ({db.count_embeddings()} embeddings)"
    )

    if len(db) < 80:
        print(f"\nWARNING: only {len(db)} users in DB — project spec requires >= 80.")


if __name__ == "__main__":
    main()
