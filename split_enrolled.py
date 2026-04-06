"""
Split enrolled images into enrollment set and test set (standalone script).

Delegates to src.enrollment.split_enrolled_images — the same function used
by enroll.py. Run this directly only if you want to split without enrolling.

Usage:
    venv/Scripts/python lab/face/split_enrolled.py
    venv/Scripts/python lab/face/split_enrolled.py --reset    # redo all splits
    venv/Scripts/python lab/face/split_enrolled.py --dry-run  # preview only
"""
import argparse
import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.enrollment import split_enrolled_images
from src.utils import list_images

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
        description="Split enrolled images into train/test sets."
    )
    parser.add_argument("--reset",   action="store_true", help="Re-do all splits.")
    parser.add_argument("--dry-run", action="store_true", help="Preview without moving files.")
    args = parser.parse_args()

    if not ENROLLED_DIR.exists():
        print(f"ERROR: {ENROLLED_DIR} does not exist.")
        sys.exit(1)

    if args.dry_run:
        print(f"[DRY RUN] seed={SEED}, test_ratio={TEST_RATIO}")
        person_dirs = [d for d in ENROLLED_DIR.iterdir() if d.is_dir()]
        for pd in sorted(person_dirs):
            imgs = list_images(pd)
            n_test = max(1, round(len(imgs) * TEST_RATIO))
            print(f"  {pd.name:<40} train={len(imgs)-n_test}  test={n_test}")
        return

    results = split_enrolled_images(
        enrolled_dir=ENROLLED_DIR,
        test_dir=ENROLLED_TEST_DIR,
        test_ratio=TEST_RATIO,
        seed=SEED,
        reset=args.reset,
    )

    skipped     = sum(1 for t, e in results.values() if e > 0 and not args.reset)
    total_train = sum(t for t, _ in results.values())
    total_test  = sum(e for _, e in results.values())

    print(
        f"\nDone.  seed={SEED}  test_ratio={TEST_RATIO}\n"
        f"  Persons   : {len(results)}\n"
        f"  Train imgs: {total_train}  -> {ENROLLED_DIR}\n"
        f"  Test  imgs: {total_test}   -> {ENROLLED_TEST_DIR}"
    )
    if total_test < 500:
        print(f"\n  WARNING: {total_test} test images — Task 1 requires >= 500.")


if __name__ == "__main__":
    main()
