"""
Download face images of a person using Bing image search (via icrawler).

Usage:
    venv/Scripts/python lab/face/fetch_person.py "Ben Affleck" --count 50
    venv/Scripts/python lab/face/fetch_person.py "Elon Musk" --count 30 --out data/test
    venv/Scripts/python lab/face/fetch_person.py "Elon Musk" --no-validate  # skip face check

The person name is used as the search query and as the subfolder name
(spaces replaced with underscores). Images without a detectable face are
discarded unless --no-validate is passed.
"""
import argparse
import logging
import shutil
import sys
import tempfile
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent))

from src.model import get_insightface_model, get_embedding

DEFAULT_OUT = Path(__file__).parent / "data" / "enrolled"

logging.disable(logging.INFO)  # suppress INFO from all loggers globally


def _folder_name(person: str) -> str:
    return person.strip().replace(" ", "_")


def fetch(person: str, count: int, out_dir: Path, validate: bool) -> None:
    from icrawler.builtin import BingImageCrawler

    folder = out_dir / _folder_name(person)
    folder.mkdir(parents=True, exist_ok=True)

    existing = sorted(folder.glob("*.jpg"))
    start_idx = len(existing)
    if start_idx >= count:
        print(f"Already have {start_idx} images for '{person}', nothing to do.")
        return

    need = count - start_idx
    print(f"Fetching images for '{person}' (need {need} more, have {start_idx}) ...")

    app = None
    if validate:
        print("Loading model for face validation...")
        app = get_insightface_model(ctx_id=0)

    # Download to a temp dir so we can inspect before keeping
    with tempfile.TemporaryDirectory() as tmp:
        # Request more than needed to account for validation failures
        fetch_count = need * 3 if validate else need
        crawler = BingImageCrawler(
            storage={"root_dir": tmp},
            feeder_threads=1,
            parser_threads=2,
            downloader_threads=4,
        )
        # Redirect fd 2 (stderr) at OS level to silence icrawler's 401/403 errors.
        # sys.stderr redirect alone is not enough because icrawler's logging
        # handlers hold a direct reference to the original file descriptor.
        import os
        _null_fd = os.open(os.devnull, os.O_WRONLY)
        _saved_fd = os.dup(2)
        os.dup2(_null_fd, 2)
        try:
            crawler.crawl(
                keyword=f"{person} face portrait",
                max_num=fetch_count,
                min_size=(80, 80),
            )
        finally:
            os.dup2(_saved_fd, 2)
            os.close(_saved_fd)
            os.close(_null_fd)

        candidates = sorted(Path(tmp).glob("*.jpg"))
        print(f"Downloaded {len(candidates)} candidates, validating...")

        saved = 0
        for candidate in candidates:
            if saved >= need:
                break

            img = cv2.imread(str(candidate))
            if img is None:
                continue

            if validate:
                emb = get_embedding(app, img, fallback=False)
                if emb is None:
                    continue   # no face found — discard

            dest = folder / f"{start_idx + saved:04d}.jpg"
            shutil.copy2(candidate, dest)
            saved += 1
            print(f"  [{saved}/{need}] {dest.name}")

    total = len(sorted(folder.glob("*.jpg")))
    print(f"\nDone.  Saved {saved} new images.  Total in folder: {total}\n  -> {folder}")
    if saved < need:
        print(f"  WARNING: only {saved}/{need} saved — rerun or increase --count.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download face images via Bing.")
    parser.add_argument("person", help='Person name, e.g. "Ben Affleck"')
    parser.add_argument(
        "--count", "-n",
        type=int,
        default=30,
        help="Target number of images to save (default: 30).",
    )
    parser.add_argument(
        "--out", "-o",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output root directory (default: data/enrolled/).",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip face detection check — save all downloaded images.",
    )
    args = parser.parse_args()

    fetch(
        person=args.person,
        count=args.count,
        out_dir=args.out,
        validate=not args.no_validate,
    )


if __name__ == "__main__":
    main()
