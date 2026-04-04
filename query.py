"""
Query the ChromaDB face database: who is this person?

Usage:
    # single image
    venv/Scripts/python lab/face/query.py path/to/photo.jpg

    # whole folder — runs on every image inside
    venv/Scripts/python lab/face/query.py path/to/folder/

    # adjust decision threshold (default 0.4)
    venv/Scripts/python lab/face/query.py photo.jpg --threshold 0.35
"""
import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent))

from src.authorization import identify
from src.database import EmbeddingDB
from src.model import get_insightface_model
from src.utils import list_images


def query_image(path: Path, app, db: EmbeddingDB, threshold: float) -> None:
    img = cv2.imread(str(path))
    if img is None:
        print(f"  [ERROR] Cannot read image: {path}")
        return

    user_id, score = identify(img, app, db, threshold=threshold)

    if user_id is None:
        status = "UNKNOWN"
    else:
        status = f"MATCH  -> {user_id}"

    print(f"  {path.name:<40}  score={score:.4f}  {status}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Query face database.")
    parser.add_argument("input", help="Image file or folder to query.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.4,
        help="Minimum cosine similarity to accept a match (default: 0.4).",
    )
    args = parser.parse_args()

    target = Path(args.input)
    if not target.exists():
        print(f"ERROR: {target} does not exist.")
        sys.exit(1)

    db = EmbeddingDB()
    if db.count_embeddings() == 0:
        print("ERROR: ChromaDB is empty. Run enroll.py first.")
        sys.exit(1)

    print("Loading model...")
    app = get_insightface_model(ctx_id=0)

    print(f"\nQuerying {db.count_embeddings()} embeddings ({len(db)} users), threshold={args.threshold}\n")

    if target.is_dir():
        images = list_images(target)
        if not images:
            print(f"No images found in {target}")
            sys.exit(1)
        for img_path in images:
            query_image(img_path, app, db, args.threshold)
    else:
        query_image(target, app, db, args.threshold)


if __name__ == "__main__":
    main()
