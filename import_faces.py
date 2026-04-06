"""
Import face images from an external folder into data/enrolled/ and optionally split them.

This is an optional step to be run between fetch_person.py and enroll.py when
you have images stored elsewhere (another dataset, manually collected photos, etc.).

Expected source structure (same as enrolled/):
    <src>/
        Alice_Smith/
            img1.jpg
            img2.jpg
        Bob_Jones/
            img1.jpg

Usage:
    venv/Scripts/python lab/face/import_faces.py /path/to/faces
    venv/Scripts/python lab/face/import_faces.py /path/to/faces --move           # move instead of copy
    venv/Scripts/python lab/face/import_faces.py /path/to/faces --no-split       # skip split step
    venv/Scripts/python lab/face/import_faces.py /path/to/faces --validate       # discard images without a face
    venv/Scripts/python lab/face/import_faces.py /path/to/faces --crop           # detect & save only the face crop
    venv/Scripts/python lab/face/import_faces.py /path/to/faces --crop --pad 0.3 # larger padding around face
    venv/Scripts/python lab/face/import_faces.py /path/to/faces --out data/test  # custom destination
    venv/Scripts/python lab/face/import_faces.py /path/to/faces --dry-run        # preview only

Notes on --crop:
  - Detects all faces in the image, saves each one as a separate file.
  - A margin (--pad, default 0.2 = 20% of face size) is added on each side.
  - Images where no face is found are silently discarded.
  - When multiple faces are detected, all are saved (useful for group photos
    if each subfolder represents a single person you want to add more crops of).
"""
import argparse
import shutil
import sys
import tomllib
import warnings
from pathlib import Path

import cv2
import numpy as np

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

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _is_image(p: Path) -> bool:
    return p.suffix.lower() in IMAGE_EXTS


def _detect_and_crop(app, img_bgr: np.ndarray, pad: float) -> list[np.ndarray]:
    """
    Detect all faces in img_bgr and return a list of cropped face images.

    Each crop is padded by `pad` fraction of the face bounding-box size on
    each side (e.g. pad=0.2 adds 20% extra margin) and clamped to image bounds.
    Returns an empty list if no face is detected.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        faces = app.get(img_bgr)

    if not faces:
        return []

    h, w = img_bgr.shape[:2]
    crops = []
    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int)
        bw, bh = x2 - x1, y2 - y1
        pad_x = int(bw * pad)
        pad_y = int(bh * pad)
        cx1 = max(0, x1 - pad_x)
        cy1 = max(0, y1 - pad_y)
        cx2 = min(w, x2 + pad_x)
        cy2 = min(h, y2 + pad_y)
        crop = img_bgr[cy1:cy2, cx1:cx2]
        if crop.size > 0:
            crops.append(crop)
    return crops


def import_person(
    src_dir: Path,
    dst_root: Path,
    move: bool,
    validate: bool,
    crop: bool,
    pad: float,
    app,
    dry_run: bool,
) -> tuple[int, int]:
    """
    Import images from src_dir into dst_root/<src_dir.name>/.

    When crop=True: detect faces, save each face crop as a separate JPEG.
    When validate=True (and crop=False): discard images with no detected face.
    When neither: copy/move files as-is.

    Returns:
        (n_saved, n_skipped)
    """
    from src.model import get_embedding

    person_name = src_dir.name
    dst_dir = dst_root / person_name
    images = sorted(p for p in src_dir.iterdir() if p.is_file() and _is_image(p))

    if not images:
        print(f"  {person_name}: no images found, skipping.")
        return 0, 0

    if dry_run:
        mode = "crop+save faces" if crop else ("copy+validate" if validate else "copy")
        print(f"  {person_name}: {len(images)} images  [{mode}]  -> {dst_dir}")
        return len(images), 0

    dst_dir.mkdir(parents=True, exist_ok=True)
    start_idx = len(list_images(dst_dir))

    saved   = 0
    skipped = 0

    for src_path in images:
        img = cv2.imread(str(src_path))
        if img is None:
            skipped += 1
            continue

        if crop:
            # Detect all faces and save each crop separately
            crops = _detect_and_crop(app, img, pad=pad)
            if not crops:
                skipped += 1
                continue
            for face_crop in crops:
                dst_path = dst_dir / f"{start_idx + saved:04d}.jpg"
                cv2.imwrite(str(dst_path), face_crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
                saved += 1

        elif validate:
            emb = get_embedding(app, img, fallback=False)
            if emb is None:
                skipped += 1
                continue
            dst_path = dst_dir / f"{start_idx + saved:04d}{src_path.suffix.lower()}"
            if move:
                shutil.move(str(src_path), dst_path)
            else:
                shutil.copy2(src_path, dst_path)
            saved += 1

        else:
            dst_path = dst_dir / f"{start_idx + saved:04d}{src_path.suffix.lower()}"
            if move:
                shutil.move(str(src_path), dst_path)
            else:
                shutil.copy2(src_path, dst_path)
            saved += 1

        # In move+crop mode the source is no longer needed after processing
        if crop and move and src_path.exists():
            src_path.unlink()

    return saved, skipped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import face images from an external folder into data/enrolled/.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "src",
        type=Path,
        help="Source folder containing per-person subfolders.",
    )
    parser.add_argument(
        "--out", "-o",
        type=Path,
        default=ENROLLED_DIR,
        help=f"Destination root (default: {ENROLLED_DIR}).",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move files instead of copying them.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run face detection and discard images without a detectable face "
             "(keeps the original full image, unlike --crop).",
    )
    parser.add_argument(
        "--crop",
        action="store_true",
        help="Detect faces, crop them out and save only the face region. "
             "Images with no face are discarded. Multiple faces per image "
             "each become a separate file.",
    )
    parser.add_argument(
        "--pad",
        type=float,
        default=0.2,
        metavar="FRAC",
        help="Padding added around each face crop as a fraction of the "
             "bounding-box size (default: 0.2 = 20%%). Only used with --crop.",
    )
    parser.add_argument(
        "--no-split",
        action="store_true",
        help="Skip the train/test split step after import.",
    )
    parser.add_argument(
        "--reset-split",
        action="store_true",
        help="Re-do the enrolled split for all persons (same as split_enrolled.py --reset).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be imported without touching any files.",
    )
    parser.add_argument(
        "--ctx",
        type=int,
        default=0,
        metavar="ID",
        help="GPU device id (default: 0). Use -1 for CPU.",
    )
    args = parser.parse_args()

    if args.crop and args.validate:
        print("ERROR: --crop and --validate are mutually exclusive. "
              "--crop already discards images with no face.")
        sys.exit(1)

    src_root: Path = args.src.resolve()
    dst_root: Path = args.out

    if not src_root.exists():
        print(f"ERROR: source folder does not exist: {src_root}")
        sys.exit(1)

    person_dirs = sorted(d for d in src_root.iterdir() if d.is_dir())
    if not person_dirs:
        print(f"ERROR: no subfolders found in {src_root}")
        sys.exit(1)

    needs_model = (args.crop or args.validate) and not args.dry_run
    action = "move" if args.move else "copy"
    mode   = "crop" if args.crop else ("validate" if args.validate else "as-is")
    print(f"Source      : {src_root}")
    print(f"Destination : {dst_root}")
    print(f"Action      : {action}  |  mode={mode}"
          + (f"  pad={args.pad}" if args.crop else "")
          + f"  |  split={'off' if args.no_split else 'on'}")
    if args.dry_run:
        print("[DRY RUN — no files will be modified]\n")

    app = None
    if needs_model:
        print("Loading face detection model...")
        from src.model import get_insightface_model
        app = get_insightface_model(ctx_id=args.ctx)
        print()

    total_saved   = 0
    total_skipped = 0

    for person_dir in person_dirs:
        n_saved, n_skip = import_person(
            src_dir=person_dir,
            dst_root=dst_root,
            move=args.move,
            validate=args.validate,
            crop=args.crop,
            pad=args.pad,
            app=app,
            dry_run=args.dry_run,
        )
        total_saved   += n_saved
        total_skipped += n_skip
        if not args.dry_run and (n_saved > 0 or n_skip > 0):
            suffix = f"  ({n_skip} discarded)" if n_skip else ""
            print(f"  {person_dir.name}: {n_saved} saved{suffix}")

    print(f"\nImport done.  {total_saved} files saved,  {total_skipped} discarded.")

    if args.dry_run or args.no_split:
        if args.no_split:
            print("Split skipped (--no-split).")
        return

    # ── Split ─────────────────────────────────────────────────────────────────
    if dst_root.resolve() != ENROLLED_DIR.resolve():
        print(f"\nNot splitting: destination is not the default enrolled dir ({ENROLLED_DIR}).")
        return

    print(f"\nSplitting enrolled images "
          f"({int((1-TEST_RATIO)*100)}/{int(TEST_RATIO*100)} train/test) ...")
    split_results = split_enrolled_images(
        enrolled_dir=ENROLLED_DIR,
        test_dir=ENROLLED_TEST_DIR,
        test_ratio=TEST_RATIO,
        seed=SEED,
        reset=args.reset_split,
    )
    total_train = sum(t for t, _ in split_results.values())
    total_test  = sum(e for _, e in split_results.values())
    print(
        f"Split done.  {total_train} train images -> {ENROLLED_DIR}\n"
        f"             {total_test} test images  -> {ENROLLED_TEST_DIR}"
    )
    if total_test < 500:
        print(f"  WARNING: {total_test} test images — Task 1 requires >= 500.")


if __name__ == "__main__":
    main()
