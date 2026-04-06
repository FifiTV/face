# P1 — Face Recognition & Authorization

Biometrics course project. Face-based user authorization system built on **InsightFace ArcFace** (ONNX runtime) with **ChromaDB** as the embedding vector store.

---

## Quick start

### 1. Install dependencies

```bash
venv/Scripts/pip install -r lab/face/requirements.txt
```

> Includes: `torch`, `torchvision`, `insightface`, `onnxruntime-gpu`, `chromadb`, `gdown`, `icrawler`, `google-api-python-client`, `google-auth-oauthlib`

### 2. Download the model

Downloads ArcFace weights (`buffalo_l`) to `lab/face/models/buffalo_l/`.

```bash
venv/Scripts/python lab/face/download_model.py
```

### 3. Download and prepare the dataset

Downloads FaceScrub from Google Drive, extracts the archives, and splits identities into four **mutually disjoint** sets (by person, not by image).

```bash
venv/Scripts/python lab/face/download_data.py
```

Split is controlled by `lab/face/config.toml`:

| Key | Default | Description |
|---|---|---|
| `seed` | `44` | Random seed for the split |
| `n_enrolled` | `100` | People added to the enrollment DB (≥ 80 required) |
| `train_ratio` | `0.70` | Share of remaining people used for training |
| `val_ratio` | `0.20` | Share used for validation |
| `test_ratio` | `0.10` | Share used for testing |

Skip steps you've already done:

```bash
venv/Scripts/python lab/face/download_data.py --skip-download   # zips already in data/raw/
venv/Scripts/python lab/face/download_data.py --skip-download --skip-extract  # only re-split
```

### 4. Split + enroll into ChromaDB

`enroll.py` does both steps automatically:

1. **Split** — each person's images in `data/enrolled/` are split per project spec  
   *("tests must use photos not used for enrollment")*
2. **Enroll** — averaged embedding per person is stored in ChromaDB

| Folder | Share | Purpose |
|---|---|---|
| `data/enrolled/` | 70% | Building embeddings (ChromaDB) |
| `data/enrolled_test/` | 30% | Genuine-user tests (Task 1, 3, 4, 7) |

```bash
venv/Scripts/python lab/face/enroll.py              # split + enroll (default)
venv/Scripts/python lab/face/enroll.py --no-split   # skip split (already done)
venv/Scripts/python lab/face/enroll.py --update     # re-enroll all, keep split
venv/Scripts/python lab/face/enroll.py --reset      # wipe DB + redo split + enroll from scratch
venv/Scripts/python lab/face/enroll.py --detect     # use face detector (for raw photos)
```

Split ratio is set by `enrolled_split.test_ratio` in `config.toml` (default `0.30`).  
To only split without enrolling: `venv/Scripts/python lab/face/split_enrolled.py`

### 5. Query the database

```bash
venv/Scripts/python lab/face/query.py photo.jpg             # single image
venv/Scripts/python lab/face/query.py data/test/            # whole folder
venv/Scripts/python lab/face/query.py photo.jpg --threshold 0.35
```

Output example:
```
Querying 99 embeddings (99 users), threshold=0.4

  BenAffleck_test.jpg     score=0.6703  MATCH  -> Ben_Affleck
  unknown.jpg             score=0.2145  UNKNOWN
```

---

## Collecting extra face images

Download images of any person from Bing image search. Images without a detectable face are discarded automatically.

```bash
venv/Scripts/python lab/face/fetch_person.py "Ben Affleck" --count 50
venv/Scripts/python lab/face/fetch_person.py "Elon Musk"   --count 30 --out data/test
venv/Scripts/python lab/face/fetch_person.py "Anna Nowak"  --count 20 --no-validate
```

Images are saved to `data/enrolled/<Name_Surname>/` by default (resumable — reruns only fetch what's missing).

---

## Importing faces from an external folder

If you already have a folder of face images organised as `<root>/<person_name>/img.jpg`, import them into `data/enrolled/` and run the split in one step:

```bash
venv/Scripts/python lab/face/import_faces.py /path/to/faces
venv/Scripts/python lab/face/import_faces.py /path/to/faces --move          # move instead of copy
venv/Scripts/python lab/face/import_faces.py /path/to/faces --validate      # discard images without a face
venv/Scripts/python lab/face/import_faces.py /path/to/faces --no-split      # skip split step
venv/Scripts/python lab/face/import_faces.py /path/to/faces --out data/test # custom destination
venv/Scripts/python lab/face/import_faces.py /path/to/faces --dry-run       # preview only
```

This is an **optional step** between `fetch_person.py` and `enroll.py`. It merges the source subfolders into `data/enrolled/` (renaming files to sequential indices to avoid collisions) and then re-runs the 70/30 split.

---

## Syncing ChromaDB with Google Drive

The enrollment DB is shared between team members as `chromadb.zip` on Google Drive.

```bash
venv/Scripts/python lab/face/sync_db.py --download   # fetch & extract
venv/Scripts/python lab/face/sync_db.py --upload     # zip & push
```

**One-time OAuth2 setup** (needed only for `--upload`):
1. [Google Cloud Console](https://console.cloud.google.com/) → Create project
2. APIs & Services → Enable **Google Drive API**
3. Credentials → Create → **OAuth 2.0 Client ID** → Desktop app
4. Download JSON → save as `lab/face/client_secrets.json`

On first `--upload` the browser opens for consent; token is saved to `token.json` for future runs. Both files are git-ignored.

---

## Data storage

Face embeddings (512-d ArcFace vectors) are stored in a **ChromaDB** persistent vector database at `embeddings/chromadb/`. Raw images are **never** stored — only embeddings (project spec requirement).

ChromaDB uses an **HNSW index with cosine distance** for fast 1:N nearest-neighbour lookup.  
Each enrolled user has **one averaged embedding** (mean of all per-image embeddings, re-normalised to unit length).

---

## Project layout

```
lab/face/
├── config.toml           # seed, split ratios, GDrive IDs, paths
├── download_model.py     # fetch ArcFace weights -> models/buffalo_l/
├── download_data.py      # fetch & split FaceScrub
├── split_enrolled.py     # split enrolled/ -> enrolled/ (70%) + enrolled_test/ (30%)
├── enroll.py             # build ChromaDB from data/enrolled/
├── query.py              # identify a face against the DB
├── fetch_person.py       # download extra images via Bing search
├── sync_db.py            # push/pull ChromaDB to/from Google Drive
├── models/buffalo_l/     # ONNX weights — git-ignored (download_model.py)
├── data/
│   ├── raw/              # original downloaded files
│   ├── train/            # training identities
│   ├── val/              # validation identities
│   ├── test/             # test identities
│   ├── enrolled/         # images used to BUILD embeddings (70% per person)
│   └── enrolled_test/    # held-out images for testing genuine users (30%)
├── embeddings/chromadb/  # ChromaDB vector store — git-ignored (sync_db.py)
├── src/
│   ├── model.py          # InsightFace wrapper + fallback crop embedding
│   ├── database.py       # EmbeddingDB (ChromaDB backend)
│   ├── enrollment.py     # averaged enrollment, enroll_from_folder()
│   ├── authorization.py  # verify (1:1) and identify (1:N)
│   ├── metrics.py        # FAR, FRR, ROC, EER
│   ├── degradation.py    # noise (PSNR), luminance YCbCr, JPEG transforms
│   └── utils.py          # load_image(), psnr(), list_images()
└── notebooks/
    ├── 00_setup.ipynb
    ├── 01_enrollment.ipynb
    └── 02–08_task*.ipynb  # one per experiment (Tasks 1–7)
```
