# P1 — Face Recognition & Authorization

Biometrics course project. Face-based user authorization system built on **InsightFace ArcFace** (ONNX runtime, no TensorFlow) with **ChromaDB** as the embedding vector store.

---

## Quick start

### 1. Install dependencies

```bash
venv/Scripts/pip install -r lab/face/requirements.txt
```

### 2. Download the model

Downloads ArcFace weights to `lab/face/models/buffalo_l/`.

```bash
venv/Scripts/python lab/face/download_model.py
```

### 3. Download and prepare the dataset

Downloads FaceScrub from Google Drive, extracts the archives, and splits identities into four **mutually disjoint** sets.

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

If you already have the data downloaded or extracted, skip those steps:

```bash
# zips already in data/raw/ — skip download
venv/Scripts/python lab/face/download_data.py --skip-download

# images already extracted — only re-split
venv/Scripts/python lab/face/download_data.py --skip-download --skip-extract
```

---

## Syncing ChromaDB with Google Drive

The enrollment DB can be shared between team members via Google Drive as `chromadb.zip`.

```bash
# fetch the DB (no setup needed — uses gdown + OAuth)
venv/Scripts/python lab/face/sync_db.py --download

# push your local DB after enrollment
venv/Scripts/python lab/face/sync_db.py --upload
```

**One-time OAuth2 setup** (needed only for `--upload`):
1. [Google Cloud Console](https://console.cloud.google.com/) → Create project
2. APIs & Services → Enable **Google Drive API**
3. Credentials → Create → **OAuth 2.0 Client ID** → Desktop app
4. Download JSON → save as `lab/face/client_secrets.json`

On first `--upload` run the browser opens for consent; the token is saved to `token.json` for future runs. Add both files to `.gitignore`.

---

## Data storage

Face embeddings (512-d ArcFace vectors) are stored in a **ChromaDB** persistent vector database at `embeddings/chromadb/`. Raw images are **never** stored in the database — only the embeddings (project spec requirement).

ChromaDB uses an HNSW index with cosine distance, which enables fast approximate nearest-neighbour lookup for 1:N identification without scanning every stored embedding linearly.

---

## Project layout

```
lab/face/
├── config.toml          # seed, split ratios, paths
├── download_model.py    # fetch ArcFace weights
├── download_data.py     # fetch & split FaceScrub
├── models/buffalo_l/    # ONNX model weights (git-ignored)
├── data/
│   ├── raw/             # original downloaded files
│   ├── train/           # training identities
│   ├── val/             # validation identities
│   ├── test/            # test identities
│   └── enrolled/        # identities in the authorization DB
├── embeddings/chromadb/ # ChromaDB vector store (git-ignored)
├── src/
│   ├── model.py         # InsightFace wrapper, get_embedding()
│   ├── database.py      # EmbeddingDB (ChromaDB backend)
│   ├── enrollment.py    # batch enrollment from folder
│   ├── authorization.py # verify (1:1) and identify (1:N)
│   ├── metrics.py       # FAR, FRR, ROC, EER
│   ├── degradation.py   # noise, luminance, JPEG transforms
│   └── utils.py         # image I/O helpers, PSNR
└── notebooks/           # one notebook per experiment (Task 1–7)
```
