"""
Sync ChromaDB embeddings with Google Drive.

The DB is stored on Drive as a single zip: chromadb.zip

Usage:
    venv/Scripts/python lab/face/sync_db.py --download   # fetch & extract
    venv/Scripts/python lab/face/sync_db.py --upload     # zip & push

OAuth2 setup (one-time):
    1. Go to https://console.cloud.google.com/
    2. Create a project -> APIs & Services -> Enable "Google Drive API"
    3. Credentials -> Create -> OAuth 2.0 Client ID -> Desktop app
    4. Download JSON -> save as  lab/face/client_secrets.json
    On first run the browser opens for consent; token is saved to token.json.
"""
import argparse
import shutil
import sys
import tomllib
import zipfile
from pathlib import Path

import gdown

# ── Config ────────────────────────────────────────────────────────────────────

BASE_DIR    = Path(__file__).parent
CONFIG_PATH = BASE_DIR / "config.toml"
SECRETS     = BASE_DIR / "client_secrets.json"
TOKEN       = BASE_DIR / "token.json"

with open(CONFIG_PATH, "rb") as f:
    cfg = tomllib.load(f)

FOLDER_ID   = cfg["data"]["gdrive_chromadb_folder_id"]
DB_DIR      = BASE_DIR / cfg["paths"]["db_dir"]         # embeddings/chromadb
ZIP_NAME    = "chromadb.zip"
ZIP_LOCAL   = BASE_DIR / "embeddings" / ZIP_NAME

SCOPES = ["https://www.googleapis.com/auth/drive"]

# ── Auth ──────────────────────────────────────────────────────────────────────

def _get_drive_service():
    """Return an authenticated Google Drive API service object."""
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build

    creds = None
    if TOKEN.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not SECRETS.exists():
                print(
                    "ERROR: client_secrets.json not found.\n"
                    "Follow the OAuth2 setup instructions at the top of this file."
                )
                sys.exit(1)
            flow = InstalledAppFlow.from_client_secrets_file(str(SECRETS), SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN, "w") as f:
            f.write(creds.to_json())

    from googleapiclient.discovery import build
    return build("drive", "v3", credentials=creds)

# ── Download ──────────────────────────────────────────────────────────────────

def download() -> None:
    print(f"[download] Searching for '{ZIP_NAME}' in Drive folder {FOLDER_ID} ...")

    # Use gdown to list files in the folder and find chromadb.zip
    import requests
    service = _get_drive_service()
    results = service.files().list(
        q=f"'{FOLDER_ID}' in parents and name='{ZIP_NAME}' and trashed=false",
        fields="files(id, name)",
    ).execute()

    files = results.get("files", [])
    if not files:
        print(f"[download] ERROR: '{ZIP_NAME}' not found in Drive folder.")
        sys.exit(1)

    file_id = files[0]["id"]
    ZIP_LOCAL.parent.mkdir(parents=True, exist_ok=True)

    print(f"[download] Downloading {ZIP_NAME} (file id: {file_id}) ...")
    gdown.download(id=file_id, output=str(ZIP_LOCAL), quiet=False)

    print(f"[download] Extracting to {DB_DIR.parent} ...")
    if DB_DIR.exists():
        shutil.rmtree(DB_DIR)

    with zipfile.ZipFile(ZIP_LOCAL, "r") as zf:
        zf.extractall(DB_DIR.parent)   # extracts chromadb/ into embeddings/

    ZIP_LOCAL.unlink(missing_ok=True)
    print(f"[download] Done. ChromaDB ready at {DB_DIR}")

# ── Upload ────────────────────────────────────────────────────────────────────

def upload() -> None:
    if not DB_DIR.exists() or not any(DB_DIR.rglob("*")):
        print(f"ERROR: {DB_DIR} is empty or missing. Run enroll.py first.")
        sys.exit(1)

    print(f"[upload] Zipping {DB_DIR} -> {ZIP_LOCAL} ...")
    ZIP_LOCAL.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_LOCAL, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in DB_DIR.rglob("*"):
            zf.write(file, file.relative_to(DB_DIR.parent))
    size_mb = ZIP_LOCAL.stat().st_size / 1024 / 1024
    print(f"[upload] Archive size: {size_mb:.1f} MB")

    from googleapiclient.http import MediaFileUpload

    service = _get_drive_service()

    # Check if chromadb.zip already exists in the folder
    results = service.files().list(
        q=f"'{FOLDER_ID}' in parents and name='{ZIP_NAME}' and trashed=false",
        fields="files(id, name)",
    ).execute()
    existing = results.get("files", [])

    media = MediaFileUpload(str(ZIP_LOCAL), mimetype="application/zip", resumable=True)

    if existing:
        file_id = existing[0]["id"]
        print(f"[upload] Updating existing file (id: {file_id}) ...")
        service.files().update(fileId=file_id, media_body=media).execute()
    else:
        print("[upload] Creating new file ...")
        service.files().create(
            body={"name": ZIP_NAME, "parents": [FOLDER_ID]},
            media_body=media,
        ).execute()

    ZIP_LOCAL.unlink(missing_ok=True)
    print("[upload] Done.")

# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Sync ChromaDB with Google Drive.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--download", action="store_true", help="Fetch chromadb.zip and extract.")
    group.add_argument("--upload",   action="store_true", help="Zip ChromaDB and push to Drive.")
    args = parser.parse_args()

    if args.download:
        download()
    else:
        upload()


if __name__ == "__main__":
    main()
