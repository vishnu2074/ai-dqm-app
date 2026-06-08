"""
python-backend/app/database.py
Hybrid database layer — persistent disk primary, Azure Blob backup.

v4.0 — Fixed data-loss-on-redeploy:
  1. Primary storage: /var/data/ai-dqm/ (Render persistent disk).
     Falls back to /tmp/ai-dqm/ for local dev.
  2. Blob download: skipped if a valid local DB already exists (persistent disk hit).
     Only downloads when starting fresh (first deploy or new machine).
  3. Blob upload: checkpoints WAL before reading the file, so the upload always
     captures every committed write (previously stale .db file was uploaded).
  4. Periodic backup default lowered 300s → 60s.
  5. Shutdown upload is triggered from main.py via @app.on_event("shutdown").
"""
from __future__ import annotations
import os
import stat
import threading
import time
from pathlib import Path
from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker


# ── Path selection ────────────────────────────────────────────────────────────
# Render attaches a persistent disk at /var/data — survives redeploys and
# container restarts on the same service.  Fall back to /tmp for local dev.

def _choose_db_dir() -> Path:
    render_dir = Path("/var/data/ai-dqm")
    tmp_dir    = Path("/tmp/ai-dqm")
    try:
        if Path("/var/data").exists():
            render_dir.mkdir(parents=True, exist_ok=True)
            print(f"[database] Using persistent disk: {render_dir}")
            return render_dir
    except Exception as e:
        print(f"[database] /var/data not writable ({e}), falling back to /tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    return tmp_dir


_DB_DIR   = _choose_db_dir()
_DB_PATH  = _DB_DIR / "ai_dqm.db"
_AGENT_DB_PATH = _DB_DIR / "agent_history.db"

os.environ.setdefault("DATABASE_URL", f"sqlite:///{_DB_PATH}")
os.environ.setdefault("AGENT_HISTORY_DB", str(_AGENT_DB_PATH))

_SQLITE_URL = f"sqlite:///{_DB_PATH}"

# ── Azure Blob — backup / disaster-recovery ───────────────────────────────────
_CONN_STR  = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
_CONTAINER = os.getenv("AZURE_STORAGE_CONTAINER", "intern26")
_BLOB_NAME = "ai-dqm/ai_dqm.db"


def _blob_client():
    if not _CONN_STR:
        return None
    try:
        from azure.storage.blob import BlobClient
        return BlobClient.from_connection_string(
            conn_str=_CONN_STR,
            container_name=_CONTAINER,
            blob_name=_BLOB_NAME,
        )
    except Exception as e:
        print(f"[database] Azure Blob client error: {e}")
        return None


def _validate_sqlite_db(db_path: Path) -> bool:
    """
    Validate that a file is a valid, non-trivially-empty SQLite database.
    Requires at least 4 KB (one full page) to rule out header-only blobs.
    """
    if not db_path.exists():
        return False
    file_size = db_path.stat().st_size
    if file_size < 4096:
        print(f"[database] DB file too small ({file_size} bytes) — treating as empty")
        return False
    try:
        import sqlite3
        conn   = sqlite3.connect(str(db_path))
        result = conn.execute("SELECT sqlite_version()").fetchone()
        tables = conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table'"
        ).fetchone()
        conn.close()
        has_tables = tables and tables[0] > 0
        if has_tables:
            print(f"[database] ✓ DB valid — SQLite {result[0]}, {file_size:,} bytes")
        else:
            print(f"[database] DB has no tables yet ({file_size} bytes) — will recreate schema")
        return True
    except Exception as e:
        print(f"[database] DB validation failed: {e}")
        return False


def _fix_db_permissions(db_path: Path):
    """Ensure the DB file and directory have correct write permissions."""
    try:
        os.chmod(str(db_path.parent), stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
        if db_path.exists():
            os.chmod(
                str(db_path),
                stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP
            )
            print(f"[database] ✓ Fixed permissions on {db_path}")

        for suffix in ["-wal", "-shm", "-journal"]:
            p = Path(str(db_path) + suffix)
            if p.exists():
                os.chmod(
                    str(p),
                    stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP
                )
    except Exception as e:
        print(f"[database] Warning: could not fix permissions: {e}")


def _checkpoint_wal(db_path: Path):
    """
    Force SQLite WAL checkpoint so the main .db file is fully up-to-date
    before we read its bytes for an upload.
    """
    try:
        import sqlite3
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA wal_checkpoint(FULL)")
        conn.close()
    except Exception:
        pass


def download_db_from_blob() -> bool:
    """
    Restore ai_dqm.db from Azure Blob on startup.
    """
    if _DB_PATH.exists() and _validate_sqlite_db(_DB_PATH):
        size_kb = _DB_PATH.stat().st_size // 1024
        print(f"[database] ✓ Using existing local DB at {_DB_PATH} ({size_kb} KB)")
        _fix_db_permissions(_DB_PATH)
        return True

    client = _blob_client()
    if client is None:
        print("[database] No Azure Blob config — starting fresh")
        return False

    try:
        temp_path = _DB_PATH.with_suffix(".db.tmp")

        with open(temp_path, "wb") as f:
            client.download_blob().readinto(f)

        if _validate_sqlite_db(temp_path):
            blob_size = temp_path.stat().st_size
            local_size = _DB_PATH.stat().st_size if _DB_PATH.exists() else 0

            if blob_size > local_size:
                if _DB_PATH.exists():
                    _DB_PATH.unlink()

                temp_path.rename(_DB_PATH)
                _fix_db_permissions(_DB_PATH)

                print(
                    f"[database] ✓ Restored from Azure Blob "
                    f"({blob_size // 1024} KB)"
                )
                return True
            else:
                temp_path.unlink()
                print(
                    "[database] Local DB is same size or larger than blob "
                    "— keeping local"
                )
                return True

        else:
            if temp_path.exists():
                temp_path.unlink()

            print("[database] Blob DB invalid or empty — starting fresh")
            return False

    except Exception as e:
        print(
            f"[database] Blob download failed ({type(e).__name__}) "
            "— starting fresh"
        )

        temp_path = _DB_PATH.with_suffix(".db.tmp")
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass

        return False


def upload_db_to_blob() -> bool:
    """
    Upload ai_dqm.db to Azure Blob.
    Checkpoints WAL first so the uploaded file contains ALL committed writes.
    """
    if not _DB_PATH.exists():
        return False

    client = _blob_client()
    if client is None:
        return False

    _checkpoint_wal(_DB_PATH)

    try:
        with open(_DB_PATH, "rb") as f:
            client.upload_blob(f, overwrite=True)

        size_kb = _DB_PATH.stat().st_size // 1024
        print(f"[database] ✓ Backed up ai_dqm.db to Azure Blob ({size_kb} KB)")
        return True

    except Exception as e:
        print(f"[database] Blob upload failed: {e}")
        return False


def _periodic_backup_loop(interval: int):
    while True:
        time.sleep(interval)

        try:
            upload_db_to_blob()
        except Exception as e:
            print(f"[database] Periodic backup error: {e}")


def start_periodic_backup(interval_seconds: int = 60):
    """
    Start daemon thread that uploads DB every interval_seconds.
    """
    t = threading.Thread(
        target=_periodic_backup_loop,
        args=(interval_seconds,),
        daemon=True,
        name="db-blob-backup",
    )

    t.start()

    print(
        f"[database] Periodic DB backup every "
        f"{interval_seconds}s → Azure Blob"
    )


download_db_from_blob()

engine = create_engine(
    _SQLITE_URL,
    connect_args={"check_same_thread": False},
    pool_pre_ping=True,
    pool_recycle=1800,
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

Base = declarative_base()


def get_db():
    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()


def seed_governance_data():
    import uuid

    try:
        from app.models import (
            GovernanceNotification,
            NotificationPreference,
        )
    except ImportError as e:
        print(f"Governance seed skipped: {e}")
        return

    db = SessionLocal()

    try:
        notif_defaults = [
            ("Quality Score Alerts", "Get notified when quality scores drop below threshold", "email", True),
            ("Anomaly Detection", "Receive alerts for detected data anomalies", "in_app", True),
            ("Rule Failures", "Get notified when data quality rules fail", "email", True),
            ("Daily Summary", "Receive daily summary of data quality metrics", "email", False),
            ("Weekly Reports", "Get weekly data quality reports via email", "email", True),
            ("Schema Changes", "Alert when schema changes are detected", "in_app", True),
            ("New Data Sources", "Notify when new data sources are connected", "slack", False),
            ("Compliance Violations", "Immediate alerts for compliance policy violations", "email", True),
        ]

        for title, desc, channel, enabled in notif_defaults:
            if not db.query(GovernanceNotification).filter_by(title=title).first():
                db.add(
                    GovernanceNotification(
                        id=str(uuid.uuid4()),
                        title=title,
                        description=desc,
                        channel=channel,
                        enabled=enabled,
                    )
                )

        pref_defaults = [
            ("Quality Score Alerts", "email", True),
            ("Anomaly Detection", "in_app", True),
            ("Rule Failures", "email", True),
            ("Daily Summary", "email", False),
            ("Weekly Reports", "email", True),
            ("Schema Changes", "in_app", True),
            ("New Data Sources", "slack", False),
            ("Compliance Violations", "email", True),
        ]

        for event_type, channel, enabled in pref_defaults:
            if not db.query(NotificationPreference).filter_by(
                event_type=event_type,
                user_email=None,
            ).first():
                db.add(
                    NotificationPreference(
                        user_email=None,
                        event_type=event_type,
                        channel=channel,
                        enabled=enabled,
                    )
                )

        db.commit()
        print("Governance seed data applied.")

    except Exception as e:
        db.rollback()
        print(f"Governance seed failed (non-fatal): {e}")

    finally:
        db.close()