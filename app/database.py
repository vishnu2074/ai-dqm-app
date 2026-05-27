# python-backend/app/database.py
"""
Hybrid database layer.

PRIMARY:   SQLAlchemy ORM → SQLite at /tmp/ai-dqm/ai_dqm.db
           /tmp is always writable in Databricks Apps containers.
           DBFS (/dbfs/) is NOT available in Apps — do not use it.

PERSISTENCE across restarts:
           On startup  → download ai_dqm.db from Azure Blob Storage
                         (container: intern26, blob: ai-dqm/ai_dqm.db)
           Periodic    → background thread uploads every 5 minutes
           On write    → upload triggered automatically after profiling runs
                         (call upload_db_to_blob() after heavy writes)

SECONDARY: Delta raw connector (database_delta) mirrors writes to Unity Catalog.
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# ── Paths ─────────────────────────────────────────────────────────────────────
_DB_DIR  = Path("/tmp/ai-dqm")
_DB_PATH = _DB_DIR / "ai_dqm.db"
_DB_DIR.mkdir(parents=True, exist_ok=True)

_AGENT_DB_PATH = _DB_DIR / "agent_history.db"

# Expose paths so any code reading os.getenv() gets the right value
os.environ.setdefault("DATABASE_URL",    f"sqlite:///{_DB_PATH}")
os.environ.setdefault("AGENT_HISTORY_DB", str(_AGENT_DB_PATH))

_SQLITE_URL = f"sqlite:///{_DB_PATH}"

# ── Azure Blob persistence ────────────────────────────────────────────────────
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


def download_db_from_blob() -> bool:
    """Download ai_dqm.db from Azure Blob on startup. Returns True on success."""
    client = _blob_client()
    if client is None:
        print("[database] No Azure Blob config — starting with fresh local DB")
        return False
    try:
        with open(_DB_PATH, "wb") as f:
            client.download_blob().readinto(f)
        size_kb = _DB_PATH.stat().st_size // 1024
        print(f"[database] ✓ Restored ai_dqm.db from Azure Blob ({size_kb} KB) — all data intact")
        return True
    except Exception as e:
        # ResourceNotFoundError on first deploy — start fresh
        print(f"[database] No existing DB blob ({type(e).__name__}) — starting with fresh DB")
        return False


def upload_db_to_blob() -> bool:
    """Upload ai_dqm.db to Azure Blob. Call after heavy writes or on schedule."""
    if not _DB_PATH.exists():
        return False
    client = _blob_client()
    if client is None:
        return False
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


def start_periodic_backup(interval_seconds: int = 300):
    """Start daemon thread that uploads DB every interval_seconds (default 5 min)."""
    t = threading.Thread(
        target=_periodic_backup_loop,
        args=(interval_seconds,),
        daemon=True,
        name="db-blob-backup",
    )
    t.start()
    print(f"[database] Periodic DB backup every {interval_seconds}s → Azure Blob")


# ── Restore DB on module import (runs once at startup) ────────────────────────
download_db_from_blob()

# ── SQLAlchemy engine ─────────────────────────────────────────────────────────
engine = create_engine(
    _SQLITE_URL,
    connect_args={"check_same_thread": False},
    pool_pre_ping=True,
    pool_recycle=1800,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
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
        from app.models import GovernanceNotification, NotificationPreference
    except ImportError as e:
        print(f"Governance seed skipped: {e}")
        return
    db = SessionLocal()
    try:
        notif_defaults = [
            ("Quality Score Alerts",  "Get notified when quality scores drop below threshold", "email",  True),
            ("Anomaly Detection",     "Receive alerts for detected data anomalies",            "in_app", True),
            ("Rule Failures",         "Get notified when data quality rules fail",             "email",  True),
            ("Daily Summary",         "Receive daily summary of data quality metrics",         "email",  False),
            ("Weekly Reports",        "Get weekly data quality reports via email",             "email",  True),
            ("Schema Changes",        "Alert when schema changes are detected",                "in_app", True),
            ("New Data Sources",      "Notify when new data sources are connected",            "slack",  False),
            ("Compliance Violations", "Immediate alerts for compliance policy violations",     "email",  True),
        ]
        for title, desc, channel, enabled in notif_defaults:
            if not db.query(GovernanceNotification).filter_by(title=title).first():
                db.add(GovernanceNotification(
                    id=str(uuid.uuid4()), title=title,
                    description=desc, channel=channel, enabled=enabled,
                ))
        pref_defaults = [
            ("Quality Score Alerts", "email",  True),
            ("Anomaly Detection",    "in_app", True),
            ("Rule Failures",        "email",  True),
            ("Daily Summary",        "email",  False),
            ("Weekly Reports",       "email",  True),
            ("Schema Changes",       "in_app", True),
            ("New Data Sources",     "slack",  False),
            ("Compliance Violations","email",  True),
        ]
        for event_type, channel, enabled in pref_defaults:
            if not db.query(NotificationPreference).filter_by(
                event_type=event_type, user_email=None
            ).first():
                db.add(NotificationPreference(
                    user_email=None, event_type=event_type,
                    channel=channel, enabled=enabled,
                ))
        db.commit()
        print("Governance seed data applied.")
    except Exception as e:
        db.rollback()
        print(f"Governance seed failed (non-fatal): {e}")
    finally:
        db.close()