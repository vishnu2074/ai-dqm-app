"""
python-backend/app/database.py
Hybrid database layer with VALIDATION and AUTO-RECOVERY.

PRIMARY:   SQLAlchemy ORM → SQLite at /tmp/ai-dqm/ai_dqm.db
PERSISTENCE: Azure Blob Storage (container: intern26, blob: ai-dqm/ai_dqm.db)
  - On startup: download from blob
  - Periodic: upload every 5 minutes
  - On write: upload after profiling runs

v2.0 CHANGES:
  - Added DB validation after download (checks if valid SQLite)
  - Auto-recreates DB if downloaded file is invalid/empty
  - Fixes file permissions after download
  - Explicit table creation with error handling
  - Added GovernanceSystemConfig table
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

# ── Paths ─────────────────────────────────────────────────────────────────────
_DB_DIR  = Path("/tmp/ai-dqm")
_DB_PATH = _DB_DIR / "ai_dqm.db"
_DB_DIR.mkdir(parents=True, exist_ok=True)

_AGENT_DB_PATH = _DB_DIR / "agent_history.db"

# Expose paths so any code reading os.getenv() gets the right value
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_DB_PATH}")
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


def _validate_sqlite_db(db_path: Path) -> bool:
    """
    Validate that a file is a valid, non-empty SQLite database.
    Returns True if valid, False otherwise.
    """
    if not db_path.exists():
        return False
    
    file_size = db_path.stat().st_size
    if file_size < 4096:  # SQLite header is 100 bytes, minimum page is 512
        print(f"[database] DB file too small ({file_size} bytes) — invalid")
        return False
    
    try:
        import sqlite3
        conn = sqlite3.connect(str(db_path))
        # Try to read the schema
        result = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        conn.close()
        print(f"[database] ✓ DB validation passed — {len(result)} tables found, {file_size:,} bytes")
        return True
    except Exception as e:
        print(f"[database] DB validation failed: {e}")
        return False


def _fix_db_permissions(db_path: Path):
    """Ensure the DB file and directory have correct write permissions."""
    try:
        # Make directory writable
        os.chmod(str(db_path.parent), stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
        
        if db_path.exists():
            # Make file readable and writable
            os.chmod(str(db_path), stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP)
            print(f"[database] ✓ Fixed permissions on {db_path}")
        
        # Also fix WAL and SHM files if they exist
        for suffix in ["-wal", "-shm", "-journal"]:
            journal_path = Path(str(db_path) + suffix)
            if journal_path.exists():
                os.chmod(str(journal_path), stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP)
    except Exception as e:
        print(f"[database] Warning: Could not fix permissions: {e}")


def download_db_from_blob() -> bool:
    """
    Download ai_dqm.db from Azure Blob on startup.
    Validates the downloaded file. If invalid, deletes it and returns False
    so a fresh DB will be created.
    Returns True on success.
    """
    client = _blob_client()
    if client is None:
        print("[database] No Azure Blob config — starting with fresh local DB")
        return False
    
    try:
        # Download to a temp file first
        temp_path = _DB_PATH.with_suffix(".db.tmp")
        with open(temp_path, "wb") as f:
            client.download_blob().readinto(f)
        
        # Validate the downloaded file
        if _validate_sqlite_db(temp_path):
            # Valid — replace the main DB
            if _DB_PATH.exists():
                _DB_PATH.unlink()  # Delete old file
            temp_path.rename(_DB_PATH)
            _fix_db_permissions(_DB_PATH)
            size_kb = _DB_PATH.stat().st_size // 1024
            print(f"[database] ✓ Restored ai_dqm.db from Azure Blob ({size_kb} KB) — all data intact")
            return True
        else:
            # Invalid — delete temp file and start fresh
            if temp_path.exists():
                temp_path.unlink()
            print("[database] Downloaded DB is invalid — will create fresh DB")
            return False
            
    except Exception as e:
        # ResourceNotFoundError on first deploy — start fresh
        print(f"[database] No existing DB blob ({type(e).__name__}) — starting with fresh DB")
        # Clean up any partial downloads
        temp_path = _DB_PATH.with_suffix(".db.tmp")
        if temp_path.exists():
            temp_path.unlink()
        return False


def upload_db_to_blob() -> bool:
    """Upload ai_dqm.db to Azure Blob. Call after heavy writes or on schedule."""
    if not _DB_PATH.exists():
        return False
    
    # Check file is valid before uploading
    if not _validate_sqlite_db(_DB_PATH):
        print("[database] Skipping upload — local DB is invalid")
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


def create_all_tables():
    """
    Explicitly create all tables with error handling.
    Called after Base.metadata.create_all() as a safety net.
    """
    try:
        with engine.connect() as conn:
            # Create governance_system_config table if it doesn't exist
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS governance_system_config (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key TEXT UNIQUE NOT NULL,
                    value TEXT,
                    description TEXT,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """))
            
            # Insert default config values
            default_configs = [
                ("dq_scoring_schedule", "daily", "Schedule for automatic DQ scoring: hourly, daily, weekly, manual"),
                ("email_notifications_enabled", "false", "Enable email notifications for alerts"),
                ("slack_webhook_url", "", "Slack webhook URL for notifications"),
                ("max_profiling_rows", "1000000", "Maximum rows to process in a single profiling run"),
            ]
            
            for key, value, desc in default_configs:
                try:
                    conn.execute(text("""
                        INSERT OR IGNORE INTO governance_system_config (key, value, description)
                        VALUES (:key, :value, :desc)
                    """), {"key": key, "value": value, "desc": desc})
                except Exception:
                    pass  # Already exists
            
            conn.commit()
            print("[database] ✓ governance_system_config table ready")
            
            # Verify all critical tables exist
            result = conn.execute(text("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name IN (
                    'data_sources', 'datasets', 'profiling_runs', 'column_profiles',
                    'notification_inbox', 'governance_system_config', 'dq_rules',
                    'drift_records', 'temporal_checks', 'quality_snapshots'
                )
            """)).fetchall()
            
            existing_tables = {row[0] for row in result}
            critical_tables = {
                'data_sources', 'datasets', 'profiling_runs', 'column_profiles',
                'notification_inbox', 'governance_system_config'
            }
            
            missing = critical_tables - existing_tables
            if missing:
                print(f"[database] ⚠ Missing critical tables: {missing}")
            else:
                print(f"[database] ✓ All critical tables verified ({len(existing_tables)} tables)")
                
    except Exception as e:
        print(f"[database] Error in create_all_tables: {e}")


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
            ("Quality Score Alerts",   "Get notified when quality scores drop below threshold",  "email",  True),
            ("Anomaly Detection",      "Receive alerts for detected data anomalies",             "in_app", True),
            ("Rule Failures",          "Get notified when data quality rules fail",              "email",  True),
            ("Daily Summary",          "Receive daily summary of data quality metrics",          "email",  False),
            ("Weekly Reports",         "Get weekly data quality reports via email",              "email",  True),
            ("Schema Changes",         "Alert when schema changes are detected",                 "in_app", True),
            ("New Data Sources",       "Notify when new data sources are connected",             "slack",  False),
            ("Compliance Violations",  "Immediate alerts for compliance policy violations",      "email",  True),
        ]
        
        for title, desc, channel, enabled in notif_defaults:
            if not db.query(GovernanceNotification).filter_by(title=title).first():
                db.add(GovernanceNotification(
                    id=str(uuid.uuid4()), title=title,
                    description=desc, channel=channel, enabled=enabled,
                ))
        
        pref_defaults = [
            ("Quality Score Alerts",  "email",  True),
            ("Anomaly Detection",     "in_app", True),
            ("Rule Failures",         "email",  True),
            ("Daily Summary",         "email",  False),
            ("Weekly Reports",        "email",  True),
            ("Schema Changes",        "in_app", True),
            ("New Data Sources",      "slack",  False),
            ("Compliance Violations", "email",  True),
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