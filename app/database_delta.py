# python-backend/app/database_delta.py
"""
Raw Databricks SQL connector for Delta table operations.
Does NOT use SQLAlchemy — avoids the version conflict with sqlalchemy-databricks.

Usage:
    from app.database_delta import delta_conn, delta_execute, delta_query

    # Write
    delta_execute("INSERT INTO datasources (name, type) VALUES (?, ?)", ["foo", "AZURE_BLOB"])

    # Read
    rows = delta_query("SELECT * FROM profiling_runs WHERE dataset_id = ?", [1])
"""
from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

_HOST      = os.getenv("DATABRICKS_HOST", "").strip().lstrip("https://").lstrip("http://").rstrip("/")
_TOKEN     = os.getenv("DATABRICKS_TOKEN", "").strip()
_HTTP_PATH = os.getenv("DATABRICKS_HTTP_PATH", "").strip()
_CATALOG   = os.getenv("DATABRICKS_CATALOG", "ai_dqm").strip()
_SCHEMA    = os.getenv("DATABRICKS_SCHEMA",  "metadata").strip()

_lock    = threading.Lock()
_conn    = None
_enabled = bool(_HOST and _TOKEN and _HTTP_PATH)


def _get_conn():
    """Lazy singleton connection — created on first use."""
    global _conn
    if not _enabled:
        return None
    with _lock:
        if _conn is not None:
            try:
                cur = _conn.cursor()
                cur.execute("SELECT 1")
                cur.close()
                return _conn
            except Exception:
                _conn = None  # stale — reconnect

        try:
            from databricks import sql as dbsql
            _conn = dbsql.connect(
                server_hostname=_HOST,
                http_path=_HTTP_PATH,
                access_token=_TOKEN,
            )
            cur = _conn.cursor()
            cur.execute(f"USE CATALOG {_CATALOG}")
            cur.execute(f"USE SCHEMA {_SCHEMA}")
            cur.close()
            print(f"[database_delta] Connected → {_CATALOG}.{_SCHEMA} @ {_HOST}")
        except Exception as e:
            print(f"[database_delta] Connection failed: {e}")
            _conn = None
    return _conn


@contextmanager
def get_cursor():
    """Context manager that yields a cursor and handles errors cleanly."""
    conn = _get_conn()
    if conn is None:
        yield None
        return
    cur = conn.cursor()
    try:
        yield cur
    finally:
        cur.close()


def delta_execute(sql: str, params: Optional[list] = None) -> bool:
    """Execute a DML statement (INSERT/UPDATE/DELETE) on Delta. Returns True on success."""
    with get_cursor() as cur:
        if cur is None:
            return False
        try:
            cur.execute(sql, params or [])
            return True
        except Exception as e:
            print(f"[database_delta] execute failed: {e}\nSQL: {sql}")
            return False


def delta_query(sql: str, params: Optional[list] = None) -> List[Dict[str, Any]]:
    """Execute a SELECT on Delta. Returns list of dicts (column→value)."""
    with get_cursor() as cur:
        if cur is None:
            return []
        try:
            cur.execute(sql, params or [])
            cols = [d[0] for d in cur.description] if cur.description else []
            return [dict(zip(cols, row)) for row in (cur.fetchall() or [])]
        except Exception as e:
            print(f"[database_delta] query failed: {e}\nSQL: {sql}")
            return []


def delta_enabled() -> bool:
    """Returns True if Delta is configured and connection is live."""
    return _enabled and _get_conn() is not None