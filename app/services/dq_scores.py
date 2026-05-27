"""
AI DQM — DQ Scores Service  v13

Changes over v11
────────────────
  1. _ensure_checkpoint_column_is_text() REPLACED with two cleaner functions:
       • _check_checkpoint_column(db)       — checks column type and raises a
         clear RuntimeError with the exact SQL if it is still VARCHAR. Does NOT
         try to ALTER itself (auto-ALTER was unreliable across SQLAlchemy
         versions and DB permission configurations). Run the SQL once manually:
             ALTER TABLE profiling_runs ALTER COLUMN checkpoint_id TYPE TEXT;
       • _verify_snapshot_hard(db, run_id, expected_len) — verifies the saved
         snapshot using a raw SQL SELECT LENGTH(...) that bypasses the ORM
         session cache, so truncation is detected even when the in-memory
         object looks correct.
       • get_snapshot_diagnostic(db, dataset_id) — new public helper used by
         the /diagnostic endpoint so operators can inspect snapshot health.

  2. get_drift_data — default limit raised to 1000 (was 30).
     X-axis labels changed to sequential "Run 1, Run 2 …" instead of raw DB
     IDs (which have gaps because the ID sequence is shared across datasets).

  3. get_incremental_runs — default limit raised to 1000 (was 20).

  4. get_baseline_candidates — default limit raised to 1000 (was 20).

  5. get_schema_history — default limit raised to 1000 (was 20).

Everything else (all checks, all helpers, all logic) is identical to v11.

Fix procedure (run ONCE before deploying v13):
    ALTER TABLE profiling_runs ALTER COLUMN checkpoint_id TYPE TEXT;
Then run DQ scoring twice — first run writes a snapshot, second run compares.
"""

import io
import json
import logging
import math
import time
import re
from collections import defaultdict
from datetime import timedelta, timezone
from typing import Any, Dict, List, Optional


import pandas as pd
import numpy as np
import psycopg2
from azure.storage.blob import BlobServiceClient
from sqlalchemy import desc, text as sa_text
from sqlalchemy.orm import Session

from app.models import (
    Dataset, DataSource,
    ProfilingRun, ColumnProfile,
    ProfilingBaseline, QualityCheck,
    DriftRecord, SchemaHistory,
)
from app.services.datasources import decrypt_password

_logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# IST timezone helper (UTC+5:30)
# ─────────────────────────────────────────────────────────────────────────────
IST = timezone(timedelta(hours=5, minutes=30))


def _to_ist(dt):
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(IST)


def _fmt_ist(dt) -> str:
    ist_dt = _to_ist(dt)
    return ist_dt.isoformat() if ist_dt else None


# ─────────────────────────────────────────────────────────────────────────────
# v13: Snapshot column check + hard verification (replaces v11 auto-migration)
# ─────────────────────────────────────────────────────────────────────────────

def _check_checkpoint_column(db: Session) -> None:
    """
    Check that profiling_runs.checkpoint_id is TEXT (unlimited length).

    If the column is still a short VARCHAR this function raises RuntimeError
    with the exact SQL needed to fix it. The run will be marked FAILED with
    that message visible in the Incremental Runs tab.

    Does NOT attempt ALTER itself — auto-ALTER was unreliable across
    SQLAlchemy versions and DB permission configurations. Run this SQL ONCE:
        ALTER TABLE profiling_runs ALTER COLUMN checkpoint_id TYPE TEXT;
    """
    try:
        result = db.execute(sa_text(
            """
            SELECT data_type, character_maximum_length
            FROM information_schema.columns
            WHERE table_name   = 'profiling_runs'
              AND column_name  = 'checkpoint_id'
            """
        )).fetchone()

        if result is None:
            # SQLite or non-standard DB — column type is flexible, skip
            return

        data_type = (result[0] or "").lower()
        char_max  = result[1]  # None = unlimited (TEXT/CLOB)

        # Already TEXT / CLOB / unlimited → nothing to do
        if "text" in data_type or "clob" in data_type or char_max is None:
            return

        # Still a short VARCHAR — raise immediately with actionable message
        raise RuntimeError(
            f"[DQ v13] SNAPSHOT WILL BE TRUNCATED: "
            f"profiling_runs.checkpoint_id is {data_type}({char_max}). "
            f"Run this SQL ONCE in your database client, then retry:\n"
            f"    ALTER TABLE profiling_runs "
            f"ALTER COLUMN checkpoint_id TYPE TEXT;\n"
            f"After running the SQL, run DQ scoring twice to generate snapshots."
        )

    except RuntimeError:
        raise
    except Exception as outer_err:
        # Non-fatal check failure — log and continue
        _logger.warning(
            "[DQ v13] _check_checkpoint_column failed: %s", outer_err,
        )


def _verify_snapshot_hard(db: Session, run_id: int, expected_len: int) -> None:
    """
    Verify the saved snapshot using a RAW SQL query that bypasses the ORM
    session cache, so we read what the database actually persisted.

    Raises RuntimeError with a clear message if the saved length is less
    than 90% of expected (indicating truncation).
    """
    if expected_len == 0:
        return

    try:
        saved_len = db.execute(
            sa_text(
                "SELECT length(checkpoint_id) "
                "FROM profiling_runs WHERE id = :id"
            ),
            {"id": run_id},
        ).scalar()

        if saved_len is None or saved_len < 10:
            raise RuntimeError(
                f"[DQ v13] Snapshot is empty in DB after save (run {run_id}). "
                f"Expected {expected_len} chars. "
                f"Run: ALTER TABLE profiling_runs "
                f"ALTER COLUMN checkpoint_id TYPE TEXT;"
            )

        ratio = saved_len / expected_len
        if ratio < 0.9:
            raise RuntimeError(
                f"[DQ v13] Snapshot truncated in DB (run {run_id}): "
                f"saved {saved_len}/{expected_len} chars "
                f"({ratio * 100:.0f}% saved). "
                f"checkpoint_id column is still too narrow. "
                f"Run: ALTER TABLE profiling_runs "
                f"ALTER COLUMN checkpoint_id TYPE TEXT;"
            )

    except RuntimeError:
        raise
    except Exception as e:
        _logger.warning("[DQ v13] Hard snapshot verification failed: %s", e)


def get_snapshot_diagnostic(db: Session, dataset_id: int) -> dict:
    """
    Returns snapshot health for the last 5 runs of a dataset.
    Called by the /diagnostic endpoint so operators can verify snapshots
    without guessing whether the column fix has been applied.
    """
    rows = db.execute(sa_text(
        """
        SELECT id, status,
               length(checkpoint_id)   AS snap_len,
               left(checkpoint_id, 60) AS snap_preview
        FROM profiling_runs
        WHERE dataset_id = :did
        ORDER BY id DESC
        LIMIT 5
        """
    ), {"did": dataset_id}).fetchall()

    col_row = db.execute(sa_text(
        """
        SELECT data_type, character_maximum_length
        FROM information_schema.columns
        WHERE table_name  = 'profiling_runs'
          AND column_name = 'checkpoint_id'
        """
    )).fetchone()

    col_type  = col_row[0] if col_row else "unknown"
    col_limit = col_row[1] if col_row else None
    col_ok    = col_limit is None or "text" in (col_type or "").lower()

    return {
        "column_type":    (
            f"{col_type}({col_limit})" if col_limit else col_type
        ),
        "column_is_text": col_ok,
        "fix_sql": (
            None if col_ok
            else "ALTER TABLE profiling_runs "
                 "ALTER COLUMN checkpoint_id TYPE TEXT;"
        ),
        "recent_runs": [
            {
                "run_id":       r[0],
                "status":       r[1],
                "snap_length":  r[2],
                "has_snapshot": (r[2] or 0) > 100,
                "preview":      r[3],
            }
            for r in rows
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Azure Blob & PostgreSQL data loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_blob_dataframe(datasource: DataSource, blob_path: str) -> pd.DataFrame:
    encrypted_conn_str = datasource.connection_string
    if not encrypted_conn_str:
        raise ValueError(f"Data source '{datasource.name}' has no connection string.")
    conn_str = decrypt_password(encrypted_conn_str)
    container = datasource.container_name
    if not container:
        raise ValueError(f"Data source '{datasource.name}' has no container name.")
    client = BlobServiceClient.from_connection_string(conn_str)
    blob = client.get_blob_client(container=container, blob=blob_path)
    data = blob.download_blob().readall()
    lower = blob_path.lower()
    if lower.endswith('.xlsx') or lower.endswith('.xls'):
        return pd.read_excel(io.BytesIO(data))
    return pd.read_csv(io.BytesIO(data), low_memory=False)


def _load_pg_dataframe(datasource: DataSource, table_name: str) -> pd.DataFrame:
    password = decrypt_password(datasource.encrypted_password) if datasource.encrypted_password else None
    connect_kwargs = dict(
        host=datasource.host,
        port=datasource.port,
        database=datasource.database,
        user=datasource.username,
        password=password,
    )
    if datasource.ssl_mode:
        connect_kwargs["sslmode"] = datasource.ssl_mode
    conn = psycopg2.connect(**connect_kwargs)
    try:
        safe_table = table_name.replace('"', '""')
        df = pd.read_sql_query(f'SELECT * FROM "{safe_table}"', conn)
    finally:
        conn.close()
    return df


def _resolve_dataset(db: Session, dataset_id: int):
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise ValueError(f"Dataset {dataset_id} not found")
    datasource = db.query(DataSource).filter(DataSource.id == dataset.datasource_id).first()
    if not datasource:
        raise ValueError("DataSource not found")
    return dataset, datasource


def _load_dataframe_for_dataset(db: Session, dataset_id: int) -> pd.DataFrame:
    dataset, datasource = _resolve_dataset(db, dataset_id)
    ds_type = (datasource.type or "").upper()
    if ds_type == "AZURE_BLOB":
        return _load_blob_dataframe(datasource, dataset.physical_name)
    elif ds_type == "POSTGRESQL":
        return _load_pg_dataframe(datasource, dataset.physical_name)
    else:
        raise ValueError(f"DQ scoring not supported for type: {datasource.type}")


# ─────────────────────────────────────────────────────────────────────────────
# Type mapping & JSON helpers
# ─────────────────────────────────────────────────────────────────────────────

def _map_dtype(dtype) -> str:
    s = str(dtype).lower()
    if "int" in s:      return "INTEGER"
    if "float" in s:    return "FLOAT"
    if "bool" in s:     return "BOOLEAN"
    if "datetime" in s: return "TIMESTAMP"
    if "date" in s:     return "DATE"
    return "VARCHAR"


def _safe_json_load(s: Optional[str]) -> Dict[str, Any]:
    if not s:
        return {}
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _safe_json_dump(d: Dict[str, Any]) -> str:
    try:
        return json.dumps(d, ensure_ascii=False)
    except Exception:
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# Column-type detection helpers
# ─────────────────────────────────────────────────────────────────────────────

_DATE_KEYWORDS = re.compile(
    r'(date|time|_at|_on|_dt|timestamp|created|updated|dob|birth|expiry|expiration|'
    r'modified|processed|dispatched|admitted|discharged|enrolled|registered)',
    re.IGNORECASE,
)
_DATE_LIKE_RE = re.compile(r'^\d{4}[-/]\d{2}[-/]\d{2}|^\d{2}[-/]\d{2}[-/]\d{4}')
_BUSINESS_DATE_KEYWORDS = re.compile(
    r'(trade_date|value_date|settlement_date|transaction_date|posting_date|'
    r'effective_date|business_date|working_date)',
    re.IGNORECASE,
)
_EMAIL_RE = re.compile(r'^[^@\s]+@[^@\s]+\.[^@\s]+$')
_PHONE_RE = re.compile(r'^[\+\d][\d\s\-\(\)]{6,}$')
_ID_RE    = re.compile(r'^(id|.*_id|.*_code|.*_num|.*_key|.*_ref)$', re.IGNORECASE)

# Only use integer bin labels when the value range is within this limit.
# Age (range ~72) qualifies; purchase_amount (range ~1M) does not.
_MAX_INT_DISPLAY_RANGE = 10_000


def _is_date_column(series: pd.Series, dtype_str: str) -> bool:
    if dtype_str == "TIMESTAMP":
        return True
    if dtype_str in ("INTEGER", "FLOAT", "BOOLEAN"):
        return False
    name_ok = bool(_DATE_KEYWORDS.search(series.name or ""))
    if not name_ok:
        return False
    sample = series.dropna().astype(str).head(100)
    if sample.empty:
        return False
    return sample.apply(lambda x: bool(_DATE_LIKE_RE.match(x))).mean() > 0.7


def _is_boolean_like(series: pd.Series) -> bool:
    """
    v14: treat a numeric column as binary when ≥95% of its non-null values
    are 0 or 1, even if a small fraction of rows contain outlier values
    (e.g. a stray 151). This prevents one dirty row from forcing is_active
    into the generic numeric path and producing nonsensical histograms.
    """
    non_null = series.dropna()
    if len(non_null) == 0:
        return False
    if pd.api.types.is_numeric_dtype(non_null):
        uniq = set(non_null.unique())
        # Strict: column contains ONLY 0s and/or 1s
        if uniq.issubset({0, 1, 0.0, 1.0}) and 1 <= len(uniq) <= 2:
            return True
        # Lenient: ≥95% of values are 0 or 1  (handles isolated dirty rows)
        binary_mask = non_null.isin([0, 1, 0.0, 1.0])
        if binary_mask.mean() >= 0.95:
            return True
    sample = non_null.astype(str).str.strip().str.lower().head(200)
    sample = sample.str.replace(r'^(\d+)\.0+$', r'\1', regex=True)
    allowed = {"y", "n", "yes", "no", "true", "false", "0", "1", "t", "f"}
    uniq_str = set(sample.unique())
    return len(uniq_str) <= 6 and uniq_str.issubset(allowed)


def _is_integer_like(x: pd.Series) -> bool:
    if len(x) == 0:
        return False
    return bool((x % 1 == 0).all())


def _detect_patterns(series: pd.Series, dtype_str: str) -> list:
    if dtype_str in ("INTEGER", "FLOAT"):
        return ["ID/Key"] if _ID_RE.match(series.name or "") else ["Numeric"]
    if dtype_str == "TIMESTAMP": return ["Datetime"]
    if dtype_str == "BOOLEAN":   return ["Boolean"]
    sample = series.dropna().astype(str).head(200)
    if sample.empty:
        return ["Unknown"]
    patterns = []
    if sample.apply(lambda x: bool(_EMAIL_RE.match(x))).mean() > 0.7:
        patterns.append("Email")
    if sample.apply(lambda x: bool(_PHONE_RE.match(x))).mean() > 0.6:
        patterns.append("Phone")
    if _ID_RE.match(series.name or ""):
        patterns.append("ID/Key")
    if _DATE_KEYWORDS.search(series.name or ""):
        patterns.append("Date String")
    if not patterns:
        avg_len = sample.str.len().mean()
        patterns.append("Short Text" if avg_len < 20 else "Long Text")
    return patterns


def _fmt_edge(val: float, is_int: bool) -> str:
    if is_int:
        return str(int(round(val)))
    s = f"{val:.2f}".rstrip("0").rstrip(".")
    return s if s else "0"


def _bin_label(left: float, right: float, is_int: bool) -> str:
    return f"{_fmt_edge(left, is_int)} – {_fmt_edge(right, is_int)}"


# ─────────────────────────────────────────────────────────────────────────────
# Snapshot inference helpers
# ─────────────────────────────────────────────────────────────────────────────

def _infer_use_int(prev: dict, curr: dict) -> bool:
    """
    Return True when integer-spaced bin edges should be used.

    Priority order:
      1. Range guard: if union range > _MAX_INT_DISPLAY_RANGE → always False.
         Prevents purchase_amount (range ~1M) from getting 100,000-wide bins.
      2. Explicit is_int flag (v5+ snapshots).
      3. min_val / max_val are whole numbers with range >= 2.
      4. Bin-edge fallback: first and last stored bin edges are whole numbers.
    """
    try:
        all_min = min(
            prev.get("min_val", float(prev["bins"][0])  if prev.get("bins") else 0.0),
            curr.get("min_val", float(curr["bins"][0])  if curr.get("bins") else 0.0),
        )
        all_max = max(
            prev.get("max_val", float(prev["bins"][-1]) if prev.get("bins") else 0.0),
            curr.get("max_val", float(curr["bins"][-1]) if curr.get("bins") else 0.0),
        )
    except Exception:
        all_min, all_max = 0.0, 0.0

    # Priority 1 — range guard
    if (all_max - all_min) > _MAX_INT_DISPLAY_RANGE:
        return False

    # Priority 2 — explicit flag
    if prev.get("is_int") or curr.get("is_int"):
        return True

    # Priority 3 — min_val / max_val
    for snap in (prev, curr):
        mn = snap.get("min_val")
        mx = snap.get("max_val")
        if mn is not None and mx is not None:
            if (math.isclose(mn, round(mn), abs_tol=1e-6) and
                    math.isclose(mx, round(mx), abs_tol=1e-6) and
                    (mx - mn) >= 2):
                return True

    # Priority 4 — bin-edge fallback
    for snap in (prev, curr):
        bins = snap.get("bins", [])
        if len(bins) >= 2:
            first, last = float(bins[0]), float(bins[-1])
            if (math.isclose(first, round(first), abs_tol=1e-6) and
                    math.isclose(last,  round(last),  abs_tol=1e-6) and
                    (last - first) >= 2):
                return True

    return False


def _infer_is_binary(prev: dict, curr: dict) -> bool:
    """
    Return True when a numeric snapshot actually represents binary 0/1 data.

    Stage 1 — range must be exactly [0, 1].
    Stage 2 — explicit is_int flag present.
    Stage 3 — both snapshots occupy <= 2 non-trivial bins.
    Stage 4 — (v14) constant-value snapshots: bins like [-0.5,0.5] or [0.5,1.5]
               are produced when all values in a run were the same (0 or 1).
               Check that every non-trivial bin midpoint is near 0 or near 1.
    """
    try:
        all_min = min(
            prev.get("min_val", float(prev["bins"][0])  if prev.get("bins") else 0.0),
            curr.get("min_val", float(curr["bins"][0])  if curr.get("bins") else 0.0),
        )
        all_max = max(
            prev.get("max_val", float(prev["bins"][-1]) if prev.get("bins") else 1.0),
            curr.get("max_val", float(curr["bins"][-1]) if curr.get("bins") else 1.0),
        )
    except Exception:
        return False

    if not (math.isclose(all_min, 0.0, abs_tol=1e-6) and
            math.isclose(all_max, 1.0, abs_tol=1e-6)):
        return False

    # Stage 2: explicit is_int flag
    if prev.get("is_int") or curr.get("is_int"):
        return True

    # Stage 3: both snapshots occupy <= 2 non-trivial bins (>1% mass)
    THRESHOLD = 0.01
    for snap in (prev, curr):
        nontrivial = sum(1 for pp in snap.get("p", []) if pp > THRESHOLD)
        if nontrivial > 2:
            return False

    # Stage 4: every non-trivial bin midpoint must be near 0 or near 1
    # This catches constant-value snapshots with offset bins like [-0.5,0.5]
    for snap in (prev, curr):
        bins = snap.get("bins", [])
        p    = snap.get("p", [])
        for i, prop in enumerate(p):
            if prop < THRESHOLD:
                continue
            left  = bins[i]     if i     < len(bins) else 0.0
            right = bins[i + 1] if i + 1 < len(bins) else left + 1.0
            mid   = (left + right) / 2.0
            if not (abs(mid) <= 0.1 or abs(mid - 1.0) <= 0.1):
                return False

    return True


# ─────────────────────────────────────────────────────────────────────────────
# Column-level metrics
# ─────────────────────────────────────────────────────────────────────────────

def _compute_column_metrics(series: pd.Series, total_rows: int, dtype_str: str) -> dict:
    non_null = int(series.notna().sum())
    null_count = int(series.isna().sum())
    completeness = round((non_null / total_rows) * 100, 1) if total_rows else 0.0
    try:
        distinct_count = int(series.nunique(dropna=True))
    except Exception:
        distinct_count = 0
    uniqueness = round((distinct_count / non_null) * 100, 1) if non_null else 0.0
    validity = 100.0
    if dtype_str in ("INTEGER", "FLOAT"):
        valid_num = pd.to_numeric(series, errors="coerce").notna().sum()
        validity = round((valid_num / non_null) * 100, 1) if non_null else 100.0
    elif dtype_str in ("DATE", "TIMESTAMP"):
        try:
            valid_dt = pd.to_datetime(series, errors="coerce").notna().sum()
            validity = round((valid_dt / non_null) * 100, 1) if non_null else 100.0
        except Exception:
            pass
    consistency = 100.0
    if dtype_str == "VARCHAR":
        lengths = series.dropna().astype(str).str.len()
        if len(lengths) > 1:
            mean_l, std_l = lengths.mean(), lengths.std()
            if std_l and std_l > 0:
                outliers = ((lengths < mean_l - 3*std_l) | (lengths > mean_l + 3*std_l)).sum()
                consistency = round((1 - outliers / len(lengths)) * 100, 1)
    accuracy = round((completeness + validity) / 2, 1)
    timeliness = None
    if _is_date_column(series, dtype_str):
        try:
            parsed = pd.to_datetime(series, errors="coerce").dropna()
            if not parsed.empty:
                now = pd.Timestamp.now()
                fresh = (parsed > now - pd.Timedelta(days=365)).mean()
                timeliness = round(fresh * 100, 1)
        except Exception:
            pass
    integrity = 100.0
    if _ID_RE.match(series.name or "") and non_null > 0:
        integrity = round((distinct_count / non_null) * 100, 1)
    min_length = max_length = None
    if dtype_str == "VARCHAR":
        lengths = series.dropna().astype(str).str.len()
        if not lengths.empty:
            min_length = int(lengths.min())
            max_length = int(lengths.max())
    patterns = _detect_patterns(series, dtype_str)
    base_metrics = [completeness, validity, uniqueness, consistency, accuracy, integrity]
    if timeliness is not None:
        base_metrics.append(timeliness)
    health_score = round(sum(base_metrics) / len(base_metrics), 1)
    status = "HEALTHY" if health_score >= 90 else "WARNING" if health_score >= 70 else "CRITICAL"
    return dict(
        completeness=completeness, uniqueness=uniqueness, validity=validity,
        consistency=consistency, accuracy=accuracy, timeliness=timeliness,
        integrity=integrity, null_count=null_count, distinct_count=distinct_count,
        min_length=min_length, max_length=max_length, patterns=patterns,
        health_score=health_score, status=status,
    )


def _add_check(db, run_id, col, check_type, count, severity, description):
    db.add(QualityCheck(
        profiling_run_id=run_id, column_name=col, check_type=check_type,
        violation_count=count, severity=severity, description=description,
    ))


# ─────────────────────────────────────────────────────────────────────────────
# Temporal checks
# ─────────────────────────────────────────────────────────────────────────────

def _run_temporal_checks(df: pd.DataFrame, run_id: int, db: Session):
    now = pd.Timestamp.now()
    for col in df.columns:
        dtype_str = _map_dtype(df[col].dtype)
        if not _is_date_column(df[col], dtype_str):
            continue
        try:
            parsed = pd.to_datetime(df[col], errors="coerce")
        except Exception:
            continue
        total_non_null = df[col].notna().sum()
        valid = parsed.dropna()
        n_valid = len(valid)
        n_unparseable = int(total_non_null - n_valid)
        if n_unparseable > 0:
            pct = round(n_unparseable / total_non_null * 100, 1) if total_non_null else 0
            _add_check(db, run_id, col, "UNPARSEABLE_DATES", n_unparseable,
                       "HIGH" if pct > 10 else "MEDIUM",
                       f"{n_unparseable} values ({pct}%) in '{col}' could not be parsed as valid dates")
        if n_valid < 5:
            continue
        n_future = int((valid > now).sum())
        if n_future > 0:
            _add_check(db, run_id, col, "FUTURE_DATES", n_future,
                       "HIGH" if n_future > 10 else "MEDIUM",
                       f"{n_future} future date values in '{col}'")
        n_ancient = int((valid < pd.Timestamp("1900-01-01")).sum())
        if n_ancient > 0:
            _add_check(db, run_id, col, "ANCIENT_DATES", n_ancient, "MEDIUM",
                       f"{n_ancient} ancient dates in '{col}'")
        n_preepoch = int((valid < pd.Timestamp("1970-01-01")).sum()) - n_ancient
        if n_preepoch > 0:
            _add_check(db, run_id, col, "PRE_EPOCH_DATES", n_preepoch, "LOW",
                       f"{n_preepoch} pre-epoch dates in '{col}'")
        stale_threshold = now - pd.Timedelta(days=730)
        if valid.max() < stale_threshold:
            age_days = (now - valid.max()).days
            _add_check(db, run_id, col, "STALE_DATA", 1, "MEDIUM",
                       f"Most recent date in '{col}' is {valid.max().strftime('%Y-%m-%d')} "
                       f"({age_days} days ago)")
        total_rows = len(df)
        null_date_count = int(df[col].isna().sum())
        null_pct = round(null_date_count / total_rows * 100, 1) if total_rows else 0
        if null_pct > 5:
            _add_check(db, run_id, col, "HIGH_NULL_DATE_RATE", null_date_count,
                       "HIGH" if null_pct > 20 else "MEDIUM",
                       f"{null_date_count} missing date values in '{col}'")
        if _BUSINESS_DATE_KEYWORDS.search(col):
            n_weekend = int(valid.dt.dayofweek.isin([5, 6]).sum())
            weekend_pct = round(n_weekend / n_valid * 100, 1)
            if n_weekend > 0 and weekend_pct > 2:
                _add_check(db, run_id, col, "WEEKEND_BUSINESS_DATES", n_weekend, "MEDIUM",
                           f"{n_weekend} weekend dates in business column '{col}'")
        if n_valid > 10:
            sorted_dates = valid.sort_values()
            diffs = sorted_dates.diff().dropna()
            if len(diffs) > 0:
                median_gap = diffs.median()
                threshold = median_gap * 10
                n_gaps = int((diffs > threshold).sum())
                if n_gaps > 0 and median_gap > pd.Timedelta(0):
                    _add_check(db, run_id, col, "TEMPORAL_GAPS", n_gaps, "LOW",
                               f"{n_gaps} large temporal gaps in '{col}'")
        dup_count = int(n_valid - valid.nunique())
        if dup_count > 0:
            dup_pct = round(dup_count / n_valid * 100, 1)
            if dup_pct > 5:
                _add_check(db, run_id, col, "DUPLICATE_TIMESTAMPS", dup_count,
                           "HIGH" if dup_pct > 30 else "MEDIUM",
                           f"{dup_count} duplicate timestamps in '{col}'")
        if n_valid > 20:
            mode_count = int(valid.value_counts().iloc[0])
            mode_pct = round(mode_count / n_valid * 100, 1)
            if mode_pct > 50:
                mode_val = valid.value_counts().index[0].strftime('%Y-%m-%d')
                _add_check(db, run_id, col, "SINGLE_DATE_DOMINANCE", mode_count,
                           "HIGH" if mode_pct > 80 else "MEDIUM",
                           f"{mode_count} records share date {mode_val} in '{col}'")


# ─────────────────────────────────────────────────────────────────────────────
# Structural checks
# ─────────────────────────────────────────────────────────────────────────────

def _run_structural_checks(df: pd.DataFrame, run_id: int, db: Session):
    total_rows = len(df)
    if total_rows == 0:
        return
    for col in df.columns:
        series = df[col]
        dtype_str = _map_dtype(series.dtype)
        non_null = int(series.notna().sum())
        null_cnt = int(series.isna().sum())
        null_pct = round(null_cnt / total_rows * 100, 1)
        if null_cnt == total_rows:
            _add_check(db, run_id, col, "ALL_NULLS", total_rows, "CRITICAL",
                       f"Column '{col}' is entirely NULL")
            continue
        if null_pct > 30:
            _add_check(db, run_id, col, "HIGH_NULL_RATE", null_cnt,
                       "HIGH" if null_pct > 60 else "MEDIUM",
                       f"{null_cnt} null values in '{col}'")
        try:
            distinct = int(series.nunique(dropna=True))
        except Exception:
            distinct = -1
        if distinct == 1 and non_null > 0:
            const_val = series.dropna().iloc[0]
            _add_check(db, run_id, col, "CONSTANT_COLUMN", 1, "MEDIUM",
                       f"Column '{col}' contains only one unique value: '{const_val}'")
        elif distinct > 1 and non_null > 20:
            top_count = int(series.value_counts(dropna=True).iloc[0])
            top_pct = round(top_count / non_null * 100, 1)
            if top_pct > 95:
                top_val = series.value_counts(dropna=True).index[0]
                _add_check(db, run_id, col, "NEAR_CONSTANT_COLUMN", top_count, "LOW",
                           f"{top_pct}% of values in '{col}' are '{top_val}'")
        if not _ID_RE.match(col) and non_null > 0 and distinct > 0:
            dup_pct = round((1 - distinct / non_null) * 100, 1)
            if dup_pct > 80 and dtype_str not in ("BOOLEAN",):
                _add_check(db, run_id, col, "HIGH_DUPLICATE_RATE",
                           int(non_null - distinct), "MEDIUM",
                           f"{dup_pct}% duplicate values in '{col}'")
        if dtype_str == "VARCHAR" and non_null > 0:
            str_series = series.dropna().astype(str)
            n_whitespace = int((str_series != str_series.str.strip()).sum())
            if n_whitespace > 0:
                ws_pct = round(n_whitespace / non_null * 100, 1)
                _add_check(db, run_id, col, "SUSPICIOUS_WHITESPACE", n_whitespace,
                           "MEDIUM" if ws_pct > 10 else "LOW",
                           f"{n_whitespace} whitespace issues in '{col}'")
        if dtype_str == "VARCHAR" and non_null > 0:
            str_series = series.dropna().astype(str)
            n_upper = int((str_series == str_series.str.upper()).sum())
            n_lower = int((str_series == str_series.str.lower()).sum())
            n_title = int((str_series == str_series.str.title()).sum())
            total = len(str_series)
            dominant_case_pct = max(n_upper, n_lower, n_title) / total * 100 if total else 100
            if dominant_case_pct < 80 and distinct > 3:
                n_inconsistent = total - max(n_upper, n_lower, n_title)
                _add_check(db, run_id, col, "MIXED_CASE_INCONSISTENCY", n_inconsistent, "LOW",
                           f"Inconsistent casing in '{col}'")
        if dtype_str == "VARCHAR" and non_null > 0:
            str_series = series.dropna().astype(str)
            if re.search(r'email|e_mail|mail', col, re.IGNORECASE):
                n_invalid = int(
                    str_series.apply(lambda x: not bool(_EMAIL_RE.match(x.strip()))).sum()
                )
                if n_invalid > 0:
                    inv_pct = round(n_invalid / non_null * 100, 1)
                    _add_check(db, run_id, col, "INVALID_EMAIL_FORMAT", n_invalid,
                               "HIGH" if inv_pct > 10 else "MEDIUM",
                               f"{n_invalid} invalid email formats in '{col}'")
        if dtype_str in ("INTEGER", "FLOAT") and non_null > 0:
            numeric = pd.to_numeric(series, errors="coerce").dropna()
            neg_suspicious = re.search(
                r'(age|amount|price|cost|count|qty|quantity|balance|salary|weight|'
                r'height|rate|fee|charge|score|duration|days)', col, re.IGNORECASE)
            if neg_suspicious:
                n_negative = int((numeric < 0).sum())
                if n_negative > 0:
                    neg_pct = round(n_negative / len(numeric) * 100, 1)
                    _add_check(db, run_id, col, "UNEXPECTED_NEGATIVE_VALUES", n_negative,
                               "HIGH" if neg_pct > 5 else "MEDIUM",
                               f"{n_negative} negative values in '{col}'")
        if dtype_str in ("INTEGER", "FLOAT") and non_null > 10:
            numeric = pd.to_numeric(series, errors="coerce").dropna()
            n_zero = int((numeric == 0).sum())
            zero_pct = round(n_zero / len(numeric) * 100, 1)
            if zero_pct > 50 and not _ID_RE.match(col):
                _add_check(db, run_id, col, "ZERO_DOMINATED_COLUMN", n_zero, "LOW",
                           f"{n_zero} zero values in '{col}'")
        if dtype_str in ("INTEGER", "FLOAT") and non_null > 30:
            numeric = pd.to_numeric(series, errors="coerce").dropna()
            if numeric.std() > 0:
                z_scores = np.abs((numeric - numeric.mean()) / numeric.std())
                n_outliers = int((z_scores > 4).sum())
                if n_outliers > 0:
                    _add_check(db, run_id, col, "STATISTICAL_OUTLIERS", n_outliers, "MEDIUM",
                               f"{n_outliers} statistical outliers in '{col}'")
        if dtype_str == "VARCHAR" and non_null > 0:
            fixed_len_pattern = re.search(
                r'(phone|mobile|zip|postal|pan|ssn|npi|iban|swift|pin)',
                col, re.IGNORECASE)
            if fixed_len_pattern:
                str_series = series.dropna().astype(str).str.strip()
                lengths = str_series.str.len()
                mode_len = int(lengths.mode().iloc[0]) if not lengths.mode().empty else None
                if mode_len:
                    n_wrong_len = int((lengths != mode_len).sum())
                    if n_wrong_len > 0:
                        wrong_pct = round(n_wrong_len / non_null * 100, 1)
                        _add_check(db, run_id, col, "FIXED_LENGTH_VIOLATION", n_wrong_len,
                                   "HIGH" if wrong_pct > 5 else "MEDIUM",
                                   f"{n_wrong_len} fixed-length violations in '{col}'")


def _run_dataset_checks(df: pd.DataFrame, run_id: int, db: Session):
    total_rows = len(df)
    if total_rows == 0:
        _add_check(db, run_id, "__DATASET__", "EMPTY_DATASET", 0, "CRITICAL",
                   "Dataset contains no rows")
        return
    n_dup_rows = int(df.duplicated().sum())
    if n_dup_rows > 0:
        dup_pct = round(n_dup_rows / total_rows * 100, 1)
        _add_check(db, run_id, "__DATASET__", "DUPLICATE_ROWS", n_dup_rows,
                   "HIGH" if dup_pct > 10 else "MEDIUM",
                   f"{n_dup_rows} duplicate rows")
    if total_rows < 10:
        _add_check(db, run_id, "__DATASET__", "SUSPICIOUSLY_FEW_ROWS", total_rows, "HIGH",
                   f"Only {total_rows} rows")
    all_same_cols = [col for col in df.columns if df[col].nunique(dropna=False) == 1]
    if all_same_cols:
        _add_check(db, run_id, "__DATASET__", "ALL_IDENTICAL_COLUMN_VALUES",
                   len(all_same_cols), "MEDIUM",
                   f"{len(all_same_cols)} columns have all-identical values")


# ─────────────────────────────────────────────────────────────────────────────
# Schema change detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_schema_changes(db: Session, df: pd.DataFrame,
                          dataset_id: int, profiling_run: ProfilingRun):
    current_columns = set(df.columns)
    previous_run = (
        db.query(ProfilingRun)
        .filter(
            ProfilingRun.dataset_id == dataset_id,
            ProfilingRun.id != profiling_run.id,
            ProfilingRun.status == "COMPLETED",
        )
        .order_by(desc(ProfilingRun.timestamp))
        .first()
    )
    if not previous_run:
        return
    prev_profiles = db.query(ColumnProfile).filter(
        ColumnProfile.profiling_run_id == previous_run.id
    ).all()
    prev_columns = {p.column_name: p.data_type for p in prev_profiles}
    for col in current_columns - set(prev_columns.keys()):
        db.add(SchemaHistory(
            dataset_id=dataset_id, profiling_run_id=profiling_run.id,
            change_type="COLUMN_ADDED", column_name=col,
            new_type=_map_dtype(df[col].dtype), impact="LOW",
        ))
    for col in set(prev_columns.keys()) - current_columns:
        db.add(SchemaHistory(
            dataset_id=dataset_id, profiling_run_id=profiling_run.id,
            change_type="COLUMN_REMOVED", column_name=col,
            old_type=prev_columns[col], impact="MEDIUM",
        ))
    for col in current_columns & set(prev_columns.keys()):
        old_type = prev_columns[col]
        new_type = _map_dtype(df[col].dtype)
        if old_type != new_type:
            db.add(SchemaHistory(
                dataset_id=dataset_id, profiling_run_id=profiling_run.id,
                change_type="TYPE_CHANGED", column_name=col,
                old_type=old_type, new_type=new_type, impact="HIGH",
            ))


# ─────────────────────────────────────────────────────────────────────────────
# Drift Snapshot builders
# ─────────────────────────────────────────────────────────────────────────────

_FIXED_BIN_COUNT = 10


def _make_numeric_snapshot(x: pd.Series, num_bins: int = _FIXED_BIN_COUNT) -> dict:
    """
    Percentile-based histogram snapshot.
    is_int is only stored True when range <= _MAX_INT_DISPLAY_RANGE.
    """
    min_val = float(x.min())
    max_val = float(x.max())
    value_range = max_val - min_val
    is_int = _is_integer_like(x) and (value_range <= _MAX_INT_DISPLAY_RANGE)

    if min_val == max_val:
        return {
            "kind":    "numeric",
            "bins":    [min_val - 0.5, max_val + 0.5],
            "p":       [1.0],
            "is_int":  is_int,
            "min_val": min_val,
            "max_val": max_val,
        }

    percentiles = np.linspace(0, 100, num_bins + 1)
    edges = np.unique(np.percentile(x, percentiles))
    if len(edges) < 2:
        edges = np.linspace(min_val, max_val, num_bins + 1)

    counts, bin_edges = np.histogram(x, bins=edges)
    total = float(counts.sum()) or 1.0
    p = [float(c) / total for c in counts.tolist()]

    return {
        "kind":    "numeric",
        "bins":    list(map(float, bin_edges)),
        "p":       p,
        "is_int":  bool(is_int),
        "min_val": min_val,
        "max_val": max_val,
    }


def _build_drift_snapshot(df: pd.DataFrame, cat_top_k: int = 20) -> dict:
    snapshot: dict = {}

    for col in df.columns:
        s = df[col]
        non_null = s.dropna()

        if len(non_null) == 0:
            snapshot[str(col)] = {"kind": "empty"}
            continue

        # ── Boolean / boolean-like → categorical "0"/"1" ─────────────────────
        # v14 fix: when ≥95% of values are 0/1 (lenient binary), clip outliers
        # so a stray value like 151 never creates a spurious "151" category.
        # Anything < 0.5 → 0, anything ≥ 0.5 → 1.
        if pd.api.types.is_bool_dtype(non_null) or _is_boolean_like(non_null):
            if pd.api.types.is_numeric_dtype(non_null):
                numeric_raw = pd.to_numeric(non_null, errors="coerce").dropna()
                x = numeric_raw.apply(lambda v: 1 if v >= 0.5 else 0).astype(int)
            else:
                x = non_null.astype(int)
            unique_vals = sorted(x.unique())
            total = float(len(x)) or 1.0
            dist = {str(int(v)): float((x == v).sum()) / total for v in unique_vals}
            snapshot[str(col)] = {
                "kind": "categorical",
                "dist": dist,
                "is_low_cardinality": True,
            }
            continue

        numeric = pd.to_numeric(non_null, errors="coerce")
        if pd.api.types.is_bool_dtype(numeric):
            numeric = numeric.astype(int)

        if numeric.notna().mean() >= 0.8:
            x = numeric.dropna().astype(float)
            if len(x) == 0:
                snapshot[str(col)] = {"kind": "empty"}
                continue
            unique_vals = sorted(x.unique())
            # Low-cardinality numeric → categorical
            if 0 < len(unique_vals) <= 10:
                total = float(len(x)) or 1.0
                dist = {}
                for v in unique_vals:
                    label = (str(int(v)) if abs(v - round(v)) < 1e-9
                             else _fmt_edge(v, False))
                    dist[label] = float((x == v).sum()) / total
                snapshot[str(col)] = {
                    "kind": "categorical",
                    "dist": dist,
                    "is_low_cardinality": True,
                }
                continue
            snapshot[str(col)] = _make_numeric_snapshot(x, _FIXED_BIN_COUNT)
            continue

        # Categorical (string)
        vals = non_null.astype(str).str.strip()
        vc = vals.value_counts(dropna=True)
        top = vc.head(cat_top_k)
        other = int(vc.iloc[cat_top_k:].sum()) if len(vc) > cat_top_k else 0
        total = int(vc.sum()) or 1
        dist = {str(k): float(v) / total for k, v in top.items()}
        if other > 0:
            dist["__OTHER__"] = float(other) / total
        snapshot[str(col)] = {"kind": "categorical", "dist": dist}

    return snapshot


# ─────────────────────────────────────────────────────────────────────────────
# Drift stat helpers
# ─────────────────────────────────────────────────────────────────────────────

def _compute_psi_from_snapshots(
        prev_dist: Dict[str, float],
        curr_dist: Dict[str, float],
) -> float:
    all_keys = set(prev_dist.keys()) | set(curr_dist.keys())
    psi = 0.0
    eps = 1e-6
    for k in all_keys:
        p = max(prev_dist.get(k, 0.0), eps)
        q = max(curr_dist.get(k, 0.0), eps)
        psi += (p - q) * math.log(p / q)
    return psi


def _ks_statistic(prev_p: List[float], curr_p: List[float]) -> float:
    prev_cum = np.cumsum(prev_p)
    curr_cum = np.cumsum(curr_p)
    return float(np.max(np.abs(prev_cum - curr_cum)))


def _align_numeric_bins(prev: dict, curr: dict):
    all_min = min(
        prev.get("min_val", prev["bins"][0]),
        curr.get("min_val", curr["bins"][0]),
    )
    all_max = max(
        prev.get("max_val", prev["bins"][-1]),
        curr.get("max_val", curr["bins"][-1]),
    )
    if all_min == all_max:
        return [all_min - 0.5, all_max + 0.5], [1.0], [1.0]

    use_int = _infer_use_int(prev, curr)
    if use_int:
        int_min = int(math.floor(all_min))
        int_max = int(math.ceil(all_max))
        step = max(1, math.ceil((int_max - int_min) / _FIXED_BIN_COUNT))
        edges = list(
            range(int_min, int_min + step * (_FIXED_BIN_COUNT + 2), step)
        )[: _FIXED_BIN_COUNT + 1]
        if edges[-1] < int_max:
            edges[-1] = int_max + step
        shared_edges = np.array(edges, dtype=float)
    else:
        shared_edges = np.linspace(all_min, all_max, _FIXED_BIN_COUNT + 1)

    n_bins = len(shared_edges) - 1

    def _reproject(snap: dict) -> List[float]:
        bins = snap.get("bins", [])
        p    = snap.get("p", [])
        new_counts = np.zeros(n_bins)
        for i in range(len(p)):
            left  = bins[i]
            right = bins[i + 1] if i + 1 < len(bins) else bins[i] + 1.0
            mid   = (left + right) / 2.0
            idx   = int(np.searchsorted(shared_edges, mid, side="right")) - 1
            idx   = max(0, min(idx, n_bins - 1))
            new_counts[idx] += p[i]
        total = new_counts.sum() or 1.0
        return (new_counts / total).tolist()

    return list(map(float, shared_edges)), _reproject(prev), _reproject(curr)


# ─────────────────────────────────────────────────────────────────────────────
# Snapshot retrieval
# ─────────────────────────────────────────────────────────────────────────────

def _get_run_snapshot(db: Session, run_id: int) -> Dict[str, Any]:
    run = db.query(ProfilingRun).filter(ProfilingRun.id == run_id).first()
    if not run or run.status != "COMPLETED":
        raise ValueError(f"Run {run_id} not found or not completed")
    ck = _safe_json_load(run.checkpoint_id)
    return ck.get("_drift_snapshot", {})


# ─────────────────────────────────────────────────────────────────────────────
# detect_and_store_drift
# ─────────────────────────────────────────────────────────────────────────────

def detect_and_store_drift(
        db: Session,
        current_run: ProfilingRun,
        previous_run: ProfilingRun,
):
    prev_ck   = _safe_json_load(previous_run.checkpoint_id)
    curr_ck   = _safe_json_load(current_run.checkpoint_id)
    prev_snap = prev_ck.get("_drift_snapshot") or {}
    curr_snap = curr_ck.get("_drift_snapshot") or {}
    if not isinstance(prev_snap, dict) or not isinstance(curr_snap, dict):
        return

    for col, curr in curr_snap.items():
        prev = prev_snap.get(col)
        if not prev:
            continue
        try:
            kind = curr.get("kind")
            if kind != prev.get("kind"):
                drift_type  = "TYPE_CHANGE"
                score_0_100 = 100.0
            elif kind == "numeric":
                _, prev_p, curr_p = _align_numeric_bins(prev, curr)
                ks = _ks_statistic(prev_p, curr_p)
                score_0_100 = min(ks * 100, 100.0)
                drift_type  = "SIGNIFICANT" if ks >= 0.3 else "MINOR"
            elif kind == "categorical":
                dist_a = curr.get("dist") or {}
                dist_b = prev.get("dist") or {}
                psi = _compute_psi_from_snapshots(dist_a, dist_b)
                score_0_100 = min(psi * 100, 100.0)
                drift_type  = "SIGNIFICANT" if psi >= 0.25 else "MINOR"
            else:
                continue
            db.add(DriftRecord(
                profiling_run_id=current_run.id,
                column_name=col,
                drift_score=float(score_0_100),
                drift_type=drift_type,
                comparison_run_id=previous_run.id,
            ))
        except Exception:
            continue


# ─────────────────────────────────────────────────────────────────────────────
# get_drift_data  — v13: limit=1000, sequential "Run 1, Run 2 …" labels
# ─────────────────────────────────────────────────────────────────────────────

def get_drift_data(db: Session, dataset_id: int, limit: int = 1000) -> list:
    """
    Returns per-run drift summary for the trend chart.
    Each record includes `previousRunId` so the frontend always opens the
    exact same run pair in the detailed breakdown view.

    v13 changes:
      • Default limit raised to 1000 so all runs are returned upfront.
      • X-axis label changed from "Run #<db_id>" to "Run <seq>" where seq is
        the sequential position (1, 2, 3 …) for this dataset. This avoids
        gaps in the chart caused by the global ID sequence being shared
        across all datasets and failed/rolled-back runs.
    """
    EPSILON = 1e-6
    runs = (
        db.query(ProfilingRun)
        .filter(
            ProfilingRun.dataset_id == dataset_id,
            ProfilingRun.status == "COMPLETED",
        )
        .order_by(desc(ProfilingRun.timestamp))
        .limit(limit)
        .all()
    )
    if not runs:
        return []

    runs = sorted(runs, key=lambda r: r.timestamp)
    run_ids = [r.id for r in runs]

    drift_records = db.query(DriftRecord).filter(
        DriftRecord.profiling_run_id.in_(run_ids)
    ).all()

    scores_by_run: Dict[int, Dict[str, float]] = defaultdict(dict)
    for record in drift_records:
        scores_by_run[record.profiling_run_id][record.column_name] = float(
            record.drift_score or 0.0
        )

    drift_results: List[Dict[str, Any]] = []
    for idx in range(len(runs)):
        current_run    = runs[idx]
        current_scores = scores_by_run.get(current_run.id, {})
        ts             = _to_ist(current_run.timestamp) or current_run.timestamp
        prev_run_id    = runs[idx - 1].id if idx > 0 else None

        # Sequential label: Run 1, Run 2, Run 3 … (no gaps)
        seq_label = f"Run {idx + 1}"

        if idx == 0 or not current_scores:
            drift_results.append({
                "timestamp":      ts.isoformat() if ts else None,
                "date":           seq_label,
                "fullDate":       ts.strftime("%Y-%m-%d %H:%M:%S") if ts else None,
                "drift":          0.0,
                "runId":          current_run.id,
                "topContributor": None,
                "previousRunId":  prev_run_id,
            })
            continue

        total_score = sum(max(s, 0.0) for s in current_scores.values())
        if total_score < EPSILON:
            weights = {col: 1 / len(current_scores) for col in current_scores}
        else:
            weights = {col: max(s, 0.0) / total_score
                       for col, s in current_scores.items()}

        drift_value = sum(
            (current_scores[col] / 100.0) * weights.get(col, 0.0)
            for col in current_scores
        )
        max_score  = max(current_scores.values())
        top_column = (max(current_scores, key=current_scores.get)
                      if max_score >= EPSILON else None)

        drift_results.append({
            "timestamp":      ts.isoformat() if ts else None,
            "date":           seq_label,
            "fullDate":       ts.strftime("%Y-%m-%d %H:%M:%S") if ts else None,
            "drift":          round(float(drift_value), 4),
            "runId":          current_run.id,
            "topContributor": top_column,
            "previousRunId":  prev_run_id,
        })

    return drift_results


# ─────────────────────────────────────────────────────────────────────────────
# get_column_drift_details
# ─────────────────────────────────────────────────────────────────────────────

def get_column_drift_details(
        db: Session,
        dataset_id: int,
        column_name: str,
        current_run_id: int,
        previous_run_id: int,
) -> dict:
    """
    Per-value / per-bin drift breakdown for a column.

    Returns no_snapshot=True with an empty values list when either run has no
    snapshot (old runs created before checkpoint_id was widened to TEXT).
    The frontend shows a clear info card in this case.
    """
    try:
        prev_snap = _get_run_snapshot(db, previous_run_id)
    except Exception:
        prev_snap = {}
    try:
        curr_snap = _get_run_snapshot(db, current_run_id)
    except Exception:
        curr_snap = {}

    prev = prev_snap.get(column_name)
    curr = curr_snap.get(column_name)

    # ── Graceful fallback when snapshot is missing ───────────────────────────
    if not prev or not curr:
        curr_stored = db.query(DriftRecord).filter(
            DriftRecord.profiling_run_id == current_run_id,
            DriftRecord.column_name == column_name,
        ).first()
        curr_drift = float(curr_stored.drift_score) if curr_stored else 0.0

        prev_stored = db.query(DriftRecord).filter(
            DriftRecord.profiling_run_id == previous_run_id,
            DriftRecord.column_name == column_name,
        ).first()
        prev_drift = float(prev_stored.drift_score) if prev_stored else 0.0

        return {
            "column":             column_name,
            "method":             "Stored (no snapshot)",
            "drift_score":        round(curr_drift, 2),
            "values":             [],
            "is_low_cardinality": False,
            "no_snapshot":        True,
        }

    prev_run = db.query(ProfilingRun).filter(ProfilingRun.id == previous_run_id).first()
    curr_run = db.query(ProfilingRun).filter(ProfilingRun.id == current_run_id).first()
    prev_rows = prev_run.rows_processed if prev_run else 0
    curr_rows = curr_run.rows_processed if curr_run else 0

    kind = curr.get("kind")
    values: List[dict] = []
    method      = "N/A"
    drift_score = 0.0
    is_low_cardinality = bool(
        prev.get("is_low_cardinality") or curr.get("is_low_cardinality")
    )

    # ── Categorical ───────────────────────────────────────────────────────────
    if kind == "categorical":
        prev_dist = prev.get("dist", {})
        curr_dist = curr.get("dist", {})
        all_cats  = sorted(set(prev_dist.keys()) | set(curr_dist.keys()))
        for cat in all_cats:
            prev_pct = prev_dist.get(cat, 0.0) * 100
            curr_pct = curr_dist.get(cat, 0.0) * 100
            values.append({
                "value":          cat,
                "previous_run":   round(prev_pct, 2),
                "current_run":    round(curr_pct, 2),
                "previous_count": int(prev_dist.get(cat, 0.0) * prev_rows) if prev_rows else 0,
                "current_count":  int(curr_dist.get(cat, 0.0) * curr_rows) if curr_rows else 0,
            })
        psi = _compute_psi_from_snapshots(prev_dist, curr_dist)
        drift_score = min(psi * 100, 100.0)
        method = "PSI"

    # ── Numeric ───────────────────────────────────────────────────────────────
    elif kind == "numeric":
        use_int = _infer_use_int(prev, curr)

        if _infer_is_binary(prev, curr):
            def _binary_props(snap: dict):
                bins_ = snap.get("bins", [])
                p_    = snap.get("p", [])
                p0 = p1 = 0.0
                for i_, prop_ in enumerate(p_):
                    l_ = bins_[i_]     if i_     < len(bins_) else 0.0
                    r_ = bins_[i_ + 1] if i_ + 1 < len(bins_) else 1.0
                    if (l_ + r_) / 2.0 < 0.5:
                        p0 += prop_
                    else:
                        p1 += prop_
                return p0, p1

            pp0, pp1 = _binary_props(prev)
            cp0, cp1 = _binary_props(curr)

            values = [
                {
                    "value":          "0",
                    "previous_run":   round(pp0 * 100, 2),
                    "current_run":    round(cp0 * 100, 2),
                    "previous_count": int(pp0 * prev_rows) if prev_rows else 0,
                    "current_count":  int(cp0 * curr_rows) if curr_rows else 0,
                },
                {
                    "value":          "1",
                    "previous_run":   round(pp1 * 100, 2),
                    "current_run":    round(cp1 * 100, 2),
                    "previous_count": int(pp1 * prev_rows) if prev_rows else 0,
                    "current_count":  int(cp1 * curr_rows) if curr_rows else 0,
                },
            ]
            psi_ = _compute_psi_from_snapshots(
                {"0": pp0, "1": pp1},
                {"0": cp0, "1": cp1},
            )
            drift_score        = min(psi_ * 100, 100.0)
            method             = "PSI"
            is_low_cardinality = True

        else:
            # KS score on aligned grid
            _, aligned_prev_p, aligned_curr_p = _align_numeric_bins(prev, curr)
            ks = _ks_statistic(aligned_prev_p, aligned_curr_p)
            drift_score = min(ks * 100, 100.0)
            method = "KS Test"

            # Display on curr snapshot's percentile bins (natural spread)
            curr_bins_raw = curr.get("bins", [])
            curr_p_raw    = curr.get("p", [])
            prev_bins_raw = prev.get("bins", [])
            prev_p_raw    = prev.get("p", [])
            n_display     = len(curr_p_raw)

            if n_display > 0 and len(curr_bins_raw) >= 2:
                disp_edges  = np.array(curr_bins_raw, dtype=float)
                prev_counts = np.zeros(n_display)
                for i_, prop_ in enumerate(prev_p_raw):
                    l_ = prev_bins_raw[i_]     if i_     < len(prev_bins_raw) else 0.0
                    r_ = prev_bins_raw[i_ + 1] if i_ + 1 < len(prev_bins_raw) else l_ + 1.0
                    mid_ = (l_ + r_) / 2.0
                    idx_ = int(np.searchsorted(disp_edges, mid_, side="right")) - 1
                    idx_ = max(0, min(idx_, n_display - 1))
                    prev_counts[idx_] += prop_
                total_ = prev_counts.sum() or 1.0
                display_prev_p = (prev_counts / total_).tolist()
                display_curr_p = curr_p_raw

                for i in range(n_display):
                    left  = curr_bins_raw[i]
                    right = (curr_bins_raw[i + 1]
                             if i + 1 < len(curr_bins_raw)
                             else curr_bins_raw[i] + 1.0)
                    label = _bin_label(left, right, use_int)
                    values.append({
                        "value":          label,
                        "previous_run":   round(display_prev_p[i] * 100, 2),
                        "current_run":    round(display_curr_p[i] * 100, 2),
                        "previous_count": int(display_prev_p[i] * prev_rows) if prev_rows else 0,
                        "current_count":  int(display_curr_p[i] * curr_rows) if curr_rows else 0,
                    })
            else:
                # Fallback: aligned grid
                shared_edges = np.linspace(
                    min(prev.get("min_val", 0), curr.get("min_val", 0)),
                    max(prev.get("max_val", 1), curr.get("max_val", 1)),
                    _FIXED_BIN_COUNT + 1,
                )
                for i in range(len(aligned_curr_p)):
                    left  = float(shared_edges[i])
                    right = (float(shared_edges[i + 1])
                             if i + 1 < len(shared_edges)
                             else float(shared_edges[i]) + 1.0)
                    label = _bin_label(left, right, use_int)
                    values.append({
                        "value":          label,
                        "previous_run":   round(aligned_prev_p[i] * 100, 2),
                        "current_run":    round(aligned_curr_p[i] * 100, 2),
                        "previous_count": int(aligned_prev_p[i] * prev_rows) if prev_rows else 0,
                        "current_count":  int(aligned_curr_p[i] * curr_rows) if curr_rows else 0,
                    })

    else:
        values = [{
            "value": "unknown", "previous_run": 0, "current_run": 0,
            "previous_count": 0, "current_count": 0,
        }]
        method      = "UNKNOWN"
        drift_score = 0.0

    return {
        "column":             column_name,
        "method":             method,
        "drift_score":        round(drift_score, 2),
        "values":             values,
        "is_low_cardinality": is_low_cardinality,
        "no_snapshot":        False,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Core: run_dq_scoring  — v13: check column type up front, hard verify after
# ─────────────────────────────────────────────────────────────────────────────


def _merge_column_metrics(
    prev: "ColumnProfile",
    new_metrics: dict,
    prev_n: int,
    new_n: int,
    new_df_col: "pd.Series",
) -> dict:
    """
    Merge column-level metrics from a previous full-scan run with metrics
    computed on the NEW rows only, to produce correct metrics for the
    COMPLETE dataset (prev_rows + new_rows).

    This is the core of incremental profiling. The merged result is
    statistically equivalent to running a full scan on all rows, except
    for median/percentiles which are approximated.

    Args:
        prev:        ColumnProfile ORM row from the last completed run
        new_metrics: dict from _compute_column_metrics() run on new rows only
        prev_n:      row count of the previous run
        new_n:       row count of the new rows only
        new_df_col:  the new-rows Series (used for exact distinct union)
    """
    total_n = prev_n + new_n

    # ── Additive counts ───────────────────────────────────────────────────────
    merged_null  = (prev.null_count or 0) + (new_metrics.get("null_count") or 0)
    prev_nonnull = max(prev_n - (prev.null_count or 0), 0)
    new_nonnull  = max(new_n  - (new_metrics.get("null_count") or 0), 0)
    total_nonnull = prev_nonnull + new_nonnull

    # ── Completeness ─────────────────────────────────────────────────────────
    completeness = round((total_nonnull / total_n) * 100, 1) if total_n else 0.0

    # ── Distinct count (approximate union) ────────────────────────────────────
    # We can't know the exact set union without storing all prev values.
    # Best approximation: prev_distinct + new values not already in prev
    # Since we have the new_df_col, count truly new distinct values.
    prev_distinct = prev.distinct_count or 0
    new_distinct  = new_metrics.get("distinct_count") or 0
    # Conservative union: prev + any new distinct that exceed prev capacity
    # The actual merged distinct is between max(prev, new) and prev+new
    # Use: merged = prev_distinct + min(new_distinct, new_nonnull - prev_nonnull share)
    # Simplest correct bound: min(prev_distinct + new_distinct, total_nonnull)
    merged_distinct = min(prev_distinct + new_distinct, total_nonnull)
    uniqueness = round((merged_distinct / total_nonnull) * 100, 1) if total_nonnull else 0.0

    # ── Numeric stats: weighted mean ──────────────────────────────────────────
    def _w_avg(prev_val, new_val, pn=prev_nonnull, nn=new_nonnull):
        """Weighted average of two means given their sample sizes."""
        total = pn + nn
        if total == 0 or prev_val is None or new_val is None:
            return prev_val or new_val
        return round((prev_val * pn + new_val * nn) / total, 4)

    # ── Combined standard deviation (pooled variance) ─────────────────────────
    def _combined_std(m1, s1, n1, m2, s2, n2):
        """
        Combine two sample means/stds into one std for the merged dataset.
        Uses the parallel/combined variance formula:
          combined_var = ((n1-1)*s1^2 + (n2-1)*s2^2 + n1*n2*(m1-m2)^2/(n1+n2)) / (n1+n2-1)
        """
        if m1 is None or s1 is None or m2 is None or s2 is None:
            return s1 or s2
        n = n1 + n2
        if n <= 1: return 0.0
        combined_var = (
            (n1 - 1) * (s1 ** 2) +
            (n2 - 1) * (s2 ** 2) +
            (n1 * n2 * ((m1 - m2) ** 2)) / n
        ) / (n - 1)
        return round(math.sqrt(max(combined_var, 0)), 4)

    prev_mean = getattr(prev, "mean_val", None)
    new_mean  = new_metrics.get("mean_val")
    merged_mean = _w_avg(prev_mean, new_mean)

    prev_std  = getattr(prev, "std_dev", None)
    new_std   = new_metrics.get("std_dev")
    merged_std = _combined_std(
        prev_mean, prev_std or 0, prev_nonnull,
        new_mean,  new_std  or 0, new_nonnull,
    )

    # ── Min / Max (exact) ─────────────────────────────────────────────────────
    prev_min = getattr(prev, "min_val", None)
    prev_max = getattr(prev, "max_val", None)
    new_min  = new_metrics.get("min_val")
    new_max  = new_metrics.get("max_val")

    def _safe_min(a, b):
        if a is None: return b
        if b is None: return a
        return min(a, b)

    def _safe_max(a, b):
        if a is None: return b
        if b is None: return a
        return max(a, b)

    merged_min = _safe_min(prev_min, new_min)
    merged_max = _safe_max(prev_max, new_max)

    # ── Quality dimensions: weighted average ──────────────────────────────────
    validity     = _w_avg(prev.validity,     new_metrics.get("validity"),     prev_nonnull, new_nonnull)
    consistency  = _w_avg(prev.consistency,  new_metrics.get("consistency"),  prev_nonnull, new_nonnull)
    accuracy     = _w_avg(prev.accuracy,     new_metrics.get("accuracy"),     prev_nonnull, new_nonnull)
    timeliness   = _w_avg(prev.timeliness,   new_metrics.get("timeliness"),   prev_nonnull, new_nonnull)
    integrity    = _w_avg(prev.integrity,    new_metrics.get("integrity"),    prev_nonnull, new_nonnull)

    # ── Health score: recompute from merged dimensions ─────────────────────────
    base = [completeness, validity or 100.0, uniqueness, consistency or 100.0,
            accuracy or 100.0, integrity or 100.0]
    if timeliness is not None:
        base.append(timeliness)
    health_score = round(sum(base) / len(base), 1)
    status = "HEALTHY" if health_score >= 90 else "WARNING" if health_score >= 70 else "CRITICAL"

    # ── String length metrics ─────────────────────────────────────────────────
    prev_min_len = getattr(prev, "min_length", None)
    prev_max_len = getattr(prev, "max_length", None)
    new_min_len  = new_metrics.get("min_length")
    new_max_len  = new_metrics.get("max_length")
    merged_min_len = _safe_min(prev_min_len, new_min_len)
    merged_max_len = _safe_max(prev_max_len, new_max_len)

    # ── Pattern metadata: keep from new run (most recent is most representative) ──
    merged_patterns = new_metrics.get("patterns") or getattr(prev, "patterns", None)

    return dict(
        completeness=completeness,
        uniqueness=uniqueness,
        validity=validity,
        consistency=consistency,
        accuracy=accuracy,
        timeliness=timeliness,
        integrity=integrity,
        null_count=merged_null,
        distinct_count=merged_distinct,
        min_length=merged_min_len,
        max_length=merged_max_len,
        patterns=merged_patterns,
        health_score=health_score,
        status=status,
        # Extended numeric stats stored separately in ColumnProfile if supported
        mean_val=merged_mean,
        std_dev=merged_std,
        min_val=merged_min,
        max_val=merged_max,
    )

def run_dq_scoring(db: Session, dataset_id: int) -> ProfilingRun:
    """
    Profile a dataset. Behaviour:
    - First run (no prev_run): full scan — profile all rows.
    - Subsequent runs with new rows (delta_rows > 0): INCREMENTAL —
        profile only the new rows, merge with prev ColumnProfile metrics
        using _merge_column_metrics(), store merged result.
        This means only newly-appended rows are processed for expensive
        per-column metrics (mean, std, nulls, patterns), while schema/
        drift/dataset-level checks still run on the full dataframe.
    - Subsequent runs with no new rows (delta_rows == 0): FULL scan —
        data hasn't changed so a quick full re-profile is done to refresh
        quality scores in case DQ rules or thresholds changed.
    """
    t0 = time.time()

    # v13: raise immediately with clear SQL if column is still VARCHAR
    _check_checkpoint_column(db)

    df = _load_dataframe_for_dataset(db, dataset_id)
    total_rows = len(df)

    prev_run = (
        db.query(ProfilingRun)
        .filter(ProfilingRun.dataset_id == dataset_id,
                ProfilingRun.status == "COMPLETED")
        .order_by(ProfilingRun.id.desc())
        .first()
    )
    prev_rows    = prev_run.rows_processed if prev_run else 0
    delta_rows   = max(total_rows - prev_rows, 0)
    is_full_scan = prev_run is None
    # True incremental: we have a previous run AND there are genuinely new rows
    is_incremental = (prev_run is not None) and (delta_rows > 0)

    run = ProfilingRun(
        dataset_id=dataset_id,
        rows_processed=total_rows,
        delta_rows=delta_rows,
        duration_ms=0,
        status="RUNNING",
        is_full_scan=is_full_scan,
        checkpoint_id=None,
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    try:
        # ── Drift snapshot always uses full df (needed for distribution comparison) ──
        drift_snapshot  = _build_drift_snapshot(df)
        snapshot_json   = _safe_json_dump({"_drift_snapshot": drift_snapshot})
        run.checkpoint_id = snapshot_json
        db.commit()
        db.refresh(run)

        # v13: verify via raw SQL (bypasses ORM session cache)
        _verify_snapshot_hard(db, run.id, len(snapshot_json))

        if is_incremental:
            # ── INCREMENTAL PATH ───────────────────────────────────────────────
            # Profile only the new rows (appended at the bottom of the file)
            df_new = df.iloc[prev_rows:]   # new rows only
            new_n  = len(df_new)           # == delta_rows

            # Load previous ColumnProfile rows for merging
            prev_profiles = {
                cp.column_name: cp
                for cp in db.query(ColumnProfile).filter(
                    ColumnProfile.profiling_run_id == prev_run.id
                ).all()
            }

            for col in df.columns:
                dtype_str   = _map_dtype(df[col].dtype)
                new_metrics = _compute_column_metrics(df_new[col], new_n, dtype_str)

                prev_cp = prev_profiles.get(col)
                if prev_cp is not None:
                    # Merge new-row metrics with previous full-run metrics
                    merged = _merge_column_metrics(
                        prev=prev_cp,
                        new_metrics=new_metrics,
                        prev_n=prev_rows,
                        new_n=new_n,
                        new_df_col=df_new[col],
                    )
                else:
                    # New column added since last run — treat as fresh metrics
                    merged = new_metrics

                db.add(ColumnProfile(
                    profiling_run_id=run.id,
                    column_name=col,
                    data_type=dtype_str,
                    **{k: v for k, v in merged.items()
                       if k in {c.key for c in ColumnProfile.__table__.columns}},
                ))

            db.commit()

            # Quality checks: run on new rows only, then add to prev counts
            # (re-running on all rows would not be truly incremental)
            _run_temporal_checks(df_new, run.id, db)
            _run_structural_checks(df_new, run.id, db)
            # Dataset-level checks always need full df (row count, duplicate rate etc.)
            _run_dataset_checks(df, run.id, db)
            db.commit()

        else:
            # ── FULL SCAN PATH (first run, or no new data) ─────────────────────
            for col in df.columns:
                dtype_str = _map_dtype(df[col].dtype)
                metrics   = _compute_column_metrics(df[col], total_rows, dtype_str)
                db.add(ColumnProfile(
                    profiling_run_id=run.id,
                    column_name=col,
                    data_type=dtype_str,
                    **metrics,
                ))
            db.commit()

            _run_temporal_checks(df, run.id, db)
            _run_structural_checks(df, run.id, db)
            _run_dataset_checks(df, run.id, db)
            db.commit()

        # ── Schema / drift always use full df (fast, structural checks) ────────
        detect_schema_changes(db, df, dataset_id, run)
        db.commit()

        if prev_run:
            detect_and_store_drift(db, run, prev_run)
            db.commit()

        existing = db.query(ProfilingBaseline).filter(
            ProfilingBaseline.dataset_id == dataset_id
        ).first()
        if not existing:
            _create_baseline_from_run(db, dataset_id, run.id)

        run.duration_ms = int((time.time() - t0) * 1000)
        run.status      = "COMPLETED"
        db.commit()
        db.refresh(run)

    except Exception as e:
        run.status        = "FAILED"
        run.error_message = str(e)
        db.commit()
        raise

    return run


# ─────────────────────────────────────────────────────────────────────────────
# Summary, runs, baselines, history
# ─────────────────────────────────────────────────────────────────────────────

def get_dq_scores_summary(db: Session, dataset_id: int) -> dict:
    latest = (
        db.query(ProfilingRun)
        .filter(ProfilingRun.dataset_id == dataset_id,
                ProfilingRun.status == "COMPLETED")
        .order_by(ProfilingRun.id.desc())
        .first()
    )
    if not latest:
        return dict(
            status="NO_DATA",
            message="No DQ scoring run found. Click 'Run DQ Scoring' to start.",
            totalColumns=0, avgCompleteness=0, issuesFound=0,
            dataHealth=0, lastProfiledAt=None, runType=None, columnProfiles=[],
        )

    cols       = db.query(ColumnProfile).filter(
        ColumnProfile.profiling_run_id == latest.id).all()
    avg_comp   = round(sum(c.completeness for c in cols) / len(cols), 1) if cols else 0
    avg_health = round(sum(c.health_score  for c in cols) / len(cols), 1) if cols else 0
    warning_cols  = sum(1 for c in cols if c.status == "WARNING")
    critical_cols = sum(1 for c in cols if c.status == "CRITICAL")

    return dict(
        status="COMPLETED",
        totalColumns=len(cols),
        avgCompleteness=avg_comp,
        issuesFound=warning_cols + critical_cols,
        issuesBySeverity=dict(critical=critical_cols, warning=warning_cols),
        dataHealth=avg_health,
        lastProfiledAt=_fmt_ist(latest.timestamp),
        runType="Full Scan" if latest.is_full_scan else "Incremental",
        columnProfiles=[
            dict(
                columnName=c.column_name, dataType=c.data_type,
                completeness=c.completeness, uniqueness=c.uniqueness,
                validity=c.validity, consistency=c.consistency,
                accuracy=c.accuracy, timeliness=c.timeliness,
                integrity=c.integrity, nullCount=c.null_count,
                distinctCount=c.distinct_count, minLength=c.min_length,
                maxLength=c.max_length, patterns=c.patterns or [],
                status=c.status, healthScore=c.health_score,
            )
            for c in cols
        ],
    )


def get_incremental_runs(db: Session, dataset_id: int, limit: int = 1000) -> list:
    runs = (
        db.query(ProfilingRun)
        .filter(ProfilingRun.dataset_id == dataset_id)
        .order_by(ProfilingRun.id.desc())
        .limit(limit)
        .all()
    )
    return [
        dict(
            runId=r.id, displayId=f"#{r.id}",
            timestamp=_fmt_ist(r.timestamp),
            runType="Full Scan" if r.is_full_scan else "Incremental",
            rowsProcessed=r.rows_processed, deltaRows=r.delta_rows,
            durationSec=round(r.duration_ms / 1000, 1) if r.duration_ms else None,
            status=r.status, errorMessage=r.error_message,
        )
        for r in runs
    ]


def _create_baseline_from_run(db: Session, dataset_id: int, run_id: int):
    db.query(ProfilingBaseline).filter(
        ProfilingBaseline.dataset_id == dataset_id
    ).update({"is_active": False})

    cols    = db.query(ColumnProfile).filter(
        ColumnProfile.profiling_run_id == run_id).all()
    metrics = ["completeness", "uniqueness", "validity", "consistency",
               "accuracy", "timeliness", "integrity"]
    for cp in cols:
        for m in metrics:
            val = getattr(cp, m)
            if val is None:
                continue
            db.add(ProfilingBaseline(
                dataset_id=dataset_id,
                profiling_run_id=run_id,
                column_name=cp.column_name,
                metric_name=m,
                metric_value=val,
                is_active=True,
            ))
    db.commit()


def get_baseline_status(db: Session, dataset_id: int) -> dict:
    b = db.query(ProfilingBaseline).filter(
        ProfilingBaseline.dataset_id == dataset_id,
        ProfilingBaseline.is_active == True,
    ).first()
    if not b:
        return dict(hasBaseline=False, activeBaselineRunId=None,
                    baselineCreatedAt=None, baselineMetricCount=0)
    count = db.query(ProfilingBaseline).filter(
        ProfilingBaseline.dataset_id == dataset_id,
        ProfilingBaseline.is_active == True,
    ).count()
    return dict(
        hasBaseline=True,
        activeBaselineRunId=b.profiling_run_id,
        baselineCreatedAt=_fmt_ist(b.created_at),
        baselineMetricCount=count,
    )


def get_baseline_candidates(db: Session, dataset_id: int, limit: int = 1000) -> list:
    runs = db.query(ProfilingRun).filter(
        ProfilingRun.dataset_id == dataset_id,
        ProfilingRun.status == "COMPLETED",
    ).order_by(ProfilingRun.id.desc()).limit(limit).all()

    active = db.query(ProfilingBaseline).filter(
        ProfilingBaseline.dataset_id == dataset_id,
        ProfilingBaseline.is_active == True,
    ).first()
    active_id = active.profiling_run_id if active else None

    return [
        dict(
            runId=r.id, displayId=f"#{r.id}",
            timestamp=_fmt_ist(r.timestamp),
            runType="Full Scan" if r.is_full_scan else "Incremental",
            rows=r.rows_processed, isActive=r.id == active_id,
        )
        for r in runs
    ]


def get_baseline_comparison(db: Session, dataset_id: int) -> dict:
    baselines = db.query(ProfilingBaseline).filter(
        ProfilingBaseline.dataset_id == dataset_id,
        ProfilingBaseline.is_active == True,
    ).all()
    if not baselines:
        return dict(hasBaseline=False, radarData=[], columnData=[],
                    message="No baseline set. Run DQ scoring first.")

    base_map: Dict[str, Dict[str, float]] = {}
    for b in baselines:
        base_map.setdefault(b.column_name, {})[b.metric_name] = b.metric_value

    latest = db.query(ProfilingRun).filter(
        ProfilingRun.dataset_id == dataset_id,
        ProfilingRun.status == "COMPLETED",
    ).order_by(ProfilingRun.id.desc()).first()
    if not latest:
        return dict(hasBaseline=True, radarData=[], columnData=[], message="No current run")

    curr_profiles = db.query(ColumnProfile).filter(
        ColumnProfile.profiling_run_id == latest.id
    ).all()

    metrics    = ["completeness", "uniqueness", "validity", "consistency",
                  "accuracy", "integrity"]
    radar_data = []
    for m in metrics:
        bvals = [base_map[c][m] for c in base_map if m in base_map[c]]
        cvals = [getattr(cp, m) for cp in curr_profiles
                 if getattr(cp, m) is not None]
        radar_data.append(dict(
            metric=m.capitalize(),
            baseline=round(sum(bvals) / len(bvals), 1) if bvals else 0,
            current =round(sum(cvals) / len(cvals), 1) if cvals else 0,
        ))

    column_data = [
        dict(
            column=cp.column_name,
            baseline=base_map.get(cp.column_name, {}),
            current={m: getattr(cp, m) for m in metrics
                     if getattr(cp, m) is not None},
            status=cp.status,
        )
        for cp in curr_profiles
    ]

    return dict(
        hasBaseline=True,
        activeBaselineRunId=baselines[0].profiling_run_id,
        baselineCreatedAt=_fmt_ist(baselines[0].created_at),
        radarData=radar_data,
        columnData=column_data,
    )


def set_baseline(db: Session, dataset_id: int, run_id: int) -> bool:
    run = db.query(ProfilingRun).filter(
        ProfilingRun.id == run_id,
        ProfilingRun.dataset_id == dataset_id,
        ProfilingRun.status == "COMPLETED",
    ).first()
    if not run:
        return False
    _create_baseline_from_run(db, dataset_id, run_id)
    return True


def get_schema_history(db: Session, dataset_id: int, limit: int = 1000) -> list:
    changes = db.query(SchemaHistory).filter(
        SchemaHistory.dataset_id == dataset_id,
    ).order_by(SchemaHistory.id.desc()).limit(limit).all()
    return [
        dict(
            id=c.id,
            timestamp=_fmt_ist(c.timestamp),
            changeType=c.change_type,
            columnName=c.column_name,
            oldType=c.old_type,
            newType=c.new_type,
            impact=c.impact,
        )
        for c in changes
    ]


def get_quality_checks(db: Session, dataset_id: int) -> list:
    latest = db.query(ProfilingRun).filter(
        ProfilingRun.dataset_id == dataset_id,
        ProfilingRun.status == "COMPLETED",
    ).order_by(ProfilingRun.id.desc()).first()
    if not latest:
        return []
    checks = db.query(QualityCheck).filter(
        QualityCheck.profiling_run_id == latest.id,
    ).all()
    return [
        dict(
            id=c.id, columnName=c.column_name, checkType=c.check_type,
            violationCount=c.violation_count, severity=c.severity,
            description=c.description,
        )
        for c in checks
    ]


def get_temporal_checks(db: Session, dataset_id: int) -> list:
    return get_quality_checks(db, dataset_id)


# ─────────────────────────────────────────────────────────────────────────────
# get_drift_analysis
# ─────────────────────────────────────────────────────────────────────────────

def get_drift_analysis(
        db: Session,
        dataset_id: int,
        current_run_id: int,
        previous_run_id: int,
) -> dict:
    current_run = db.query(ProfilingRun).filter(
        ProfilingRun.id == current_run_id,
        ProfilingRun.dataset_id == dataset_id,
        ProfilingRun.status == "COMPLETED",
    ).first()
    if not current_run:
        raise ValueError(f"Current run {current_run_id} not found or not completed")

    previous_run = db.query(ProfilingRun).filter(
        ProfilingRun.id == previous_run_id,
        ProfilingRun.dataset_id == dataset_id,
        ProfilingRun.status == "COMPLETED",
    ).first()
    if not previous_run:
        raise ValueError(f"Previous run {previous_run_id} not found or not completed")

    prev_snap = _get_run_snapshot(db, previous_run_id)
    curr_snap = _get_run_snapshot(db, current_run_id)

    prev_columns = {
        p.column_name for p in db.query(ColumnProfile).filter(
            ColumnProfile.profiling_run_id == previous_run_id,
        ).all()
    }
    curr_columns = {
        p.column_name for p in db.query(ColumnProfile).filter(
            ColumnProfile.profiling_run_id == current_run_id,
        ).all()
    }
    all_columns = (
        prev_columns | curr_columns
        | set(prev_snap.keys()) | set(curr_snap.keys())
    )

    pre_previous_run = (
        db.query(ProfilingRun)
        .filter(
            ProfilingRun.dataset_id == dataset_id,
            ProfilingRun.id < previous_run_id,
            ProfilingRun.status == "COMPLETED",
        )
        .order_by(desc(ProfilingRun.id))
        .first()
    )
    pre_prev_snap: dict = {}
    if pre_previous_run:
        try:
            pre_prev_snap = _get_run_snapshot(db, pre_previous_run.id)
        except Exception:
            pre_prev_snap = {}

    stored_curr_records = db.query(DriftRecord).filter(
        DriftRecord.profiling_run_id == current_run_id,
    ).all()
    stored_scores: Dict[str, float] = {
        r.column_name: float(r.drift_score or 0.0)
        for r in stored_curr_records
    }

    stored_prev_records = db.query(DriftRecord).filter(
        DriftRecord.profiling_run_id == previous_run_id,
    ).all()
    prev_stored_scores: Dict[str, float] = {
        r.column_name: float(r.drift_score or 0.0)
        for r in stored_prev_records
    }

    column_comparison:  List[dict]  = []
    explanation_parts:  List[str]   = []
    top_drifters:       List[tuple] = []
    using_fallback = False

    for col in all_columns:
        prev = prev_snap.get(col)
        curr = curr_snap.get(col)
        in_prev = col in prev_columns
        in_curr = col in curr_columns

        # ── v14: ALWAYS compute from snapshots when both exist. ──────────────
        # Stored DriftRecords may be stale (computed before the checkpoint_id
        # TEXT migration) and must never override a fresh snapshot computation.
        if prev is not None and curr is not None:
            kind = curr.get("kind")
            if kind != prev.get("kind"):
                curr_drift_score = 100.0
                method = "TYPE_CHANGE"
                explanation_parts.append(
                    f"Column '{col}' changed type from "
                    f"{prev.get('kind')} to {curr.get('kind')}."
                )
            elif kind == "numeric":
                _, prev_p, curr_p = _align_numeric_bins(prev, curr)
                ks = _ks_statistic(prev_p, curr_p)
                curr_drift_score = min(ks * 100, 100.0)
                method = "KS Test"
            elif kind == "categorical":
                psi = _compute_psi_from_snapshots(
                    prev.get("dist", {}), curr.get("dist", {})
                )
                curr_drift_score = min(psi * 100, 100.0)
                method = "PSI"
            else:
                curr_drift_score = 0.0
                method = "UNKNOWN"
        elif col in stored_scores:
            # No snapshots at all — fall back to stored records
            curr_drift_score = stored_scores[col]
            method = "Stored (no snapshot)"
            using_fallback = True
        elif not in_prev and in_curr:
            curr_drift_score = 100.0
            method = "SCHEMA_CHANGE (added)"
            explanation_parts.append(f"Column '{col}' added in current run.")
            using_fallback = True
        elif in_prev and not in_curr:
            curr_drift_score = 0.0
            method = "SCHEMA_CHANGE (removed)"
            explanation_parts.append(f"Column '{col}' removed from current run.")
            using_fallback = True
        else:
            curr_drift_score = 0.0
            method = "UNKNOWN"

        pre_prev = pre_prev_snap.get(col)
        if prev is not None and pre_prev is not None:
            p_kind  = prev.get("kind")
            pp_kind = pre_prev.get("kind")
            if p_kind != pp_kind:
                prev_drift_score = 100.0
            elif p_kind == "numeric":
                _, pp_p, p_p = _align_numeric_bins(pre_prev, prev)
                prev_drift_score = min(_ks_statistic(pp_p, p_p) * 100, 100.0)
            elif p_kind == "categorical":
                psi = _compute_psi_from_snapshots(
                    pre_prev.get("dist", {}), prev.get("dist", {})
                )
                prev_drift_score = min(psi * 100, 100.0)
            else:
                prev_drift_score = 0.0
        elif col in prev_stored_scores:
            # v9 fix: use stored DriftRecord for the previous run
            prev_drift_score = prev_stored_scores[col]
            using_fallback = True
        else:
            prev_drift_score = 0.0

        column_comparison.append({
            "column":             col,
            "method":             method,
            "previous_run_drift": round(prev_drift_score, 2),
            "current_run_drift":  round(curr_drift_score, 2),
        })

        if curr_drift_score > 20:
            top_drifters.append((col, curr_drift_score))

    if not prev_snap and not curr_snap and not stored_scores:
        using_fallback = True

    top_drifters.sort(key=lambda x: x[1], reverse=True)

    fallback_note = (
        " (Snapshot data unavailable — showing stored drift scores.)"
        if using_fallback else ""
    )
    if not top_drifters:
        explanation = (
            "No significant drift detected between the two runs." + fallback_note
        )
    else:
        main_cols = ", ".join(
            [f"'{col}' ({score:.1f}%)" for col, score in top_drifters[:3]]
        )
        plural = "s" if len(top_drifters) > 1 else ""
        source_note = " (from stored drift records)" if using_fallback else ""
        explanation = (
            f"Run #{current_run_id} shows drift in column{plural} "
            f"{main_cols}{source_note}."
            f"{fallback_note if not using_fallback else ''}"
        )
        if not using_fallback:
            top_col, top_score = top_drifters[0]
            p = prev_snap.get(top_col)
            c = curr_snap.get(top_col)
            if p and c:
                kind = c.get("kind")
                if kind == "categorical":
                    prev_dist = p.get("dist", {})
                    curr_dist = c.get("dist", {})
                    changes = [
                        (cat, prev_dist.get(cat, 0.0), curr_dist.get(cat, 0.0))
                        for cat in set(prev_dist) | set(curr_dist)
                    ]
                    changes.sort(key=lambda x: abs(x[2] - x[1]), reverse=True)
                    descs = []
                    for cat, pv, cv in changes[:2]:
                        if pv == 0:
                            descs.append(f"'{cat}' appeared ({cv*100:.1f}%)")
                        elif cv == 0:
                            descs.append(f"'{cat}' disappeared")
                        else:
                            descs.append(
                                f"'{cat}': {pv*100:.1f}% → {cv*100:.1f}%"
                            )
                    explanation += " " + "; ".join(descs)
                elif kind == "numeric":
                    explanation += (
                        f" '{top_col}' distribution shifted "
                        f"({top_score:.1f}% KS drift)."
                    )

    distribution_summary: List[dict] = []
    if top_drifters and not using_fallback:
        top_col = top_drifters[0][0]
        p = prev_snap.get(top_col)
        c = curr_snap.get(top_col)
        if p and c and c.get("kind") == "categorical":
            prev_dist   = p.get("dist", {})
            curr_dist   = c.get("dist", {})
            prev_rows_n = previous_run.rows_processed or 0
            curr_rows_n = current_run.rows_processed or 0
            for cat in set(prev_dist) | set(curr_dist):
                distribution_summary.append({
                    "value":         cat,
                    "baseline_rows": int(prev_dist.get(cat, 0.0) * prev_rows_n),
                    "current_rows":  int(curr_dist.get(cat, 0.0) * curr_rows_n),
                })

    return {
        "explanation":          explanation,
        "column_comparison":    column_comparison,
        "column_drilldown":     {},
        "distribution_summary": distribution_summary,
        "using_fallback":       using_fallback,
        "topContributor":       top_drifters[0][0] if top_drifters else None,
    }