# python-backend/app/services/dq_engine.py
from __future__ import annotations
 
import json
import os
import re
import math
import tempfile
from datetime import datetime, date
from typing import Any, Dict, List, Optional, Tuple
 
import pandas as pd
from sqlalchemy.orm import Session
 
from app.models import DataSource, Dataset, DatasetVersion, DQRule, DQRuleRun, DQRuleRunResult
from app.services.datasources import decrypt_password
 
# Storage
STORAGE_ROOT = os.environ.get("DQ_STORAGE_ROOT", os.path.join(os.getcwd(), "storage"))
DQ_TMP_DIR = os.path.join(STORAGE_ROOT, "dq_previews")
DQ_BLOB_CACHE_DIR = os.path.join(STORAGE_ROOT, "dq_blob_cache")
 
 
def _ensure_dirs() -> None:
    os.makedirs(STORAGE_ROOT, exist_ok=True)
    os.makedirs(DQ_TMP_DIR, exist_ok=True)
    os.makedirs(DQ_BLOB_CACHE_DIR, exist_ok=True)
 
 
def _safe_jsonable(x: Any) -> Any:
    """
    Python 3.14/Starlette refuses NaN/Inf in JSON.
    Convert NaN/Inf to None recursively.
    """
    if x is None:
        return None
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    if isinstance(x, (str, int, bool)):
        return x
    if isinstance(x, dict):
        return {k: _safe_jsonable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_safe_jsonable(v) for v in x]
    # pandas/numpy scalars
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    return x
 
 
def _resolve_dataset_path(dataset: Dataset) -> str:
    """
    Your Dataset model uses physical_name.
    For Azure, this is expected to be the blob key like:
      ai-dqm/raw/cpg/customers/v1/customers.csv
    """
    for attr in ["file_path", "path", "location", "local_path", "storage_path", "physical_name"]:
        if hasattr(dataset, attr):
            val = getattr(dataset, attr)
            if isinstance(val, str) and val.strip():
                return val.strip()
    raise ValueError("Dataset has no usable path field (expected physical_name/file_path/path/etc).")
 
 
def _get_or_create_base_version(db: Session, dataset: Dataset) -> DatasetVersion:
    v1 = (
        db.query(DatasetVersion)
        .filter(DatasetVersion.dataset_id == dataset.id)
        .filter(DatasetVersion.version_number == 1)
        .first()
    )
    if v1:
        return v1
 
    base_path = _resolve_dataset_path(dataset)
 
    v1 = DatasetVersion(
        dataset_id=dataset.id,
        version_number=1,
        file_path=base_path,  # may be local path OR blob key
        created_by="System",
        description="Original dataset (immutable)",
        parent_version_id=None,
    )
    db.add(v1)
    db.commit()
    db.refresh(v1)
    return v1
 
 
def _get_version(db: Session, dataset_id: int, version_id: Optional[int]) -> DatasetVersion:
    if version_id is not None:
        v = (
            db.query(DatasetVersion)
            .filter(DatasetVersion.dataset_id == dataset_id)
            .filter(DatasetVersion.id == version_id)
            .first()
        )
        if not v:
            raise ValueError("Dataset version not found")
        return v
 
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise ValueError("Dataset not found")
    return _get_or_create_base_version(db, dataset)
 
 
def _is_probably_local_path(p: str) -> bool:
    # windows drive or absolute unix or explicit relative file with extension
    return (
        os.path.isabs(p)
        or re.match(r"^[A-Za-z]:\\", p or "") is not None
        or os.path.exists(p)
    )
 
 
def _download_blob_to_cache(connection_string: str, container_name: str, blob_name: str) -> str:
    """
    Download blob to local cache and return local file path.
    Requires azure-storage-blob installed.
    """
    try:
        from azure.storage.blob import BlobServiceClient
    except Exception:
        raise ValueError("azure-storage-blob is not installed. Install it to read Azure Blob datasets.")
 
    _ensure_dirs()
    safe_name = blob_name.replace("/", "__")
    local_path = os.path.join(DQ_BLOB_CACHE_DIR, safe_name)
 
    # If already cached, reuse
    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        return local_path
 
    bsc = BlobServiceClient.from_connection_string(connection_string)
    container = bsc.get_container_client(container_name)
    blob = container.get_blob_client(blob_name)
 
    if not blob.exists():
        raise ValueError(f"Dataset file not found in Azure Blob: container='{container_name}', blob='{blob_name}'")
 
    with open(local_path, "wb") as f:
        stream = blob.download_blob()
        f.write(stream.readall())
 
    return local_path
 
 
def _resolve_to_local_file(db: Session, dataset: Dataset, path_or_blob: str) -> str:
    """
    If local path -> return it.
    If Azure datasource -> download blob and return local path.
    """
    # Local
    if _is_probably_local_path(path_or_blob) and os.path.exists(path_or_blob):
        return path_or_blob
 
    # Azure blob
    ds = db.query(DataSource).filter(DataSource.id == dataset.datasource_id).first()
    if ds and (ds.type or "").upper() == "AZURE_BLOB":
        if not ds.connection_string or not ds.container_name:
            raise ValueError("Azure datasource missing connection_string or container_name in DB.")
        blob_key = path_or_blob.lstrip("/")  # just in case
        conn_str = decrypt_password(ds.connection_string)
        return _download_blob_to_cache(conn_str, ds.container_name, blob_key)
 
    # Unknown source type
    raise ValueError(f"Dataset file not found at: {path_or_blob}")
 
 
def _read_dataset(local_path: str) -> pd.DataFrame:
    ext = os.path.splitext(local_path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(local_path)
    if ext == ".parquet":
        return pd.read_parquet(local_path)
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(local_path)
    return pd.read_csv(local_path)
 
 
# ---------------------------
# Condition parsing/execution
# ---------------------------
def _parse_condition(rule: DQRule) -> Dict[str, Any]:
    cond = (rule.condition or "").strip()
    col = rule.column
 
    if cond.upper() == "NOT NULL":
        return {"op": "not_null", "col": col}
 
    m = re.match(r'REGEX_MATCH\(\s*([a-zA-Z0-9_]+)\s*,\s*"(.*)"\s*\)\s*$', cond)
    if m:
        return {"op": "regex", "col": m.group(1), "pattern": m.group(2)}
 
    if cond.upper().replace(" ", "") == "DISTINCT_COUNT=TOTAL_COUNT":
        return {"op": "unique", "col": col}
 
    m = re.match(r"^([a-zA-Z0-9_]+)\s*<=\s*CURRENT_DATE\s*$", cond)
    if m:
        return {"op": "date_lte_today", "col": m.group(1)}
 
    m = re.match(r"^([a-zA-Z0-9_]+)\s*>\s*([0-9]+(\.[0-9]+)?)\s*$", cond)
    if m:
        return {"op": "gt", "col": m.group(1), "value": float(m.group(2))}
 
    m = re.match(
        r"^([a-zA-Z0-9_]+)\s*>\s*([0-9]+(\.[0-9]+)?)\s+AND\s+\1\s*<\s*([0-9]+(\.[0-9]+)?)\s*$",
        cond,
        flags=re.IGNORECASE,
    )
    if m:
        return {"op": "range", "col": m.group(1), "min": float(m.group(2)), "max": float(m.group(4))}
 
    return {"op": "unknown", "col": col, "raw": cond}
 
 
def _apply_rule(df: pd.DataFrame, rule: DQRule) -> Tuple[pd.Series, Optional[str]]:
    spec = _parse_condition(rule)
    op = spec["op"]
    col = spec.get("col")
 
    if op != "unknown" and col not in df.columns:
        return pd.Series([False] * len(df), index=df.index), f"Column '{col}' not found"
 
    try:
        if op == "not_null":
            return df[col].notna(), None
 
        if op == "regex":
            s = df[col].astype(str)
            return s.str.match(spec["pattern"], na=False), None
 
        if op == "unique":
            s = df[col]
            mask = ~s.duplicated(keep=False)
            mask = mask & s.notna()
            return mask, None
 
        if op == "date_lte_today":
            s = pd.to_datetime(df[col], errors="coerce")
            return s.notna() & (s.dt.date <= date.today()), None
 
        if op == "gt":
            s = pd.to_numeric(df[col], errors="coerce")
            return s.notna() & (s > float(spec["value"])), None
 
        if op == "range":
            s = pd.to_numeric(df[col], errors="coerce")
            lo, hi = float(spec["min"]), float(spec["max"])
            return s.notna() & (s > lo) & (s < hi), None
 
        # Dynamic evaluation fallback (generic rule support)
        try:
       
            condition = rule.condition
       
            # Convert SQL-like syntax to pandas syntax
            condition = condition.replace("AND", "&")
            condition = condition.replace("OR", "|")
       
            # Handle IN clause
            condition = re.sub(
                r'([a-zA-Z0-9_]+)\s+IN\s+\((.*?)\)',
                lambda m: f'df["{m.group(1)}"].isin([{m.group(2)}])',
                condition
            )
       
            # Replace column references
            for colname in df.columns:
                condition = re.sub(rf'\b{colname}\b', f'df["{colname}"]', condition)
       
            mask = eval(condition)
       
            return mask.fillna(False), None
       
        except Exception as e:
       
            return pd.Series([False] * len(df), index=df.index), str(e)
 
    except Exception as e:
        return pd.Series([False] * len(df), index=df.index), str(e)
 
def _preview_path(dataset_id: int, run_id: int) -> str:
    _ensure_dirs()
    return os.path.join(DQ_TMP_DIR, f"dataset_{dataset_id}_run_{run_id}_preview.csv")
 
 
def preview_apply_rules(
    db: Session,
    dataset_id: int,
    version_id: Optional[int],
    rule_codes: Optional[List[str]],
    mode: str = "flag",
    preview_rows: int = 25,
    samples_per_rule: int = 10,
) -> Dict[str, Any]:
    if mode not in ("flag", "filter"):
        raise ValueError("mode must be 'flag' or 'filter'")
 
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise ValueError("Dataset not found")
 
    version = _get_version(db, dataset_id, version_id)
 
    # ✅ IMPORTANT: resolve blob/local to a local file
    local_input_path = _resolve_to_local_file(db, dataset, version.file_path)
    df = _read_dataset(local_input_path)
 
    # select rules
    q = db.query(DQRule).filter(DQRule.dataset_id == dataset_id)
    if rule_codes and len(rule_codes) > 0:
        q = q.filter(DQRule.rule_code.in_(rule_codes))
    else:
        q = q.filter(DQRule.status == "Active")
    rules: List[DQRule] = q.order_by(DQRule.id.asc()).all()
 
    # create run record
    run = DQRuleRun(
        dataset_id=dataset_id,
        input_version_id=version.id,
        status="PREVIEW",
        mode=mode,
        started_at=datetime.utcnow(),
    )
    db.add(run)
    db.commit()
    db.refresh(run)
 
    masks: List[pd.Series] = []
    rule_results_payload: List[Dict[str, Any]] = []
 
    for r in rules:
        mask, err = _apply_rule(df, r)
        masks.append(mask)
 
        bad = df.loc[~mask].head(samples_per_rule)
 
        # ✅ sanitize NaN/Inf in samples
        bad_safe = bad.where(pd.notna(bad), None)
        samples = bad_safe.to_dict(orient="records")
 
        pass_rate = round(float(mask.mean() * 100.0), 2) if len(df) else 0.0
        violations = int((~mask).sum())
 
        db.add(
            DQRuleRunResult(
                run_id=run.id,
                rule_code=r.rule_code,
                rule_name=r.name,
                rule_type=r.type,
                column=r.column,
                condition=r.condition,
                pass_rate=pass_rate,
                violation_count=violations,
                samples_json=json.dumps(_safe_jsonable(samples)),
                error_message=err,
            )
        )
 
        rule_results_payload.append(
            {
                "rule_id": r.rule_code,
                "name": r.name,
                "type": r.type,
                "column": r.column,
                "condition": r.condition,
                "pass_rate": pass_rate,
                "violations": violations,
                "samples": samples,
                "error": err,
            }
        )
 
    db.commit()
 
    # overall mask
    if masks:
        overall = masks[0].copy()
        for m in masks[1:]:
            overall = overall & m
    else:
        overall = pd.Series([True] * len(df), index=df.index)
 
    df_out = df.copy()
 
    if mode == "flag":
 
    # Boolean column indicating whether row passed all rules
 
        df_out["dq_is_valid"] = overall.astype(bool)
   
        failed_rules = []
 
        violation_counts = []
   
        for idx in df_out.index:
   
            failed = []
   
            for i, rule in enumerate(rules):
   
                try:
 
                    if not bool(masks[i].loc[idx]):
 
                        failed.append(rule.rule_code)
 
                except Exception:
 
                    failed.append(rule.rule_code)
   
            failed_rules.append(",".join(failed))
 
            violation_counts.append(len(failed))
   
        df_out["dq_failed_rules"] = failed_rules
 
        df_out["dq_violation_count"] = violation_counts
 
 
    if mode == "filter":
        df_out = df_out.loc[overall].copy()
 
    # ✅ save preview file locally
    tmp = _preview_path(dataset_id, run.id)
    df_out.to_csv(tmp, index=False)
 
    run.temp_output_path = tmp
    run.finished_at = datetime.utcnow()
    db.commit()
 
    rows_in = int(len(df))
    rows_out = int(len(df_out))
    overall_pass = round((rows_out / rows_in) * 100.0, 2) if rows_in else 0.0
 
    # ✅ sanitize preview rows
    preview_df = df_out.head(preview_rows)
 
    preview_df = preview_df.where(pd.notna(preview_df), None)
 
    preview_rows_payload = preview_df.to_dict(orient="records")
 
    try:
        from app.routers.notification_inbox_routes import create_inbox_notification
 
        # 🚨 Trigger only if bad quality
        if overall_pass < 90:
            severity = "critical" if overall_pass < 70 else "warning"
 
            create_inbox_notification(
                title="DQ Run Alert",
                message=f"Data quality dropped to {overall_pass}%. Check rule violations.",
                category="dq",
                severity=severity,
                link="/dq-rules",
                dataset=str(dataset_id),
            )
 
        else:
            # Optional success notification
            create_inbox_notification(
                title="DQ Run Completed",
                message=f"DQ run completed with pass rate {overall_pass}%",
                category="dq",
                severity="info",
                link="/dq-rules",
                dataset=str(dataset_id),
            )
 
    except Exception:
        pass
 
    return _safe_jsonable(
        {
            "status": "success",
            "run_id": run.id,
            "mode": mode,
            "summary": {"rows_in": rows_in, "rows_out": rows_out, "overall_pass_rate": overall_pass},
            "rule_results": rule_results_payload,
            "preview_rows": preview_rows_payload,
        }
    )
 
 
def save_preview_as_new_version(
    db: Session,
    dataset_id: int,
    run_id: int,
    description: Optional[str] = None,
    created_by: str = "Admin",
) -> Dict[str, Any]:
    run = (
        db.query(DQRuleRun)
        .filter(DQRuleRun.dataset_id == dataset_id)
        .filter(DQRuleRun.id == run_id)
        .first()
    )
    if not run:
        raise ValueError("Run not found")
 
    if run.status != "PREVIEW":
        raise ValueError("Only PREVIEW runs can be saved")
 
    if not run.temp_output_path or not os.path.exists(run.temp_output_path):
        raise ValueError("Preview output missing. Run preview again.")
 
    last = (
        db.query(DatasetVersion)
        .filter(DatasetVersion.dataset_id == dataset_id)
        .order_by(DatasetVersion.version_number.desc())
        .first()
    )
    next_num = int(last.version_number + 1) if last else 2
 
    _ensure_dirs()
    versions_dir = os.path.join(STORAGE_ROOT, "datasets", str(dataset_id), "versions")
    os.makedirs(versions_dir, exist_ok=True)
 
    out_path = os.path.join(versions_dir, f"v{next_num}.csv")
 
    with open(run.temp_output_path, "rb") as src, open(out_path, "wb") as dst:
        dst.write(src.read())
 
    new_v = DatasetVersion(
        dataset_id=dataset_id,
        version_number=next_num,
        file_path=out_path,  # saved versions are local (you can also upload back to blob if you want)
        created_by=created_by,
        description=description or f"Saved from DQ run {run_id}",
        parent_version_id=run.input_version_id,
    )
    db.add(new_v)
    db.commit()
    db.refresh(new_v)
 
    run.status = "SAVED"
    run.output_version_id = new_v.id
    run.finished_at = datetime.utcnow()
    db.commit()
 
    try:
        from app.routers.notification_inbox_routes import create_inbox_notification
 
        create_inbox_notification(
            title="New Dataset Version Created",
            message=f"Dataset {dataset_id} saved as version v{new_v.version_number}",
            category="dataset",
            severity="info",
            link="/datasets",
            dataset=str(dataset_id),
        )
 
    except Exception:
        pass
 
    return {
        "status": "success",
        "output_version_id": new_v.id,
        "version_number": new_v.version_number,
        "file_path": new_v.file_path,
        "message": "Saved as a new dataset version. Original dataset unchanged.",
    }
 
 
def get_preview_file_path(db: Session, dataset_id: int, run_id: int) -> str:
    run = (
        db.query(DQRuleRun)
        .filter(DQRuleRun.dataset_id == dataset_id)
        .filter(DQRuleRun.id == run_id)
        .first()
    )
    if not run:
        raise ValueError("Run not found")
    if not run.temp_output_path or not os.path.exists(run.temp_output_path):
        raise ValueError("Preview file not found. Run preview again.")
    return run.temp_output_path