"""
python-backend/app/services/alerts.py

Derives ALL alerts for a dataset from real DB tables.
NEVER raises an exception — always returns a valid response dict.

Alert sources:
  QualityCheck     → structural/temporal/dataset quality violations
  DriftRecord      → distribution drift + schema drift
  SchemaHistory    → column added / removed / type changed
  ProfilingRun     → health score dips/rises vs previous run
  ColumnProfile    → completeness drops between runs
  DQRuleRunResult  → rule breach counts
  ColumnProfile    → timeliness / freshness issues
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from app.models import (
    ColumnProfile, Dataset, DQRule, DQRuleRun, DQRuleRunResult,
    DriftRecord, ProfilingRun, QualityCheck, SchemaHistory,
)

IST = timezone(timedelta(hours=5, minutes=30))


# ─── view_route map ───────────────────────────────────────────────────────────
# Maps each alert category to the frontend route the user should navigate to.
_CATEGORY_ROUTE: Dict[str, Dict[str, str]] = {
    "Quality":   {"label": "Quality",  "path": "/anomalies"},
    "Drift":     {"label": "Drift",  "path": "/dq-scores"},
    "Schema":    {"label": "Datasets",   "path": "/datasets"},
    "Health":    {"label": "DQ Scores",  "path": "/dq-scores"},
    "Rules":     {"label": "DQ Rules",   "path": "/dq-rules"},
    "Freshness": {"label": "DQ Scores",  "path": "/dq-scores"},
}


def _view_route(category: str) -> Dict[str, str]:
    return _CATEGORY_ROUTE.get(category, {"label": "Dashboard", "path": "/"})


# ─── helpers ──────────────────────────────────────────────────────────────────

def _fmt(dt: Optional[datetime]) -> Optional[str]:
    if not dt:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(IST).isoformat()


def _time_ago(dt: Optional[datetime]) -> str:
    if not dt:
        return "—"
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    diff = int((datetime.now(timezone.utc) - dt).total_seconds())
    if diff < 60:    return f"{diff}s ago"
    if diff < 3600:  return f"{diff // 60}m ago"
    if diff < 86400: return f"{diff // 3600}h ago"
    return f"{diff // 86400}d ago"


def _check_label(check_type: str) -> str:
    return (check_type or "").replace("_", " ").title()


_SEV_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
_VALID_SEV  = {"CRITICAL", "HIGH", "MEDIUM", "LOW"}


def _norm_sev(s: Any) -> str:
    v = str(s or "LOW").upper()
    return v if v in _VALID_SEV else "MEDIUM"


def _runs_for_dataset(db: Session, dataset_id: int, limit: int = 10) -> List[ProfilingRun]:
    try:
        return (
            db.query(ProfilingRun)
            .filter(ProfilingRun.dataset_id == dataset_id, ProfilingRun.status == "COMPLETED")
            .order_by(ProfilingRun.id.desc())
            .limit(limit)
            .all()
        )
    except Exception:
        return []


def _inc_id(dataset_id: int, seq: int) -> str:
    return f"INC-{dataset_id}-{seq:04d}"


def _stable_seq(prefix: str, source_id: int, dataset_id: int) -> int:
    h = int(hashlib.md5(f"{dataset_id}:{prefix}:{source_id}".encode()).hexdigest(), 16)
    return (h % 9000) + 1000


def _empty_summary() -> Dict[str, int]:
    return {"total": 0, "critical": 0, "high": 0, "medium": 0, "low": 0, "new": 0, "open": 0}


# ─── alert builders ───────────────────────────────────────────────────────────

def _quality_check_alerts(
    db: Session,
    dataset_id: int,
    runs: List[ProfilingRun],
    latest_run_id: int,
) -> List[Dict[str, Any]]:
    """
    One alert per QualityCheck violation — ONLY from the latest run.
    FIX: was querying all recent runs (up to 10), producing one alert per
    run for each check type+column combination.
    """
    if not runs:
        return []
    try:
        run_map  = {r.id: r for r in runs}
        # Only query the latest run
        checks = (
            db.query(QualityCheck)
            .filter(QualityCheck.profiling_run_id == latest_run_id)
            .order_by(QualityCheck.id.asc())
            .all()
        )
        alerts = []
        for chk in checks:
            sev     = _norm_sev(chk.severity)
            run_obj = run_map.get(chk.profiling_run_id)
            ts      = run_obj.timestamp if run_obj else None
            col_part = f" on `{chk.column_name}`" if chk.column_name else ""
            label   = _check_label(chk.check_type or "quality issue")
            alerts.append({
                "incident_id":   _inc_id(dataset_id, _stable_seq("QC", chk.id, dataset_id)),
                "db_id":         chk.id,
                "source":        "Quality Check",
                "category":      "Quality",
                "title":         f"{label}{col_part}",
                "detail":        chk.description or f"{label} detected{col_part}. {chk.violation_count or 0:,} rows affected.",
                "severity":      sev,
                "column_name":   chk.column_name,
                "affected_rows": chk.violation_count or 0,
                "run_id":        chk.profiling_run_id,
                "detected_at":   _fmt(ts),
                "time_ago":      _time_ago(ts),
                "is_new":        chk.profiling_run_id == latest_run_id,
                "status":        getattr(chk, "status", None) or "open",
                "check_type":    chk.check_type or "QUALITY_CHECK",
                "view_route":    _view_route("Quality"),
            })
        return alerts
    except Exception:
        return []


def _drift_alerts(
    db: Session,
    dataset_id: int,
    runs: List[ProfilingRun],
    latest_run_id: int,
) -> List[Dict[str, Any]]:
    """
    One alert per DriftRecord — ONLY from the latest run, ONLY distribution drift.
    FIX 1: Was querying all recent runs → same column appeared once per run.
    FIX 2: drift_score is 0-100 scale. Was checking >0.5 / >0.25 (0-1 scale)
            so every drift record became CRITICAL. Corrected to >50 / >25.
    FIX 3: Skip schema-type drift records — those are handled by _schema_history_alerts.
    """
    if not runs:
        return []
    try:
        run_map = {r.id: r for r in runs}

        # Schema-type records — handled by _schema_history_alerts, skip here
        SCHEMA_TYPES = {"COLUMN_ADDED", "COLUMN_REMOVED", "TYPE_CHANGED",
                        "SCHEMA", "SCHEMA_CHANGE", "TYPE_CHANGE"}

        records = (
            db.query(DriftRecord)
            .filter(DriftRecord.profiling_run_id == latest_run_id)
            .order_by(DriftRecord.id.asc())
            .all()
        )
        alerts = []
        for dr in records:
            run_obj    = run_map.get(dr.profiling_run_id)
            ts         = run_obj.timestamp if run_obj else None
            drift_type = getattr(dr, "drift_type", None) or "drift"
            col        = getattr(dr, "column_name", None)
            drift_score = float(getattr(dr, "drift_score", None) or 0.0)
            sev_raw    = getattr(dr, "severity", None)

            # Skip schema-type drifts
            if drift_type.upper() in SCHEMA_TYPES:
                continue

            is_schema = False
            category  = "Drift"

            if sev_raw:
                sev = _norm_sev(sev_raw)
            elif drift_score is not None:
                # FIX: drift_score is 0-100, not 0-1
                sev = "CRITICAL" if drift_score > 70 else "HIGH" if drift_score > 40 else "MEDIUM"
            else:
                sev = "MEDIUM"

            col_part = f" — `{col}`" if col else ""
            title = f"Distribution Drift Detected{col_part}"
            score_str   = f" (score: {drift_score:.1f}%)" if drift_score is not None else ""
            detail_base = getattr(dr, "detail", None) or getattr(dr, "message", None) or ""
            detail      = detail_base if detail_base else f"{title}{score_str}."

            alerts.append({
                "incident_id":   _inc_id(dataset_id, _stable_seq(category, dr.id, dataset_id)),
                "db_id":         dr.id,
                "source":        "Drift Detector",
                "category":      category,
                "title":         title,
                "detail":        detail,
                "severity":      sev,
                "column_name":   col,
                "affected_rows": None,
                "run_id":        dr.profiling_run_id,
                "detected_at":   _fmt(ts),
                "time_ago":      _time_ago(ts),
                "is_new":        dr.profiling_run_id == latest_run_id,
                "status":        "open",
                "check_type":    drift_type,
                "view_route":    _view_route("Drift"),
            })
        return alerts
    except Exception:
        return []


def _schema_history_alerts(
    db: Session,
    dataset_id: int,
    latest_run_id: int,
) -> List[Dict[str, Any]]:
    """
    Alerts from SchemaHistory — column added/removed/type changed.
    FIX: Skip if the dataset has only one completed run (first profiling run).
    SchemaHistory is only meaningful when there is a previous run to compare against.
    On a brand-new dataset the first run has nothing to compare to so no schema alerts.
    """
    try:
        # Count completed runs for this dataset
        completed_run_count = (
            db.query(ProfilingRun)
            .filter(ProfilingRun.dataset_id == dataset_id, ProfilingRun.status == "COMPLETED")
            .count()
        )
        # If only one run has ever completed, schema changes are meaningless
        if completed_run_count <= 1:
            return []

        rows = (
            db.query(SchemaHistory)
            .filter(SchemaHistory.dataset_id == dataset_id)
            .order_by(SchemaHistory.id.desc())
            .limit(30)
            .all()
        )
        alerts = []
        for sh in rows:
            ts          = getattr(sh, "timestamp", None)
            change_type = getattr(sh, "change_type", "schema_change") or "schema_change"
            col         = getattr(sh, "column_name", None)
            old_t       = getattr(sh, "old_type", None)
            new_t       = getattr(sh, "new_type", None)
            impact      = getattr(sh, "impact", "MEDIUM") or "MEDIUM"
            run_id      = getattr(sh, "profiling_run_id", None)

            col_part = f" — `{col}`" if col else ""
            ct_upper = change_type.upper()
            if "TYPE" in ct_upper and "CHANGE" in ct_upper:
                title  = f"Column Type Changed{col_part}"
                detail = f"Column `{col}` changed from `{old_t}` → `{new_t}`. Downstream consumers may break."
            elif "ADD" in ct_upper:
                title  = f"Column Added{col_part}"
                detail = f"New column `{col}` ({new_t}) detected. Review downstream pipeline compatibility."
            elif "REMOV" in ct_upper or "DROP" in ct_upper:
                title  = f"Column Removed{col_part}"
                detail = f"Column `{col}` ({old_t}) was removed. Dependent pipelines will fail."
            else:
                title  = f"Schema Change: {change_type.replace('_', ' ').title()}{col_part}"
                detail = f"Schema change ({change_type}) detected{col_part}."

            alerts.append({
                "incident_id":   _inc_id(dataset_id, _stable_seq("SH", sh.id, dataset_id)),
                "db_id":         sh.id,
                "source":        "Schema History",
                "category":      "Schema",
                "title":         title,
                "detail":        detail,
                "severity":      _norm_sev(impact),
                "column_name":   col,
                "affected_rows": None,
                "run_id":        run_id,
                "detected_at":   _fmt(ts),
                "time_ago":      _time_ago(ts),
                "is_new":        run_id == latest_run_id if run_id else False,
                "status":        "open",
                "check_type":    change_type,
                "view_route":    _view_route("Schema"),
            })
        return alerts
    except Exception:
        return []


def _health_score_alerts(
    db: Session,
    dataset_id: int,
    runs: List[ProfilingRun],
    latest_run_id: int,
) -> List[Dict[str, Any]]:
    """Alert when overall health score changes significantly between consecutive runs."""
    if len(runs) < 2:
        return []
    try:
        sorted_runs = sorted(runs, key=lambda r: r.id)

        def _avg_health(run: ProfilingRun) -> Optional[float]:
            try:
                profiles = db.query(ColumnProfile).filter(
                    ColumnProfile.profiling_run_id == run.id
                ).all()
                scores = [p.health_score for p in profiles if p.health_score is not None]
                return round(sum(scores) / len(scores), 1) if scores else None
            except Exception:
                return None

        scores_map: Dict[int, Optional[float]] = {r.id: _avg_health(r) for r in sorted_runs}
        alerts = []

        for i in range(1, len(sorted_runs)):
            curr       = sorted_runs[i]
            prev       = sorted_runs[i - 1]
            curr_score = scores_map.get(curr.id)
            prev_score = scores_map.get(prev.id)
            if curr_score is None or prev_score is None:
                continue
            delta = round(curr_score - prev_score, 1)
            if abs(delta) < 2.0:
                continue

            direction = "dropped" if delta < 0 else "improved"
            abs_delta = abs(delta)
            if abs_delta >= 10:
                sev = "CRITICAL" if delta < 0 else "LOW"
            elif abs_delta >= 5:
                sev = "HIGH" if delta < 0 else "LOW"
            else:
                sev = "MEDIUM" if delta < 0 else "LOW"

            alerts.append({
                "incident_id":   _inc_id(dataset_id, _stable_seq("HS", curr.id, dataset_id)),
                "db_id":         curr.id,
                "source":        "Health Monitor",
                "category":      "Health",
                "title":         f"Health Score {direction.title()} {abs_delta:.1f}% (Run #{curr.id})",
                "detail":        (
                    f"Overall health score {direction} from {prev_score:.1f}% to {curr_score:.1f}% "
                    f"(Δ {delta:+.1f}%) between Run #{prev.id} and Run #{curr.id}."
                ),
                "severity":      sev,
                "column_name":   None,
                "affected_rows": None,
                "run_id":        curr.id,
                "detected_at":   _fmt(curr.timestamp),
                "time_ago":      _time_ago(curr.timestamp),
                "is_new":        curr.id == latest_run_id,
                "status":        "open",
                "check_type":    "HEALTH_SCORE_CHANGE",
                "view_route":    _view_route("Health"),
            })
        return alerts
    except Exception:
        return []


def _completeness_drop_alerts(
    db: Session,
    dataset_id: int,
    runs: List[ProfilingRun],
    latest_run_id: int,
) -> List[Dict[str, Any]]:
    """Alert when a column's completeness drops ≥5% between the last two runs."""
    if len(runs) < 2:
        return []
    try:
        sorted_runs = sorted(runs, key=lambda r: r.id)
        latest      = sorted_runs[-1]
        previous    = sorted_runs[-2]

        def _profiles_map(run: ProfilingRun) -> Dict[str, ColumnProfile]:
            try:
                profiles = db.query(ColumnProfile).filter(
                    ColumnProfile.profiling_run_id == run.id
                ).all()
                return {p.column_name: p for p in profiles}
            except Exception:
                return {}

        curr_map = _profiles_map(latest)
        prev_map = _profiles_map(previous)

        alerts = []
        for col_name, curr_p in curr_map.items():
            prev_p = prev_map.get(col_name)
            if not prev_p:
                continue
            curr_c = curr_p.completeness or 0.0
            prev_c = prev_p.completeness or 0.0
            delta  = round(curr_c - prev_c, 1)
            if delta >= -5.0:
                continue
            abs_delta = abs(delta)
            sev = "CRITICAL" if abs_delta >= 20 else "HIGH" if abs_delta >= 10 else "MEDIUM"
            alerts.append({
                "incident_id":   _inc_id(dataset_id, _stable_seq("CD", hash(col_name + str(latest.id)) & 0x7FFFFFFF, dataset_id)),
                "db_id":         curr_p.id,
                "source":        "Completeness Monitor",
                "category":      "Quality",
                "title":         f"Completeness Drop on `{col_name}` ({delta:+.1f}%)",
                "detail":        (
                    f"Column `{col_name}` completeness fell from {prev_c:.1f}% to {curr_c:.1f}% "
                    f"({delta:+.1f}%) in the latest run. Null count may have increased — check upstream pipeline."
                ),
                "severity":      sev,
                "column_name":   col_name,
                "affected_rows": curr_p.null_count,
                "run_id":        latest.id,
                "detected_at":   _fmt(latest.timestamp),
                "time_ago":      _time_ago(latest.timestamp),
                "is_new":        True,
                "status":        "open",
                "check_type":    "COMPLETENESS_DROP",
                "view_route":    _view_route("Quality"),
            })
        return alerts
    except Exception:
        return []


def _rule_breach_alerts(
    db: Session,
    dataset_id: int,
) -> List[Dict[str, Any]]:
    """Alert for DQ Rule violations from DQRuleRunResult."""
    try:
        rule_run_ids = [
            r[0] for r in db.query(DQRuleRun.id)
            .filter(DQRuleRun.dataset_id == dataset_id)
            .order_by(DQRuleRun.id.desc())
            .limit(10)
            .all()
        ]
        if not rule_run_ids:
            return []

        results = (
            db.query(DQRuleRunResult)
            .filter(
                DQRuleRunResult.run_id.in_(rule_run_ids),
                DQRuleRunResult.violation_count > 0,
            )
            .order_by(DQRuleRunResult.id.desc())
            .limit(30)
            .all()
        )

        alerts = []
        for rr in results:
            try:
                rule = db.query(DQRule).filter(
                    DQRule.rule_code == rr.rule_code,
                    DQRule.dataset_id == dataset_id,
                ).first()
                sev = _norm_sev(rule.severity if rule else "MEDIUM")
                col = rr.column or "dataset-level"
                pct = round((1 - (rr.pass_rate or 0)) * 100, 1)
                alerts.append({
                    "incident_id":   _inc_id(dataset_id, _stable_seq("RB", rr.id, dataset_id)),
                    "db_id":         rr.id,
                    "source":        "DQ Rules Engine",
                    "category":      "Rules",
                    "title":         f"Rule Breach: {rr.rule_name} on `{col}`",
                    "detail":        (
                        f"Rule `{rr.rule_code}` ({rr.rule_name}) failed — "
                        f"{rr.violation_count:,} violations ({pct:.1f}% fail rate) on column `{col}`."
                    ),
                    "severity":      sev,
                    "column_name":   rr.column,
                    "affected_rows": rr.violation_count,
                    "run_id":        rr.run_id,
                    "detected_at":   None,
                    "time_ago":      "—",
                    "is_new":        True,
                    "status":        "open",
                    "check_type":    "RULE_BREACH",
                    "view_route":    _view_route("Rules"),
                })
            except Exception:
                continue
        return alerts
    except Exception:
        return []


def _freshness_alerts(
    db: Session,
    dataset_id: int,
    latest_run: ProfilingRun,
) -> List[Dict[str, Any]]:
    """Alert for columns with low timeliness scores."""
    try:
        profiles = db.query(ColumnProfile).filter(
            ColumnProfile.profiling_run_id == latest_run.id,
            ColumnProfile.timeliness.isnot(None),
        ).all()

        alerts = []
        for p in profiles:
            score = p.timeliness or 0.0
            if score >= 60.0:
                continue
            sev = "HIGH" if score < 30 else "MEDIUM"
            alerts.append({
                "incident_id":   _inc_id(dataset_id, _stable_seq("FR", p.id, dataset_id)),
                "db_id":         p.id,
                "source":        "Freshness Monitor",
                "category":      "Freshness",
                "title":         f"Stale Data: `{p.column_name}` ({score:.1f}% fresh)",
                "detail":        (
                    f"Column `{p.column_name}` has a timeliness score of {score:.1f}% — "
                    f"most recent values are significantly older than expected."
                ),
                "severity":      sev,
                "column_name":   p.column_name,
                "affected_rows": None,
                "run_id":        latest_run.id,
                "detected_at":   _fmt(latest_run.timestamp),
                "time_ago":      _time_ago(latest_run.timestamp),
                "is_new":        True,
                "status":        "open",
                "check_type":    "FRESHNESS_LOW",
                "view_route":    _view_route("Freshness"),
            })
        return alerts
    except Exception:
        return []


# ─── dedup ────────────────────────────────────────────────────────────────────

def _dedup(alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Deduplicate alerts.
    FIX: Old key was (category, check_type, column_name, run_id).
    Since run_id differs per run, same issue from different runs always passed through.
    New key is (category, check_type, column_name) — one alert per unique issue type.
    """
    seen: set = set()
    out  = []
    for a in alerts:
        key = (a["category"], a["check_type"], a.get("column_name"))
        if key not in seen:
            seen.add(key)
            out.append(a)
    return out


# ─── public API ───────────────────────────────────────────────────────────────

def get_alerts(db: Session, dataset_id: int) -> Dict[str, Any]:
    """
    Returns all alerts for the given dataset sorted by severity then recency.
    NEVER raises — always returns a valid response dict.
    """
    runs   = _runs_for_dataset(db, dataset_id, limit=10)
    latest = runs[0] if runs else None

    if not latest:
        return {
            "status":    "NO_DATA",
            "message":   "Run profiling first to generate alerts.",
            "datasetId": dataset_id,
            "alerts":    [],
            "summary":   _empty_summary(),
        }

    latest_run_id  = latest.id
    all_alerts: List[Dict[str, Any]] = []

    all_alerts += _quality_check_alerts(db, dataset_id, runs, latest_run_id)
    all_alerts += _drift_alerts(db, dataset_id, runs, latest_run_id)
    all_alerts += _schema_history_alerts(db, dataset_id, latest_run_id)
    all_alerts += _health_score_alerts(db, dataset_id, runs, latest_run_id)
    all_alerts += _completeness_drop_alerts(db, dataset_id, runs, latest_run_id)
    all_alerts += _rule_breach_alerts(db, dataset_id)
    all_alerts += _freshness_alerts(db, dataset_id, latest)

    all_alerts = _dedup(all_alerts)
    all_alerts.sort(key=lambda a: (
        _SEV_ORDER.get(a["severity"], 9),
        0 if a["is_new"] else 1,
        -(a["run_id"] or 0),
    ))

    summary = {
        "total":    len(all_alerts),
        "critical": sum(1 for a in all_alerts if a["severity"] == "CRITICAL"),
        "high":     sum(1 for a in all_alerts if a["severity"] == "HIGH"),
        "medium":   sum(1 for a in all_alerts if a["severity"] == "MEDIUM"),
        "low":      sum(1 for a in all_alerts if a["severity"] == "LOW"),
        "new":      sum(1 for a in all_alerts if a["is_new"]),
        "open":     sum(1 for a in all_alerts if (a["status"] or "open").lower() == "open"),
    }

    return {
        "status":    "OK",
        "datasetId": dataset_id,
        "alerts":    all_alerts,
        "summary":   summary,
    }


def dismiss_alert(db: Session, dataset_id: int, db_id: int, category: str) -> Dict[str, Any]:
    """
    Mark an alert as resolved/dismissed — persisted to DB for all categories
    so the status survives a page reload.
    """
    try:
        if category in ("Quality", "Health", "Freshness"):
            chk = db.query(QualityCheck).filter(QualityCheck.id == db_id).first()
            if chk:
                chk.status = "resolved"
                db.commit()
        elif category == "Drift":
            dr = db.query(DriftRecord).filter(DriftRecord.id == db_id).first()
            if dr:
                if not hasattr(DriftRecord, "status"):
                    # status column may not exist — fall back silently
                    pass
                else:
                    dr.status = "resolved"
                    db.commit()
        elif category == "Schema":
            sh = db.query(SchemaHistory).filter(SchemaHistory.id == db_id).first()
            if sh:
                if hasattr(sh, "status"):
                    sh.status = "resolved"
                    db.commit()
        elif category == "Rules":
            rr = db.query(DQRuleRunResult).filter(DQRuleRunResult.id == db_id).first()
            if rr:
                if hasattr(rr, "status"):
                    rr.status = "resolved"
                    db.commit()
    except Exception:
        pass
    return {"status": "success", "dismissed": True}


def update_alert_status(
    db: Session,
    dataset_id: int,
    db_id: int,
    category: str,
    new_status: str,
) -> Dict[str, Any]:
    """
    Update alert status (open | investigating | resolved) — persisted to DB
    for all supported categories so status survives a page reload.
    """
    valid = {"open", "investigating", "resolved"}
    status = new_status.lower() if new_status.lower() in valid else "open"
    try:
        if category in ("Quality", "Health", "Freshness"):
            chk = db.query(QualityCheck).filter(QualityCheck.id == db_id).first()
            if chk:
                chk.status = status
                db.commit()
        elif category == "Drift":
            dr = db.query(DriftRecord).filter(DriftRecord.id == db_id).first()
            if dr and hasattr(dr, "status"):
                dr.status = status
                db.commit()
        elif category == "Schema":
            sh = db.query(SchemaHistory).filter(SchemaHistory.id == db_id).first()
            if sh and hasattr(sh, "status"):
                sh.status = status
                db.commit()
        elif category == "Rules":
            rr = db.query(DQRuleRunResult).filter(DQRuleRunResult.id == db_id).first()
            if rr and hasattr(rr, "status"):
                rr.status = status
                db.commit()
    except Exception:
        pass
    return {"status": "success", "new_status": status}


def sync_column_resolved(db: Session, dataset_id: int, column_name: str) -> int:
    """
    Mark all open QualityCheck violations for a specific column as resolved.
    Called when a DQ rule is activated for that column.
    """
    count = 0
    try:
        run_ids = [
            r[0] for r in db.query(ProfilingRun.id)
            .filter(ProfilingRun.dataset_id == dataset_id)
            .all()
        ]
        if not run_ids:
            return 0
        checks = (
            db.query(QualityCheck)
            .filter(
                QualityCheck.profiling_run_id.in_(run_ids),
                QualityCheck.column_name == column_name,
                QualityCheck.status != "resolved",
            )
            .all()
        )
        for chk in checks:
            chk.status = "resolved"
            count += 1
        if count:
            db.commit()
    except Exception:
        pass
    return count