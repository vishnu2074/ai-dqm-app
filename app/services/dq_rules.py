# python-backend/app/services/dq_rules.py
from __future__ import annotations
 
from datetime import datetime, timezone, date
from typing import Any, Dict, List, Optional, Tuple
from types import SimpleNamespace
import re
import json
from app.services.ai_rule_generator import generate_ai_rules
from sqlalchemy.orm import Session
import pandas as pd
 
from app.models import Dataset, ProfilingRun, ColumnProfile, DQRule, DQRuleChangeLog
from app.services import dq_engine as dq_engine_service  # noqa: F401
from app.services.dq_engine import (
    _get_version,
    _resolve_to_local_file,
    _read_dataset,
    _apply_rule,
)  # noqa
 
 
SEVERITY_WEIGHT = {
    "Critical": 1.0,
    "High": 0.8,
    "Medium": 0.55,
    "Low": 0.35,
}
 
ALLOWED_TYPES = {"Validity", "Completeness", "Uniqueness", "Consistency", "Accuracy", "Timeliness", "Integrity"}
ALLOWED_STATUS = {"Active", "Paused", "Failed"}
ALLOWED_SEVERITY = {"Critical", "High", "Medium", "Low"}
 
 
def _humanize_time_ago(ts: datetime) -> str:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    delta = now - ts
    sec = int(delta.total_seconds())
    if sec < 60:
        return "just now"
    mins = sec // 60
    if mins < 60:
        return f"{mins} mins ago" if mins != 1 else "1 min ago"
    hrs = mins // 60
    if hrs < 24:
        return f"{hrs} hours ago" if hrs != 1 else "1 hour ago"
    days = hrs // 24
    return f"{days} days ago" if days != 1 else "1 day ago"
 
 
def _latest_completed_run(db: Session, dataset_id: int) -> Optional[ProfilingRun]:
    return (
        db.query(ProfilingRun)
        .filter(ProfilingRun.dataset_id == dataset_id)
        .filter(ProfilingRun.status == "COMPLETED")
        .order_by(ProfilingRun.timestamp.desc())
        .first()
    )
 
 
def _get_column_profiles(db: Session, run_id: int) -> List[ColumnProfile]:
    return db.query(ColumnProfile).filter(ColumnProfile.profiling_run_id == run_id).all()
 
 
def _metric_for_rule(profile: ColumnProfile, rule_type: str) -> float:
    rt = (rule_type or "").lower()
    if rt == "completeness":
        return float(profile.completeness or 0.0)
    if rt == "uniqueness":
        return float(profile.uniqueness or 0.0)
    if rt == "validity":
        return float(profile.validity or 0.0)
    if rt == "consistency":
        return float(profile.consistency or 0.0)
    if rt == "accuracy":
        return float(profile.accuracy or 0.0)
    if rt == "timeliness":
        return float(profile.timeliness or 0.0)
    if rt == "integrity":
        return float(profile.integrity or 0.0)
    return float(profile.health_score or 0.0)
 
 
def _violations_for_rule(profile: ColumnProfile, rule_type: str, total_rows: int) -> int:
    rt = (rule_type or "").lower()
    if rt == "completeness":
        return int(profile.null_count or 0)
 
    if rt in ("uniqueness", "integrity"):
        non_null = max(0, total_rows - int(profile.null_count or 0))
        distinct = int(profile.distinct_count or 0)
        dup = max(0, non_null - distinct)
        return int(dup)
 
    pass_rate = _metric_for_rule(profile, rule_type)
    return int(max(0, round(total_rows * (1.0 - (pass_rate / 100.0)))))
 
 
def _impact_score(severity: str, violations: int, total_rows: int, pass_rate: float) -> int:
    w = SEVERITY_WEIGHT.get(severity, 0.55)
    sev_part = 60.0 * w
    qual_part = max(0.0, (100.0 - pass_rate)) * 0.25
    ratio = (violations / total_rows) if total_rows else 0.0
    viol_part = min(40.0, ratio * 4000.0)
    score = sev_part + qual_part + viol_part
    return int(round(min(100.0, score)))
 
 
def _normalize_patterns(pats: Any) -> List[str]:
    if pats is None:
        return []
    if isinstance(pats, list):
        return [str(x) for x in pats]
    try:
        if isinstance(pats, str) and pats.strip().startswith("["):
            arr = json.loads(pats)
            if isinstance(arr, list):
                return [str(x) for x in arr]
    except Exception:
        pass
    return [str(pats)]
 
 
def _next_rule_code(db: Session, dataset_id: int) -> str:
    rows = db.query(DQRule.rule_code).filter(DQRule.dataset_id == dataset_id).all()
    mx = 0
    for (code,) in rows:
        if not code:
            continue
        m = re.match(r"^RULE-(\d{3,})$", code.strip().upper())
        if m:
            mx = max(mx, int(m.group(1)))
    return f"RULE-{mx+1:03d}"
 
 
def _get_known_columns(db: Session, dataset_id: int) -> List[str]:
    run = _latest_completed_run(db, dataset_id)
    if not run:
        return []
    cols = _get_column_profiles(db, run.id)
    return [c.column_name for c in cols if c.column_name]
 
 
def _guess_column_from_text(columns: List[str], text: str) -> Optional[str]:
    t = (text or "").lower()
    for c in columns:
        if c and c.lower() in t:
            return c
    tokens = set(re.findall(r"[a-zA-Z0-9_]+", t))
    best, best_score = None, 0
    for c in columns:
        ct = set(re.findall(r"[a-zA-Z0-9_]+", c.lower()))
        score = len(tokens & ct)
        if score > best_score:
            best_score, best = score, c
    return best if best_score > 0 else None
 
 
def _nl_to_condition(columns: List[str], nl: str, explicit_column: Optional[str] = None) -> Tuple[str, Optional[str], str]:
    text = (nl or "").strip()
    if not text:
        raise ValueError("Natural language rule text is empty.")
 
    col = explicit_column or _guess_column_from_text(columns, text)
    if not col:
        raise ValueError("Could not infer column from text. Please specify the column.")
 
    t = text.lower().strip()
 
    if any(k in t for k in ["not null", "non null", "cannot be null", "must not be null", "should not be null", "required"]):
        return "NOT NULL", col, "Completeness"
 
    if any(k in t for k in ["unique", "no duplicates", "distinct"]):
        return "DISTINCT_COUNT = TOTAL_COUNT", col, "Uniqueness"
 
    if any(k in t for k in ["not in future", "no future", "before today", "less than today", "must be <= current date", "must be before current date"]):
        return f"{col} <= CURRENT_DATE", col, "Consistency"
 
    m = re.search(r"(?:greater than|more than|>|>=)\s*([0-9]+(?:\.[0-9]+)?)", t)
    if m:
        val = m.group(1)
        return f"{col} > {val}", col, "Validity"
 
    m = re.search(r"(?:between|range)\s*([0-9]+(?:\.[0-9]+)?)\s*(?:and|to)\s*([0-9]+(?:\.[0-9]+)?)", t)
    if m:
        lo, hi = m.group(1), m.group(2)
        return f"{col} > {lo} AND {col} < {hi}", col, "Validity"
 
    m = re.search(r"(?:match|matches|regex|pattern)\s*[:=]?\s*(\^.*$)", text.strip(), flags=re.IGNORECASE)
    if m:
        pattern = m.group(1).strip().replace('"', '\\"')
        return f'REGEX_MATCH({col}, "{pattern}")', col, "Validity"
 
    raise ValueError(
        "Could not translate natural language into a supported rule. "
        "Try: 'customer_id must be unique', 'email must match <regex>', 'income > 0', "
        "'signup_date not in future', 'city not null'."
    )
 
 
def _compute_rules_for_run(db: Session, dataset_id: int, run: ProfilingRun, rules: List[DQRule]) -> List[Dict[str, Any]]:
    profiles = _get_column_profiles(db, run.id)
    total_rows = int(run.rows_processed or 0)
    prof_by_col = {p.column_name: p for p in profiles}
 
    out: List[Dict[str, Any]] = []
    for r in rules:
        prof = prof_by_col.get(r.column)
        pass_rate = _metric_for_rule(prof, r.type) if prof else 0.0
        violations = _violations_for_rule(prof, r.type, total_rows) if prof else 0
        impact = _impact_score(r.severity, violations, total_rows, pass_rate)
        out.append({
            "id": r.rule_code,
            "name": r.name,
            "type": r.type,
            "column": r.column,
            "condition": r.condition,
            "severity": r.severity,
            "status": r.status,
            "violations": violations,
            "lastRun": _humanize_time_ago(run.timestamp),
            "passRate": round(float(pass_rate), 1),
            "impactScore": impact,
        })
    return out
 
 
def get_active_rules(db: Session, dataset_id: int) -> Dict[str, Any]:
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise ValueError("Dataset not found")
 
    run = _latest_completed_run(db, dataset_id)
    if not run:
        return {"status": "NO_DATA", "message": "Run profiling first to generate rule metrics.", "rules": []}
 
    rules = (
        db.query(DQRule)
        .filter(DQRule.dataset_id == dataset_id)
        .filter(DQRule.status.in_(["Active", "Paused"]))
        .order_by(DQRule.id.asc())
        .all()
    )
 
    out = _compute_rules_for_run(db, dataset_id, run, rules)
    return {"status": "success", "runId": run.id, "rules": out}
 
 
def get_rules_summary(db: Session, dataset_id: int, trend_points: int = 8) -> Dict[str, Any]:
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise ValueError("Dataset not found")
 
    run = _latest_completed_run(db, dataset_id)
    if not run:
        return {
            "status": "NO_DATA",
            "totalRules": 0,
            "avgPassRate": 0.0,
            "avgPassRateDelta": 0.0,
            "activeViolations": 0,
            "aiRecommendations": 0,
            "performanceData": [],
        }
 
    profiles = _get_column_profiles(db, run.id)
    total_rows = int(run.rows_processed or 0)
    prof_by_col = {p.column_name: p for p in profiles}
 
    rules = (
        db.query(DQRule)
        .filter(DQRule.dataset_id == dataset_id)
        .filter(DQRule.status == "Active")
        .order_by(DQRule.id.asc())
        .all()
    )
 
    total_rules = len(rules)
 
    if total_rows == 0 or total_rules == 0:
        return {
            "status": "success",
            "totalRules": total_rules,
            "avgPassRate": 0.0,
            "avgPassRateDelta": 0.0,
            "activeViolations": 0,
            "aiRecommendations": 0,
            "performanceData": [],
        }
 
    total_violations = sum(
        _violations_for_rule(prof_by_col.get(r.column), r.type, total_rows)
        for r in rules
        if prof_by_col.get(r.column)
    )
 
    overall_pass_rate = round(
        max(0.0, 100.0 * (1.0 - (total_violations / (total_rows * total_rules)))),
        1,
    )
 
    runs = (
        db.query(ProfilingRun)
        .filter(ProfilingRun.dataset_id == dataset_id)
        .filter(ProfilingRun.status == "COMPLETED")
        .order_by(ProfilingRun.timestamp.desc())
        .limit(trend_points)
        .all()
    )
    runs = list(reversed(runs))
 
    perf: List[Dict[str, Any]] = []
    for r in runs:
        cols = _get_column_profiles(db, r.id)
        prof_map = {p.column_name: p for p in cols}
        rows_r = int(r.rows_processed or 0)
        if rows_r == 0:
            continue
 
        total_viol = sum(
            _violations_for_rule(prof_map.get(rule.column), rule.type, rows_r)
            for rule in rules
            if prof_map.get(rule.column)
        )
 
        pass_rate_r = round(
            max(0.0, 100.0 * (1.0 - (total_viol / (rows_r * total_rules)))),
            1,
        )
 
        perf.append({
            "date": r.timestamp.strftime("%b %d"),
            "passRate": pass_rate_r
        })
 
    delta = round(perf[-1]["passRate"] - perf[-2]["passRate"], 1) if len(perf) >= 2 else 0.0
 
    return {
        "status": "success",
        "totalRules": total_rules,
        "avgPassRate": overall_pass_rate,
        "avgPassRateDelta": delta,
        "activeViolations": total_violations,
        "aiRecommendations": 0,
        "performanceData": perf,
    }
 
 
def get_rule_history(db: Session, dataset_id: int, limit: int = 50) -> Dict[str, Any]:
    rows = (
        db.query(DQRuleChangeLog)
        .filter(DQRuleChangeLog.dataset_id == dataset_id)
        .order_by(DQRuleChangeLog.change_date.desc())
        .limit(limit)
        .all()
    )
 
    live_rules = {
        r.rule_code: r
        for r in db.query(DQRule).filter(DQRule.dataset_id == dataset_id).all()
    }
 
    history = []
    for h in rows:
        live = live_rules.get(h.rule_code)
        history.append({
            "id": h.rule_code,
            "name": h.rule_name,
            "version": h.version,
            "changedBy": h.changed_by,
            "changeDate": h.change_date.strftime("%d %b %Y, %H:%M") if h.change_date else "—",
            "changeType": h.change_type,
            "performance": h.performance_delta,
            "ruleType":   live.type      if live else None,
            "column":     live.column    if live else None,
            "condition":  live.condition if live else None,
            "severity":   live.severity  if live else None,
            "currentStatus": live.status if live else "Deleted",
        })
    return {"status": "success", "history": history}
 
 
# ============================
# ✅ Performance Δ + Version helpers
# ============================
 
def _format_perf_delta(delta_pp: float) -> str:
    d = float(delta_pp)
    if d > 0:
        return f"+{d:.1f}%"
    if d < 0:
        return f"{d:.1f}%"
    return "0.0%"
 
 
def _parse_version(v: Optional[str]) -> Tuple[int, int]:
    if not v:
        return (1, 0)
    m = re.match(r"^v?(\d+)\.(\d+)$", str(v).strip().lower())
    if not m:
        return (1, 0)
    return (int(m.group(1)), int(m.group(2)))
 
 
def _next_version_for_rule(db: Session, dataset_id: int, rule_code: str) -> str:
    last = (
        db.query(DQRuleChangeLog)
        .filter(DQRuleChangeLog.dataset_id == dataset_id)
        .filter(DQRuleChangeLog.rule_code == rule_code)
        .order_by(DQRuleChangeLog.change_date.desc())
        .first()
    )
    major, minor = _parse_version(last.version if last else None)
    return f"v{major}.{minor + 1}"
 
 
def _load_latest_dataset_df(db: Session, dataset_id: int) -> Optional[pd.DataFrame]:
    ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not ds:
        return None
    try:
        version = _get_version(db, dataset_id, version_id=None)
        local_path = _resolve_to_local_file(db, ds, version.file_path)
        return _read_dataset(local_path)
    except Exception:
        return None
 
 
def _pass_rate_for_rule_in_latest_run(db: Session, dataset_id: int, rule_type: str, column: str, condition: str) -> float:
    df = _load_latest_dataset_df(db, dataset_id)
    if df is None or df.empty:
        return 0.0
 
    dummy_rule = SimpleNamespace(type=rule_type, column=column, condition=condition)
    mask, err = _apply_rule(df, dummy_rule)
    if err is not None:
        return 0.0
 
    try:
        return float(round(mask.mean() * 100.0, 2))
    except Exception:
        return 0.0
 
 
def update_rule(db: Session, dataset_id: int, rule_code: str, updates: Dict[str, Any]) -> Dict[str, Any]:
    rule = (
        db.query(DQRule)
        .filter(DQRule.dataset_id == dataset_id)
        .filter(DQRule.rule_code == rule_code)
        .first()
    )
    if not rule:
        raise ValueError("Rule not found")
 
    if "severity" in updates and updates["severity"] is not None and updates["severity"] not in ALLOWED_SEVERITY:
        raise ValueError(f"severity must be one of: {', '.join(sorted(ALLOWED_SEVERITY))}")
    if "status" in updates and updates["status"] is not None and updates["status"] not in ALLOWED_STATUS:
        raise ValueError(f"status must be one of: {', '.join(sorted(ALLOWED_STATUS))}")
    if "type" in updates and updates["type"] is not None and updates["type"] not in ALLOWED_TYPES:
        raise ValueError(f"type must be one of: {', '.join(sorted(ALLOWED_TYPES))}")
 
    old_pass = _pass_rate_for_rule_in_latest_run(
        db, dataset_id, rule_type=rule.type, column=rule.column, condition=rule.condition
    )
 
    allowed = {"name", "type", "column", "condition", "severity", "status"}
    for k, v in updates.items():
        if k in allowed and v is not None:
            setattr(rule, k, v)
 
    new_pass = _pass_rate_for_rule_in_latest_run(
        db, dataset_id, rule_type=rule.type, column=rule.column, condition=rule.condition
    )
 
    only_status_changed = (set(updates.keys()) == {"status"}) or (len(updates) == 1 and "status" in updates)
 
    if only_status_changed:
        new_status = updates.get("status", rule.status)
        if new_status == "Paused":
            change_type = "Paused"
            perf_delta = f"paused at {old_pass:.1f}%"
        else:
            change_type = "Activated"
            perf_delta = f"resumed at {new_pass:.1f}%"
    else:
        change_type = "Modified"
        delta = round(new_pass - old_pass, 1)
        perf_delta = _format_perf_delta(delta)
 
    db.add(DQRuleChangeLog(
        dataset_id=dataset_id,
        rule_code=rule.rule_code,
        rule_name=rule.name,
        version=_next_version_for_rule(db, dataset_id, rule.rule_code),
        changed_by="Admin",
        change_type=change_type,
        performance_delta=perf_delta,
        change_date=datetime.now(timezone.utc),
    ))
 
    db.commit()
    db.refresh(rule)
    try:
        from app.routers.notification_inbox_routes import create_inbox_notification
        ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        ds_name = (ds.display_name or ds.physical_name or f"Dataset {dataset_id}") if ds else f"Dataset {dataset_id}"
 
        create_inbox_notification(
            title=f"DQ Rule Updated: {rule.name}",
            message=f"Rule '{rule.name}' was {change_type.lower()} on dataset '{ds_name}'. {perf_delta}",
            category="rule",
            severity="info",
            link=f"/dq-rules",
            dataset=ds_name,
            dataset_id=dataset_id,
        )
    except Exception:
        pass
   
    return {"status": "success", "rule": {
        "id": rule.rule_code,
        "name": rule.name,
        "type": rule.type,
        "column": rule.column,
        "condition": rule.condition,
        "severity": rule.severity,
        "status": rule.status,
    }}
 
 
def delete_rule(db: Session, dataset_id: int, rule_code: str) -> Dict[str, Any]:
    rule = (
        db.query(DQRule)
        .filter(DQRule.dataset_id == dataset_id)
        .filter(DQRule.rule_code == rule_code)
        .first()
    )
    if not rule:
        raise ValueError("Rule not found")
 
    rule_name = rule.name
 
    last_pass = _pass_rate_for_rule_in_latest_run(
        db, dataset_id, rule_type=rule.type, column=rule.column, condition=rule.condition
    )
    perf_note = f"was {last_pass:.1f}%" if last_pass > 0 else "N/A"
 
    db.delete(rule)
 
    db.add(DQRuleChangeLog(
        dataset_id=dataset_id,
        rule_code=rule_code,
        rule_name=rule_name,
        version=_next_version_for_rule(db, dataset_id, rule_code),
        changed_by="Admin",
        change_type="Deleted",
        performance_delta=perf_note,
        change_date=datetime.now(timezone.utc),
    ))
 
   
    db.commit()
 
    try:
        from app.routers.notification_inbox_routes import create_inbox_notification
        ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        ds_name = (ds.display_name or ds.physical_name or f"Dataset {dataset_id}") if ds else f"Dataset {dataset_id}"
 
        create_inbox_notification(
            title=f"DQ Rule Deleted: {rule_name}",
            message=f"Rule '{rule_name}' was permanently deleted from dataset '{ds_name}'.",
            category="rule",
            severity="warning",
            link=f"/dq-rules",
            dataset=ds_name,
            dataset_id=dataset_id,
        )
    except Exception:
        pass
 
    return {"status": "success", "deleted": True, "rule_code": rule_code}
   
 
def create_rule(
    db: Session,
    dataset_id: int,
    *,
    input_mode: str,
    text: str,
    name: Optional[str] = None,
    rule_type: Optional[str] = None,
    column: Optional[str] = None,
    severity: str = "Medium",
    status: str = "Active",
) -> Dict[str, Any]:
    ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not ds:
        raise ValueError("Dataset not found")
 
    mode = (input_mode or "").lower().strip()
    if mode not in {"nl", "regex", "dsl"}:
        raise ValueError("input_mode must be one of: nl, regex, dsl")
 
    if severity not in ALLOWED_SEVERITY:
        raise ValueError(f"severity must be one of: {', '.join(sorted(ALLOWED_SEVERITY))}")
    if status not in ALLOWED_STATUS:
        raise ValueError(f"status must be one of: {', '.join(sorted(ALLOWED_STATUS))}")
 
    columns = _get_known_columns(db, dataset_id)
 
    inferred_type = "Validity"
    condition = ""
    nl_text = None
    regex_pattern = None
 
    if mode == "regex":
        if not column:
            col_guess = _guess_column_from_text(columns, text)
            if not col_guess:
                raise ValueError("For regex rules, please specify the target column.")
            column = col_guess
 
        pattern = (text or "").strip()
        if not pattern:
            raise ValueError("Regex pattern is empty.")
 
        regex_pattern = pattern
        pattern_escaped = pattern.replace('"', '\\"')
        condition = f'REGEX_MATCH({column}, "{pattern_escaped}")'
        inferred_type = "Validity"
 
    elif mode == "dsl":
        condition = (text or "").strip()
        if not condition:
            raise ValueError("Condition (DSL) is empty.")
        if not column:
            column = _guess_column_from_text(columns, condition) or ""
        inferred_type = rule_type or "Validity"
 
    else:  # nl
        nl_text = (text or "").strip()
        condition, resolved_col, inferred_type = _nl_to_condition(columns, text, explicit_column=column)
        column = resolved_col or column or ""
 
    final_type = (rule_type or inferred_type or "Validity").strip()
    if final_type not in ALLOWED_TYPES:
        raise ValueError(f"type must be one of: {', '.join(sorted(ALLOWED_TYPES))}")
 
    if not column:
        raise ValueError("Could not resolve rule column. Please specify the column.")
 
    code = _next_rule_code(db, dataset_id)
    final_name = (name or "").strip() or f"{column} Rule"
 
    rule = DQRule(
        dataset_id=dataset_id,
        rule_code=code,
        name=final_name,
        type=final_type,
        column=column,
        condition=condition,
        severity=severity,
        status=status,
        input_mode=mode,
        nl_text=nl_text,
        regex_pattern=regex_pattern,
        meta=None,
    )
    db.add(rule)
 
    db.add(DQRuleChangeLog(
        dataset_id=dataset_id,
        rule_code=code,
        rule_name=final_name,
        version="v1.0",
        changed_by="Admin",
        change_type="Created",
        performance_delta="N/A",
        change_date=datetime.now(timezone.utc),
    ))
 
    db.commit()
    db.refresh(rule)# Notify
 
    try:
        from app.routers.notification_inbox_routes import create_inbox_notification
        ds_name = ds.display_name or ds.physical_name or f"Dataset {dataset_id}"
        create_inbox_notification(
            title=f"DQ Rule Created: {final_name}",
            message=f"New {final_type} rule on column '{column}' added to dataset '{ds_name}'. Severity: {severity}.",
            category="rule",
            severity="info" if severity.lower() in ("low", "medium") else "warning",
            link=f"/dq-rules",
            dataset=ds_name,
            dataset_id=dataset_id,
        )
    except Exception as _ne:
        pass
   
 
    return {"status": "success", "rule": {
        "id": rule.rule_code,
        "name": rule.name,
        "type": rule.type,
        "column": rule.column,
        "condition": rule.condition,
        "severity": rule.severity,
        "status": rule.status,
    }}
 
 
def approve_ai_recommended_rule(db: Session, dataset_id: int, rule_payload: Dict[str, Any]) -> Dict[str, Any]:
    ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not ds:
        raise ValueError("Dataset not found")
 
    rule = rule_payload or {}
 
    name = str(rule.get("name") or "").strip() or None
    rule_type = str(rule.get("type") or "").strip() or None
    column = str(rule.get("column") or "").strip() or None
    condition = str(rule.get("condition") or "").strip()
    severity = str(rule.get("severity") or "Medium").strip()
 
    if not condition:
        raise ValueError("AI rule condition is required.")
    if not rule_type:
        raise ValueError("AI rule type is required.")
    if not column:
        raise ValueError("AI rule column is required.")
 
    existing_pending_rules = (
        db.query(DQRule)
        .filter(DQRule.dataset_id == dataset_id)
        .filter(DQRule.input_mode == "ai")
        .filter(DQRule.status == "Pending Review")
        .filter(DQRule.type == rule_type)
        .filter(DQRule.column == column)
        .filter(DQRule.condition == condition)
        .all()
    )
 
    created = create_rule(
        db,
        dataset_id,
        input_mode="ai",
        text=condition,
        name=name,
        rule_type=rule_type,
        column=column,
        severity=severity,
        status="Active",
    )
 
    if existing_pending_rules:
        for pending_rule in existing_pending_rules:
            db.delete(pending_rule)
        db.commit()
 
    return created
 
 
# ----------------------------
# Discovered Rules (unchanged)
# ----------------------------
 
def _is_probably_id(colname: str) -> bool:
    c = (colname or "").lower()
    return c == "id" or c.endswith("_id") or "customer_id" in c
 
 
def _is_probably_date(colname: str) -> bool:
    c = (colname or "").lower()
    return "date" in c or "time" in c
 
 
def _looks_like_email_series(s) -> bool:
    try:
        sample = s.dropna().astype(str).head(200)
        if sample.empty:
            return False
        hits = sample.str.contains("@", regex=False).mean()
        return hits >= 0.3
    except Exception:
        return False
 
 
def _looks_like_phone_series(s) -> bool:
    try:
        sample = s.dropna().astype(str).head(200)
        if sample.empty:
            return False
        hits = sample.str.contains(r"\d{8,}", regex=True).mean()
        return hits >= 0.3
    except Exception:
        return False
 
 
def _rule_sig(rule_type: str, column: str, condition: str) -> str:
    return f"{(rule_type or '').strip().lower()}|{(column or '').strip().lower()}|{(condition or '').strip()}"
 
 
def get_discovered_rules(db: Session, dataset_id: int):
    ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
 
    if not ds:
        raise ValueError("Dataset not found")
 
    try:
        version = _get_version(db, dataset_id, version_id=None)
        local_path = _resolve_to_local_file(db, ds, version.file_path)
 
        # 🔹 load dataset
        df = _read_dataset(local_path)
 
    except Exception:
        return {
            "status": "success",
            "rules": [],
            "message": "Dataset could not be loaded"
        }
 
    if df is None or df.empty:
        return {
            "status": "success",
            "rules": [],
            "message": "Dataset is empty"
        }
 
    # 🔹 Fetch batch-generated rules
    rules = (
        db.query(DQRule)
        .filter(DQRule.dataset_id == dataset_id)
        .filter(DQRule.status == "Pending Review")
        .order_by(DQRule.id.asc())
        .all()
    )
 
    discovered_rules = []
 
    total_rows = len(df)
 
    for i, r in enumerate(rules):
        # 🔹 Apply rule using rule engine
        mask, error = _apply_rule(df, r)
 
        if error is not None or mask is None:
            occurrences = 0
            confidence = 0
        else:
            # mask = rows passing rule
            pass_count = int(mask.sum())
            violations = int(total_rows - pass_count)
 
            occurrences = violations
 
            if total_rows > 0:
                confidence = round((pass_count / total_rows) * 100, 2)
            else:
                confidence = 0
 
        discovered_rules.append({
            "id": r.rule_code or f"DISC-AI-{i+1:03d}",
            "name": r.name,
            "type": r.type,
            "column": r.column,
            "condition": r.condition,
            "confidence": confidence,
            "occurrences": occurrences,
            "status": "Pending Review"
        })
 
    return {
        "status": "success",
        "rules": discovered_rules
    }
 
 
def get_dataset_columns(db: Session, dataset_id: int) -> Dict[str, Any]:
    ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not ds:
        raise ValueError("Dataset not found")
 
    cols = _get_known_columns(db, dataset_id)
 
    # Fallback: if profiling not run, try reading the latest dataset header
    # (so UI still gets something if possible)
    if not cols:
        df = _load_latest_dataset_df(db, dataset_id)
        if df is not None and not df.empty:
            cols = [str(c) for c in df.columns.tolist()]
 
    # de-dup, stable order
    seen = set()
    out = []
    for c in cols:
        if c and c not in seen:
            seen.add(c)
            out.append(c)
 
    return {"status": "success", "columns": out}
 
 
def get_ai_recommended_rules(db: Session, dataset_id: int):
    ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
 
    if not ds:
        raise ValueError("Dataset not found")
 
    try:
        version = _get_version(db, dataset_id, version_id=None)
        local_path = _resolve_to_local_file(db, ds, version.file_path)
        df = _read_dataset(local_path)
 
    except Exception as e:
        print("⚠️ Dataset loading failed:", str(e))
        return {"status": "success", "rules": []}
 
    if df is None or df.empty:
        return {"status": "success", "rules": []}
 
    # remove previously generated AI rules for this dataset
    db.query(DQRule).filter(
        DQRule.dataset_id == dataset_id,
        DQRule.input_mode == "ai",
        DQRule.status == "Pending Review"
    ).delete(synchronize_session=False)
 
    db.commit()
 
    try:
        from app.routers.notification_inbox_routes import create_inbox_notification
 
        if recommended_rules:
            create_inbox_notification(
                title="AI Suggested New Rules",
                message=f"{len(recommended_rules)} new AI rules generated for dataset",
                category="ai",
                severity="info",
                link="/dq-rules",
                dataset=str(dataset_id),
                dataset_id=dataset_id,
            )
 
    except Exception:
        pass
 
    try:
        ai_rules = generate_ai_rules(df)
 
    except Exception as e:
        print("⚠️ AI generation crashed:", str(e))
        ai_rules = []
 
    # ensure AI rules format is valid
    if not isinstance(ai_rules, list):
        ai_rules = []
 
    dataset_columns = set(df.columns)
 
    recommended_rules = []
    seen_rules = set()
 
    for r in ai_rules:
        if not isinstance(r, dict):
            continue
 
        column = r.get("column")
 
        if column not in dataset_columns:
            continue
 
        rule_type = r.get("type")
        condition = r.get("condition")
 
        if not rule_type or not condition:
            continue
 
        rule_key = (rule_type, column, condition)
 
        if rule_key in seen_rules:
            continue
 
        seen_rules.add(rule_key)
 
        existing = (
            db.query(DQRule)
            .filter(DQRule.dataset_id == dataset_id)
            .filter(DQRule.type == rule_type)
            .filter(DQRule.column == column)
            .filter(DQRule.condition == condition)
            .first()
        )
 
        # If rule already exists AND is activated → skip showing again
        if existing and existing.status in ["Active", "Paused"]:
            continue
 
        rule_code = _next_rule_code(db, dataset_id)
 
        new_rule = DQRule(
            dataset_id=dataset_id,
            rule_code=rule_code,
            name=r.get("name", f"{column} validation rule"),
            type=rule_type,
            column=column,
            condition=condition,
            severity="Medium",
            status="Pending Review",
            input_mode="ai",
        )
 
        db.add(new_rule)
 
        recommended_rules.append({
            "id": rule_code,
            "name": r.get("name", f"{column} validation rule"),
            "type": rule_type,
            "column": column,
            "condition": condition,
            "confidence": float(r.get("confidence", 0.9)) * 100,
            "severity": "Medium"
        })
 
    db.commit()
 
    return {
        "status": "success",
        "rules": recommended_rules
    }