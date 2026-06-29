"""
AI DQM — Anomalies Service

FIXES:
  1. Uses requests HTTP directly to Azure AI Foundry (no openai SDK)
  2. Remediation suggestions are LLM-only, ordered by likelihood of resolving the issue
  3. fix_anomaly accepts an optional selected_remediation to generate a more targeted rule
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import time as _time
import requests as _http
from sqlalchemy import desc
from sqlalchemy.orm import Session

from dotenv import load_dotenv
load_dotenv()

from app.models import Dataset, DQRule, DQRuleChangeLog, ProfilingRun, QualityCheck
from app.services.llm_tracker import track_llm_call, status_code_suffix

# ─── Azure AI Foundry HTTP client ─────────────────────────────────────────────

_KEY      = os.getenv("AZURE_OPENAI_API_KEY", "")
# Strip any path from endpoint — keep only the base host
_RAW_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
# e.g. https://xyz.services.ai.azure.com/models/chat/completions?api-version=... → https://xyz.services.ai.azure.com
_ENDPOINT = _RAW_ENDPOINT.split("/models/")[0] if "/models/" in _RAW_ENDPOINT else _RAW_ENDPOINT
_MODEL    = os.getenv("AZURE_OPENAI_MODEL", "Llama-3.3-70B-Instruct")
_VERSION  = os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview")


def _llm(prompt: str, max_tokens: int = 800) -> Optional[str]:
    """Direct HTTP to Azure AI Foundry multi-model endpoint.
    Model name goes in the request body, not the URL.
    Correct URL format: {endpoint}/models/chat/completions?api-version=...
    """
    if not _KEY or not _ENDPOINT:
        print("[anomalies] LLM skipped: AZURE_OPENAI_API_KEY or AZURE_OPENAI_ENDPOINT not set")
        return None

    # Correct URL for AI Foundry: endpoint/models/chat/completions
    _ep = _ENDPOINT
    for _sfx in ["/chat/completions", "/models"]:
        while _ep.endswith(_sfx):
            _ep = _ep[:-len(_sfx)].rstrip("/")
    if "services.ai.azure.com" in _ep:
        url = f"{_ep}/models/chat/completions"
    else:
        url = f"{_ep}/chat/completions"
    headers = {"Content-Type": "application/json", "api-key": _KEY}
    payload = {
        "model":       _MODEL,
        "messages":    [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens":  max_tokens,
    }
    _t0 = _time.time()
    try:
        r = _http.post(url, headers=headers, json=payload,
                       timeout=40)
        r.raise_for_status()
        body = r.json()
        out = body["choices"][0]["message"]["content"]
        usage = body.get("usage", {})
        track_llm_call(
            feature="anomaly", model=_MODEL,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            latency_ms=(_time.time() - _t0) * 1000,
            success=True, input_length=len(prompt), output_length=len(out or ""),
        )
        return out
    except _http.exceptions.ConnectionError as e:
        track_llm_call(feature="anomaly", model=_MODEL,
            latency_ms=(_time.time() - _t0) * 1000, success=False,
            error_type="ConnectionError", input_length=len(prompt))
        print(f"[anomalies] LLM connection error: {e}")
    except _http.exceptions.Timeout:
        track_llm_call(feature="anomaly", model=_MODEL,
            latency_ms=(_time.time() - _t0) * 1000, success=False,
            error_type="Timeout", input_length=len(prompt))
        print("[anomalies] LLM request timed out")
    except Exception as e:
        track_llm_call(feature="anomaly", model=_MODEL,
            latency_ms=(_time.time() - _t0) * 1000, success=False,
            error_type=status_code_suffix(e), input_length=len(prompt))
        print(f"[anomalies] LLM error: {e}")
    return None


# ─── Static maps ─────────────────────────────────────────────────────────────

_CHECK_TYPE_LABELS: Dict[str, str] = {
    "UNPARSEABLE_DATES":          "Unparseable Dates",
    "FUTURE_DATES":               "Future Dates",
    "ANCIENT_DATES":              "Ancient Dates",
    "STALE_DATA":                 "Stale Data",
    "TEMPORAL_GAPS":              "Temporal Gaps",
    "DUPLICATE_TIMESTAMPS":       "Duplicate Timestamps",
    "SINGLE_DATE_DOMINANCE":      "Single Date Dominance",
    "WEEKEND_BUSINESS_DATES":     "Weekend / Business Date Mismatch",
    "ALL_NULLS":                  "All Nulls",
    "HIGH_NULL_RATE":             "High Null Rate",
    "CONSTANT_COLUMN":            "Constant Column",
    "NEAR_CONSTANT_COLUMN":       "Near-Constant Column",
    "HIGH_DUPLICATE_RATE":        "High Duplicate Rate",
    "SUSPICIOUS_WHITESPACE":      "Suspicious Whitespace",
    "MIXED_CASE_INCONSISTENCY":   "Mixed Case Inconsistency",
    "INVALID_EMAIL_FORMAT":       "Invalid Email Format",
    "UNEXPECTED_NEGATIVE_VALUES": "Unexpected Negative Values",
    "ZERO_DOMINATED_COLUMN":      "Zero-Dominated Column",
    "STATISTICAL_OUTLIERS":       "Outlier Detection",
    "FIXED_LENGTH_VIOLATION":     "Fixed Length Violation",
    "EMPTY_DATASET":              "Empty Dataset",
    "DUPLICATE_ROWS":             "Duplicate Rows",
    "SUSPICIOUSLY_FEW_ROWS":      "Suspiciously Few Rows",
    "ALL_IDENTICAL_COLUMN_VALUES":"All Identical Values",
    "PRE_EPOCH_DATES":            "Pre Epoch Date Check",
}

_SEVERITY_RISK: Dict[str, int] = {
    "CRITICAL": 85, "HIGH": 65, "MEDIUM": 40, "LOW": 20,
}

_CHECK_TYPE_CONTEXT: Dict[str, str] = {
    "UNPARSEABLE_DATES":          "values cannot be parsed as valid dates — the format may be mixed, corrupted, or non-standard",
    "FUTURE_DATES":               "values contain dates beyond today, which is invalid for this column",
    "ANCIENT_DATES":              "values contain dates implausibly far in the past, likely defaulted or corrupted",
    "STALE_DATA":                 "the most recent timestamp in this column is old — no recent records exist",
    "TEMPORAL_GAPS":              "expected time intervals are missing — certain date/time ranges have no records at all",
    "DUPLICATE_TIMESTAMPS":       "multiple rows share identical timestamps where each timestamp should be unique",
    "SINGLE_DATE_DOMINANCE":      "one single date value appears in the vast majority of records, suggesting a default fill or broken ETL",
    "WEEKEND_BUSINESS_DATES":     "dates fall on weekends or holidays where only business-day records are expected",
    "ALL_NULLS":                  "every value in this column is NULL — no usable data exists",
    "HIGH_NULL_RATE":             "a large share of values are NULL, indicating systematic missing data at the source or pipeline level",
    "CONSTANT_COLUMN":            "every row holds exactly the same value — the column carries no information or was incorrectly filled",
    "NEAR_CONSTANT_COLUMN":       "nearly all rows share one value with rare variation, suggesting a defaulting or encoding fault",
    "HIGH_DUPLICATE_RATE":        "a very high proportion of records are duplicates, violating expected row uniqueness",
    "SUSPICIOUS_WHITESPACE":      "values contain leading, trailing, or embedded whitespace that will break joins and exact-match lookups",
    "MIXED_CASE_INCONSISTENCY":   "the same logical value appears in multiple casing formats, causing grouping failures",
    "INVALID_EMAIL_FORMAT":       "values do not conform to a valid email address structure",
    "UNEXPECTED_NEGATIVE_VALUES": "negative numbers appear where only zero or positive values are expected",
    "ZERO_DOMINATED_COLUMN":      "the vast majority of values are zero, suggesting an uninitialized default or failed load",
    "STATISTICAL_OUTLIERS":       "values are statistically extreme — more than 3 standard deviations from the column mean",
    "FIXED_LENGTH_VIOLATION":     "values do not match the required fixed length for this identifier or code column",
    "EMPTY_DATASET":              "the entire dataset has zero rows — the data load likely failed or the source was empty",
    "DUPLICATE_ROWS":             "exact duplicate rows exist across the full dataset",
    "SUSPICIOUSLY_FEW_ROWS":      "the row count is far below what is expected",
    "ALL_IDENTICAL_COLUMN_VALUES":"every row holds the same value in this column with no variation",
    "PRE_EPOCH_DATES":            "values contain dates before the Unix epoch (1970-01-01)",
}

_FALLBACK_CONDITIONS: Dict[str, str] = {
    "ALL_NULLS":                  "NOT NULL",
    "HIGH_NULL_RATE":             "NOT NULL",
    "UNEXPECTED_NEGATIVE_VALUES": "{col} >= 0",
    "ZERO_DOMINATED_COLUMN":      "{col} != 0",
    "FUTURE_DATES":               "{col} <= CURRENT_DATE",
    "ANCIENT_DATES":              "{col} >= '1900-01-01'",
    "PRE_EPOCH_DATES":            "{col} >= '1970-01-01'",
    "STALE_DATA":                 "{col} >= CURRENT_DATE - INTERVAL 90 DAY",
    "INVALID_EMAIL_FORMAT":       'REGEX_MATCH({col}, "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$")',
    "HIGH_DUPLICATE_RATE":        "DISTINCT_COUNT = TOTAL_COUNT",
    "DUPLICATE_ROWS":             "DISTINCT_COUNT = TOTAL_COUNT",
    "DUPLICATE_TIMESTAMPS":       "DISTINCT_COUNT = TOTAL_COUNT",
    "STATISTICAL_OUTLIERS":       "{col} BETWEEN MEAN - 3*STDDEV AND MEAN + 3*STDDEV",
    "SUSPICIOUS_WHITESPACE":      "TRIM({col}) = {col}",
    "MIXED_CASE_INCONSISTENCY":   "LOWER({col}) = {col}",
}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _latest_run(db: Session, dataset_id: int) -> Optional[ProfilingRun]:
    return (db.query(ProfilingRun)
        .filter(ProfilingRun.dataset_id == dataset_id, ProfilingRun.status == "COMPLETED")
        .order_by(desc(ProfilingRun.timestamp)).first())

def _label(check_type: str) -> str:
    return _CHECK_TYPE_LABELS.get(check_type, check_type.replace("_", " ").title())

def _ctx(check_type: str) -> str:
    return _CHECK_TYPE_CONTEXT.get(check_type, "a data quality issue was detected")

def _risk(severity: str) -> int:
    return _SEVERITY_RISK.get((severity or "").upper(), 20)

def _impact(violation_count: int, total_rows: int) -> float:
    if not total_rows: return 0.0
    return round(min((violation_count / total_rows) * 100, 100), 1)

def _time_ago(dt: Optional[datetime]) -> str:
    if not dt: return "Unknown"
    now = datetime.now(timezone.utc)
    if dt.tzinfo is None: dt = dt.replace(tzinfo=timezone.utc)
    diff = int((now - dt).total_seconds())
    if diff < 60:    return f"{diff}s ago"
    if diff < 3600:  return f"{diff // 60}m ago"
    if diff < 86400: return f"{diff // 3600}h ago"
    return f"{diff // 86400}d ago"

def _next_rule_code(db: Session, dataset_id: int) -> str:
    rows = db.query(DQRule.rule_code).filter(DQRule.dataset_id == dataset_id).all()
    mx = 0
    for (code,) in rows:
        try:
            num = int((code or "").split("-")[-1])
            mx  = max(mx, num)
        except Exception:
            pass
    return f"RULE-{mx + 1:03d}"

def _format_check(check: QualityCheck, idx: int, total_rows: int,
                  run_timestamp: Optional[datetime]) -> Dict[str, Any]:
    sev_raw     = (check.severity or "LOW").upper()
    remediation: List[str] = []
    if check.llm_remediation:
        try:
            remediation = json.loads(check.llm_remediation)
        except Exception:
            remediation = [check.llm_remediation]

    return {
        "id":               f"ANOM-{idx:03d}",
        "db_id":            check.id,
        "type":             _label(check.check_type or ""),
        "check_type":       check.check_type,
        "column":           check.column_name or "dataset-level",
        "description":      check.description or "",
        "severity":         sev_raw.capitalize(),
        "status":           (check.status or "open").capitalize(),
        "detectedAt":       _time_ago(run_timestamp),
        "affectedRows":     check.violation_count or 0,
        "impactScore":      _impact(check.violation_count or 0, total_rows),
        "riskScore":        _risk(sev_raw),
        "llmAnalysed":      bool(check.llm_root_cause),
        "rootCause":        check.llm_root_cause,
        "remediation":      remediation,
        "resolvedByRuleId": check.resolved_by_rule_id,
    }


# ─── Public: list anomalies ───────────────────────────────────────────────────

def get_anomalies(db: Session, dataset_id: int) -> Dict[str, Any]:
    ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not ds:
        raise ValueError("Dataset not found")

    run = _latest_run(db, dataset_id)
    if not run:
        return {
            "status":   "NO_DATA",
            "message":  "Run profiling first to generate anomaly data.",
            "anomalies": [],
            "summary":  {"total": 0, "critical": 0, "avgImpact": 0.0, "open": 0},
        }

    checks = (db.query(QualityCheck)
        .filter(QualityCheck.profiling_run_id == run.id)
        .order_by(QualityCheck.id.asc()).all())

    if not checks:
        return {
            "status":    "success", "runId": run.id,
            "anomalies": [],
            "summary":   {"total": 0, "critical": 0, "avgImpact": 0.0, "open": 0},
        }

    # ── Carry forward resolved/investigating status from previous runs ──────
    # When a new profiling run happens, new QualityCheck rows are created with
    # status="open". We look up the most recent check for the same
    # check_type+column_name from older runs and inherit its status and
    # resolved_by_rule_id — BUT only if the linked rule still exists.
    # If the rule was deleted (resolved_by_rule_id is now NULL), we reset to open.
    prev_run = (
        db.query(ProfilingRun)
        .filter(
            ProfilingRun.dataset_id == dataset_id,
            ProfilingRun.status == "COMPLETED",
            ProfilingRun.id != run.id,
        )
        .order_by(ProfilingRun.id.desc())
        .first()
    )
    if prev_run:
        prev_checks = db.query(QualityCheck).filter(
            QualityCheck.profiling_run_id == prev_run.id
        ).all()
        # Build lookup: (check_type, column_name) → prev check
        prev_lookup = {
            (c.check_type, c.column_name): c
            for c in prev_checks
        }
        for check in checks:
            key = (check.check_type, check.column_name)
            prev = prev_lookup.get(key)
            if not prev:
                continue
            # Only inherit non-open statuses
            if (prev.status or "open").lower() in ("resolved", "investigating"):
                # Check: if resolved by a rule, verify that rule still exists
                if prev.status.lower() == "resolved" and prev.resolved_by_rule_id:
                    rule_exists = db.query(DQRule).filter(
                        DQRule.id == prev.resolved_by_rule_id
                    ).first()
                    if rule_exists:
                        check.status = prev.status
                        check.resolved_by_rule_id = prev.resolved_by_rule_id
                        check.llm_root_cause = check.llm_root_cause or prev.llm_root_cause
                        check.llm_remediation = check.llm_remediation or prev.llm_remediation
                    # else: rule deleted → stay open (don't inherit)
                elif prev.status.lower() == "investigating":
                    check.status = prev.status
        db.commit()

    total_rows = run.rows_processed or 1
    formatted  = [_format_check(c, i + 1, total_rows, run.timestamp) for i, c in enumerate(checks)]
    total      = len(formatted)
    critical   = sum(1 for a in formatted if a["severity"].upper() == "CRITICAL")
    avg_impact = round(sum(a["impactScore"] for a in formatted) / total, 1) if total else 0.0
    open_count = sum(1 for a in formatted if a["status"].lower() == "open")

    return {
        "status":    "success", "runId": run.id,
        "anomalies": formatted,
        "summary":   {"total": total, "critical": critical, "avgImpact": avg_impact, "open": open_count},
    }


# ─── Public: LLM analysis (root cause + ordered remediation) ─────────────────

def analyse_anomaly(db: Session, dataset_id: int, check_id: int, force: bool = False) -> Dict[str, Any]:
    check = db.query(QualityCheck).filter(QualityCheck.id == check_id).first()
    if not check:
        raise ValueError("Anomaly not found")

    # Return cache only if it's a real LLM result (not a previous failure message)
    _bad_cache_phrases = ("LLM analysis unavailable", "could not connect", "could not be parsed")
    cached_is_bad = not check.llm_root_cause or any(
        p in (check.llm_root_cause or "") for p in _bad_cache_phrases
    )

    if check.llm_root_cause and not force and not cached_is_bad:
        remediation: List[str] = []
        try:
            remediation = json.loads(check.llm_remediation or "[]")
        except Exception:
            pass
        return {"rootCause": check.llm_root_cause, "remediation": remediation, "cached": True}

    # Clear bad cached values before retrying
    if cached_is_bad:
        check.llm_root_cause = None
        check.llm_remediation = None
        db.commit()

    col       = check.column_name or "the dataset"
    chk_label = _label(check.check_type or "")
    chk_ctx   = _ctx(check.check_type or "")
    sev       = (check.severity or "").upper()
    n_rows    = check.violation_count or 0
    desc      = (check.description or "").strip()

    prompt = f"""You are a senior data quality engineer performing a root cause investigation.

A data quality check has failed. Analyse ONLY this specific failure.

=== FAILURE DETAILS ===
Check type   : {check.check_type} — {chk_label}
Failure mode : {chk_ctx}
Column       : {col}
Severity     : {sev}
Violations   : {n_rows} rows affected
System note  : {desc}

=== YOUR TASK ===

1. ROOT CAUSE (3-4 sentences):
   - Explain specifically why column "{col}" triggered a {chk_label} failure
   - Reference what "{chk_ctx}" implies about the upstream data pipeline or source
   - Name the most likely cause: ETL logic bug, schema change, source system issue, manual entry error, data load truncation, defaulting, etc.
   - Be specific to this column and failure type

2. REMEDIATION (exactly 4 steps, ORDERED from most likely to solve the root cause to least likely):
   - Step 1: The most direct and likely fix — addresses the core root cause immediately
   - Step 2: A secondary fix — cleans up downstream effects or prevents recurrence
   - Step 3: A validation/monitoring step — catches if the issue reappears
   - Step 4: A longer-term structural fix — pipeline, schema, or process change
   - Each step must be specific and actionable — no generic advice
   - Include actual SQL, code snippets, or pipeline steps where appropriate

Respond ONLY with valid JSON, no markdown fences:
{{"root_cause": "...", "remediation": ["most likely fix", "secondary fix", "validation step", "structural fix"]}}"""

    raw = _llm(prompt, max_tokens=900)

    if raw:
        try:
            raw    = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            parsed = json.loads(raw)
            root_cause  = str(parsed.get("root_cause", "")).strip()
            remediation = parsed.get("remediation", [])
            if not isinstance(remediation, list):
                remediation = [str(remediation)]
            remediation = [str(r).strip() for r in remediation[:4]]
        except Exception as e:
            print(f"[anomalies] JSON parse error: {e}")
            root_cause  = f"LLM response could not be parsed: {str(e)}"
            remediation = []
    else:
        root_cause  = "LLM analysis unavailable — could not connect to Azure AI Foundry endpoint."
        remediation = []

    # If LLM failed or returned no remediation, generate rule-based fallback suggestions
    # (but only as a last resort, clearly labelled)
    if not remediation:
        remediation = [
            f"Inspect column '{col}' for values where {chk_ctx}.",
            f"Trace the ETL step that last populated '{col}' and review its transformation logic.",
            f"Add a DQ rule to enforce that '{col}' does not exhibit {chk_label.lower()} going forward.",
            f"Backfill or correct the {n_rows} affected rows to resolve the existing violations.",
        ]

    check.llm_root_cause  = root_cause
    check.llm_remediation = json.dumps(remediation)
    db.commit()

    return {"rootCause": root_cause, "remediation": remediation, "cached": False}


# ─── Public: Fix Anomaly with optional selected remediation ──────────────────

def fix_anomaly(
    db: Session,
    dataset_id: int,
    check_id: int,
    selected_remediation: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate a DQ rule and mark the anomaly resolved.

    If selected_remediation is provided (the user picked a specific suggestion),
    it's included in the LLM prompt so the rule directly implements that fix.
    Falls back to deterministic conditions if LLM unavailable.
    """
    check = db.query(QualityCheck).filter(QualityCheck.id == check_id).first()
    if not check:
        raise ValueError("Anomaly not found")

    ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not ds:
        raise ValueError("Dataset not found")

    col       = check.column_name or "dataset_level"
    chk_label = _label(check.check_type or "")
    chk_ctx   = _ctx(check.check_type or "")
    desc      = (check.description or "").strip()
    sev       = (check.severity or "Medium").capitalize()

    # Build the fix prompt — incorporate selected remediation if provided
    remediation_context = ""
    if selected_remediation and selected_remediation.strip():
        remediation_context = f"""
=== SELECTED REMEDIATION APPROACH ===
The user selected this specific fix to implement as a DQ rule:
"{selected_remediation.strip()}"

Your rule condition should directly implement or enforce this approach on column "{col}".
"""

    prompt = f"""You are a data quality engineer. Generate a single DQ validation rule to detect and prevent this anomaly.

=== ANOMALY ===
Check type   : {check.check_type} — {chk_label}
Failure mode : {chk_ctx}
Column       : {col}
Severity     : {sev}
Description  : {desc}
{remediation_context}
=== RULE REQUIREMENTS ===
- Target column "{col}" specifically
- The condition must directly catch: {chk_ctx}
- Use valid DSL syntax. Examples:
    NOT NULL
    {col} > 0
    {col} >= 0
    {col} <= CURRENT_DATE
    {col} >= '1900-01-01'
    LENGTH({col}) = 10
    DISTINCT_COUNT = TOTAL_COUNT
    REGEX_MATCH({col}, "^pattern$")
    TRIM({col}) = {col}
- Rule type: Completeness, Validity, Uniqueness, Consistency, or Accuracy

Respond ONLY with valid JSON, no markdown:
{{
  "name": "short name describing the check on {col}",
  "type": "Completeness|Validity|Uniqueness|Consistency|Accuracy",
  "column": "{col}",
  "condition": "exact DSL condition",
  "severity": "{sev}"
}}"""

    raw = _llm(prompt, max_tokens=400)

    if raw:
        try:
            raw       = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            rule_data = json.loads(raw)
        except Exception:
            rule_data = None
    else:
        rule_data = None

    if not rule_data:
        fallback_cond = _FALLBACK_CONDITIONS.get(check.check_type or "", "NOT NULL")
        fallback_cond = fallback_cond.replace("{col}", col)
        rule_data = {
            "name":      f"{col} — {chk_label} Guard",
            "type":      "Validity",
            "column":    col,
            "condition": fallback_cond,
            "severity":  sev,
        }

    # Save rule as Active
    rule_code = _next_rule_code(db, dataset_id)
    new_rule  = DQRule(
        dataset_id=dataset_id,
        rule_code=rule_code,
        name=str(rule_data.get("name", f"{col} fix rule")),
        type=str(rule_data.get("type", "Validity")),
        column=str(rule_data.get("column", col)),
        condition=str(rule_data.get("condition", "NOT NULL")),
        severity=str(rule_data.get("severity", "Medium")),
        status="Active",
        input_mode="dsl",
    )
    db.add(new_rule)
    db.add(DQRuleChangeLog(
        dataset_id=dataset_id,
        rule_code=rule_code,
        rule_name=new_rule.name,
        version="v1.0",
        changed_by="AI Anomaly Fix",
        change_type="Created",
        performance_delta="N/A",
    ))

    check.status = "resolved"
    db.flush()
    check.resolved_by_rule_id = new_rule.id
    db.commit()
    db.refresh(new_rule)

    return {
        "status":   "success",
        "ruleCode": rule_code,
        "ruleId":   new_rule.id,
        "ruleName": new_rule.name,
        "message":  f"Rule '{new_rule.name}' created and activated. Anomaly marked as Resolved.",
    }


# ─── Public: update anomaly status ────────────────────────────────────────────

def update_anomaly_status(db: Session, dataset_id: int, check_id: int,
                          new_status: str) -> Dict[str, Any]:
    allowed = {"open", "investigating", "resolved"}
    if new_status.lower() not in allowed:
        raise ValueError(f"Status must be one of: {allowed}")
    check = db.query(QualityCheck).filter(QualityCheck.id == check_id).first()
    if not check:
        raise ValueError("Anomaly not found")
    check.status = new_status.lower()
    db.commit()
    return {"status": "success", "newStatus": check.status}