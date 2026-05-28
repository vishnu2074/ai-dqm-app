"""
AI DQM Health Metrics Router v4
=================================
Corrected against ACTUAL DB schema from app/models/__init__.py

Key column corrections vs v3:
  datasets:           display_name  (NOT name)
  profiling_runs:     timestamp, duration_ms, status="COMPLETED"  (no created_at/updated_at/summary)
  drift_records:      profiling_run_id, drift_score, drift_type  (no severity, no run_id)
  column_profiles:    profiling_run_id  (NOT run_id)
  dq_rules:           status="Active"  (no is_active, no enabled, no source)
  dq_rule_run_results: pass_rate, violation_count  (no status/result field)
  quality_snapshots:  score, snap_date  (no overall_score, no health_score, no created_at)
  lineage_edges:      source, target (strings)  (no source_dataset_id/target_dataset_id)
  governance_policies: status="Draft"|"Active" (no is_active, no enabled, no source)
  temporal_checks:    status="open"|"resolved", severity, llm_root_cause  (no is_anomaly)
"""

import os
import re
import sqlite3
from datetime import datetime
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/api/health-metrics", tags=["health-metrics"])


# ── DB helpers ────────────────────────────────────────────────────────────────

def _query(sql: str, params: dict = None) -> List[Dict[str, Any]]:
    """Execute SQL via SQLAlchemy engine (preferred) or sqlite3 fallback."""
    if params is None:
        params = {}
    try:
        from app.database import engine
        from sqlalchemy import text as _text
        with engine.connect() as conn:
            result = conn.execute(_text(sql), params)
            cols = list(result.keys())
            return [dict(zip(cols, row)) for row in result.fetchall()]
    except Exception:
        pass
    # sqlite3 fallback
    try:
        from app.database import engine
        db_url = str(engine.url)
        db_file = db_url.replace("sqlite:////", "/").replace("sqlite:///", "")
    except Exception:
        db_file = os.getenv("AIDQM_DB_PATH", "")
    if not db_file or not os.path.exists(db_file):
        return []
    try:
        sqlite_sql = re.sub(r":([a-zA-Z_][a-zA-Z0-9_]*)", "?", sql)
        sqlite_params = tuple(params.values())
        con = sqlite3.connect(db_file, timeout=10)
        con.row_factory = sqlite3.Row
        rows = [dict(r) for r in con.execute(sqlite_sql, sqlite_params).fetchall()]
        con.close()
        return rows
    except Exception as e:
        print(f"[health-metrics] query error: {e} | {sql[:80]}")
        return []


def _scalar(sql: str, params: dict = None, default=0):
    rows = _query(sql, params or {})
    if not rows:
        return default
    return list(rows[0].values())[0] if rows[0] else default


def _safe(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


def _status(value, good=80, warn=60):
    if value is None:
        return "neutral"
    try:
        v = float(value)
    except Exception:
        return "neutral"
    if v >= good:
        return "healthy"
    if v >= warn:
        return "warning"
    return "critical"


def _pct(value, good=85, warn=65):
    return _status(value, good, warn)


# ── Debug endpoint ────────────────────────────────────────────────────────────

@router.get("/debug")
def debug_info():
    db_url, db_file, tables, row_counts = "", "", [], {}
    try:
        from app.database import engine
        db_url = str(engine.url)
        db_file = db_url.replace("sqlite:////", "/").replace("sqlite:///", "")
    except Exception as e:
        db_url = f"error: {e}"
    if db_file and os.path.exists(db_file):
        try:
            con = sqlite3.connect(db_file, timeout=5)
            tables = [r[0] for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()]
            for t in tables:
                try:
                    row_counts[t] = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                except Exception:
                    row_counts[t] = "error"
            con.close()
        except Exception as e:
            tables = [f"error: {e}"]
    return {
        "sqlalchemy_url": db_url,
        "db_file": db_file,
        "db_exists": os.path.exists(db_file) if db_file else False,
        "tables": tables,
        "row_counts": row_counts,
        "cwd": os.getcwd(),
    }


# ── Schema introspection endpoint (new — for diagnosing future column issues) ─

@router.get("/schema-check")
def schema_check():
    """Returns actual columns for every table the health metrics router queries."""
    tables_to_check = [
        "datasets", "profiling_runs", "column_profiles", "drift_records",
        "dq_rules", "dq_rule_runs", "dq_rule_run_results", "quality_snapshots",
        "lineage_edges", "knowledge_graph_edges", "temporal_checks",
        "governance_policies", "governance_audit_log", "governance_dismissed_suggestions",
        "notification_inbox", "governance_notifications", "profiling_baselines",
    ]
    result = {}
    try:
        from app.database import engine
        db_url = str(engine.url)
        db_file = db_url.replace("sqlite:////", "/").replace("sqlite:///", "")
        if db_file and os.path.exists(db_file):
            con = sqlite3.connect(db_file, timeout=5)
            for t in tables_to_check:
                try:
                    cols = con.execute(f"PRAGMA table_info({t})").fetchall()
                    result[t] = [c[1] for c in cols]
                except Exception as e:
                    result[t] = f"error: {e}"
            con.close()
    except Exception as e:
        return {"error": str(e)}
    return result


# ── Datasets list ─────────────────────────────────────────────────────────────

@router.get("/datasets-list")
def get_datasets_list():
    # FIXED: column is display_name, not name
    rows = _query("SELECT id, display_name FROM datasets ORDER BY id LIMIT 200")
    return [
        {"id": r["id"], "display_name": r.get("display_name") or r.get("physical_name") or f"Dataset {r['id']}"}
        for r in rows
    ]


# ── Tab builders ──────────────────────────────────────────────────────────────

def _tab_global_llm() -> Dict:
    """
    Derived from profiling_runs (actual columns: timestamp, duration_ms, status=COMPLETED).
    No created_at/updated_at — use timestamp for timing and duration_ms for latency.
    """
    # FIXED: use timestamp column; status is uppercase "COMPLETED"
    runs = _query("SELECT id, status, duration_ms, timestamp, dataset_id FROM profiling_runs ORDER BY timestamp DESC LIMIT 200")
    total_runs = len(runs)

    completed = [r for r in runs if str(r.get("status", "")).upper() == "COMPLETED"]
    failed    = [r for r in runs if str(r.get("status", "")).upper() == "FAILED"]

    # Hallucination proxy: no summary field exists → use completion rate as quality signal
    hallucination_rate = round(len(failed) / total_runs * 100, 1) if total_runs else 0.0

    # Latency: use duration_ms directly (it IS the run duration)
    latencies = [r["duration_ms"] for r in completed if r.get("duration_ms") and r["duration_ms"] > 0]
    avg_latency = round(sum(latencies) / len(latencies), 0) if latencies else 0

    # Relevance: profiling runs that produced column_profiles
    # FIXED: FK is profiling_run_id not run_id
    runs_with_profiles = _safe(lambda: int(_scalar(
        "SELECT COUNT(DISTINCT profiling_run_id) FROM column_profiles"
    )), 0)
    relevance = round(runs_with_profiles / len(completed) * 100, 1) if completed else 0.0
    relevance = min(100.0, relevance)

    consistency = round(max(0.0, 100.0 - hallucination_rate * 1.5), 1)

    return {
        "tab": "Global AI / LLM",
        "metrics": [
            {
                "id": "hallucination_rate",
                "label": "Run Failure Rate",
                "value": hallucination_rate,
                "unit": "%",
                "status": _status(100 - hallucination_rate, 95, 80),
                "formula": "failed_runs / total_runs × 100",
                "details": {"failed": len(failed), "total": total_runs},
            },
            {
                "id": "avg_llm_latency_ms",
                "label": "Avg Run Duration",
                "value": int(avg_latency),
                "unit": "ms",
                "status": "healthy" if avg_latency < 30000 else "warning" if avg_latency < 120000 else "critical",
                "formula": "mean(duration_ms) across completed profiling runs",
                "details": {"samples": len(latencies)},
            },
            {
                "id": "response_relevance",
                "label": "Profile Coverage",
                "value": relevance,
                "unit": "%",
                "status": _pct(relevance),
                "formula": "runs_with_column_profiles / completed_runs × 100",
                "details": {"runs_with_profiles": runs_with_profiles, "completed": len(completed)},
            },
            {
                "id": "response_consistency",
                "label": "Run Consistency",
                "value": consistency,
                "unit": "%",
                "status": _pct(consistency),
                "formula": "100 - (failure_rate × 1.5)",
                "details": {},
            },
        ],
        "explainability": {
            "overview": "Global AI/LLM metrics are derived from profiling run outcomes — completion rates, duration, and profile coverage.",
            "improvement": "Ensure all profiling runs complete successfully. Check data source credentials and LLM endpoint on Render.",
            "low_success_rate": "High failure rate means profiling jobs are crashing. Check Render logs for error_message in profiling_runs.",
        },
    }


def _tab_profiling() -> Dict:
    # FIXED: timestamp (not created_at), duration_ms (not updated_at-created_at), status uppercase
    runs = _query("SELECT id, status, duration_ms, timestamp, dataset_id, rows_processed FROM profiling_runs ORDER BY timestamp DESC LIMIT 200")
    total = len(runs)
    completed = [r for r in runs if str(r.get("status", "")).upper() == "COMPLETED"]
    failed    = [r for r in runs if str(r.get("status", "")).upper() == "FAILED"]
    n_completed = len(completed)

    success_rate = round(n_completed / total * 100, 1) if total else 0.0

    # Metadata grounding: runs with rows_processed > 0
    grounded = [r for r in completed if (r.get("rows_processed") or 0) > 0]
    grounding = round(len(grounded) / n_completed * 100, 1) if n_completed else 0.0

    # Drift coverage: runs that generated drift_records (FIXED: FK is profiling_run_id)
    runs_with_drift = _safe(lambda: int(_scalar(
        "SELECT COUNT(DISTINCT profiling_run_id) FROM drift_records WHERE profiling_run_id IS NOT NULL"
    )), 0)
    drift_cov = round(runs_with_drift / n_completed * 100, 1) if n_completed else 0.0

    # Average runtime from duration_ms
    durations = [r["duration_ms"] / 1000.0 for r in completed if r.get("duration_ms") and r["duration_ms"] > 0]
    avg_runtime = round(sum(durations) / len(durations), 1) if durations else 0.0

    return {
        "tab": "Profiling AI",
        "metrics": [
            {
                "id": "profiling_success_rate",
                "label": "Profiling Success Rate",
                "value": success_rate,
                "unit": "%",
                "status": _pct(success_rate),
                "formula": "COMPLETED_runs / total_runs × 100",
                "details": {"completed": n_completed, "total": total, "failed": len(failed)},
            },
            {
                "id": "metadata_grounding_score",
                "label": "Rows Processed Rate",
                "value": grounding,
                "unit": "%",
                "status": _pct(grounding),
                "formula": "completed_runs_with_rows_processed > 0 / completed_runs × 100",
                "details": {"with_rows": len(grounded), "completed": n_completed},
            },
            {
                "id": "drift_detection_accuracy",
                "label": "Drift Detection Coverage",
                "value": drift_cov,
                "unit": "%",
                "status": _pct(drift_cov),
                "formula": "runs_producing_drift_records / completed_runs × 100",
                "details": {"runs_with_drift": runs_with_drift, "completed": n_completed},
            },
            {
                "id": "avg_profiling_runtime_s",
                "label": "Avg Profiling Runtime",
                "value": avg_runtime,
                "unit": "s",
                "status": "healthy" if avg_runtime < 120 else "warning" if avg_runtime < 600 else "critical",
                "formula": "mean(duration_ms / 1000) across completed runs",
                "details": {"samples": len(durations)},
            },
        ],
        "explainability": {
            "overview": "Profiling AI metrics measure how reliably datasets are profiled and drift is detected.",
            "improvement": "Improve success rate by checking data source credentials in the main app.",
            "low_success_rate": "Failed runs typically mean the data source is unreachable or credentials have expired.",
        },
    }


def _tab_dq_scores() -> Dict:
    # FIXED: quality_snapshots has only 'score' and 'snap_date' (no overall_score, no created_at)
    snapshots = _query("SELECT id, dataset_id, score, snap_date FROM quality_snapshots ORDER BY snap_date DESC LIMIT 100")

    health_scores = []
    for s in snapshots:
        v = s.get("score")
        if v is not None:
            try:
                health_scores.append(float(v))
            except Exception:
                pass

    avg_health = round(sum(health_scores) / len(health_scores), 1) if health_scores else 0.0
    accuracy   = round(sum(1 for s in health_scores if 0 <= s <= 100) / len(health_scores) * 100, 1) if health_scores else 100.0

    # FIXED: dq_rule_run_results has pass_rate and violation_count (no status/result field)
    rule_scores = _query("SELECT id, pass_rate, violation_count FROM dq_rule_run_results LIMIT 200")
    passed     = [r for r in rule_scores if (r.get("pass_rate") or 0) >= 1.0]
    compliance = round(len(passed) / len(rule_scores) * 100, 1) if rule_scores else 0.0

    # Degradation velocity: slope over last 5 snapshots
    recent = sorted([s for s in snapshots if s.get("score") is not None],
                    key=lambda x: str(x.get("snap_date", "")))[-5:]
    if len(recent) >= 2:
        vals     = [float(r["score"]) for r in recent]
        velocity = round(vals[-1] - vals[0], 1)
    else:
        velocity = 0.0

    return {
        "tab": "DQ Scores",
        "metrics": [
            {
                "id": "health_score_accuracy",
                "label": "Health Score Accuracy",
                "value": accuracy,
                "unit": "%",
                "status": _pct(accuracy),
                "formula": "quality_snapshots_in_valid_range(0-100) / total_snapshots × 100",
                "details": {"valid": len(health_scores), "total": len(snapshots)},
            },
            {
                "id": "rule_compliance_accuracy",
                "label": "Rule Pass Rate (pass_rate=1.0)",
                "value": compliance,
                "unit": "%",
                "status": _pct(compliance) if rule_scores else "neutral",
                "formula": "rule_results_with_pass_rate=1.0 / total_rule_results × 100",
                "details": {"passed": len(passed), "total": len(rule_scores)},
            },
            {
                "id": "avg_health_score",
                "label": "Avg Dataset Health Score",
                "value": avg_health,
                "unit": "%",
                "status": _pct(avg_health),
                "formula": "mean(score) across quality_snapshots",
                "details": {"snapshots": len(health_scores)},
            },
            {
                "id": "health_degradation_velocity",
                "label": "Health Degradation Velocity",
                "value": velocity,
                "unit": "pts",
                "status": "healthy" if velocity >= -5 else "warning" if velocity >= -15 else "critical",
                "formula": "snapshot[last].score - snapshot[first].score over last 5 (negative = degrading)",
                "details": {"window": len(recent)},
            },
        ],
        "explainability": {
            "overview": "DQ Scores measure dataset health via quality snapshots and rule execution results.",
            "improvement": "Run DQ rule evaluation jobs to populate dq_rule_run_results. Run profiling to add daily quality snapshots.",
            "low_success_rate": "Zero rule results means DQ rules haven't been executed yet. Trigger a rule evaluation run from the DQ Engine tab.",
        },
    }


def _tab_dq_rules() -> Dict:
    # FIXED: status is "Active" (capitalised), no is_active/enabled/source columns
    rules  = _query("SELECT id, status, type, name, rule_code FROM dq_rules LIMIT 500")
    total  = len(rules)
    active = [r for r in rules if str(r.get("status", "")).lower() == "active"]

    rule_runs = _query("SELECT id, dataset_id, status, started_at, finished_at FROM dq_rule_runs LIMIT 500")
    results   = _query("SELECT id, run_id, rule_code, pass_rate, violation_count FROM dq_rule_run_results LIMIT 500")

    executed_run_ids = set(r.get("run_id") for r in results if r.get("run_id"))
    # Execution rate: rule_runs that have results
    exec_rate = round(len(executed_run_ids) / len(rule_runs) * 100, 1) if rule_runs else 0.0

    # AI-suggested rules: no source column — use rule_code prefix as heuristic
    ai_rules = [r for r in rules if str(r.get("rule_code", "")).startswith("AI_") or
                str(r.get("name", "")).lower().startswith("ai ")]
    accepted = [r for r in ai_rules if str(r.get("status", "")).lower() == "active"]
    accept_rate = round(len(accepted) / len(ai_rules) * 100, 1) if ai_rules else 0.0

    # Pass rate distribution from results
    avg_pass = round(
        sum(float(r.get("pass_rate") or 0) for r in results) / len(results) * 100, 1
    ) if results else 0.0

    return {
        "tab": "DQ Rules",
        "metrics": [
            {
                "id": "rule_execution_success_rate",
                "label": "Rule Execution Rate",
                "value": exec_rate,
                "unit": "%",
                "status": _pct(exec_rate) if rule_runs else "neutral",
                "formula": "rule_runs_with_results / total_rule_runs × 100",
                "details": {"runs_with_results": len(executed_run_ids), "total_runs": len(rule_runs), "total_rules": total},
            },
            {
                "id": "rule_recommendation_acceptance_rate",
                "label": "Active Rule Ratio",
                "value": round(len(active) / total * 100, 1) if total else 0.0,
                "unit": "%",
                "status": _pct(round(len(active) / total * 100, 1)) if total else "neutral",
                "formula": "active_rules / total_rules × 100",
                "details": {"active": len(active), "total": total},
            },
            {
                "id": "avg_pass_rate",
                "label": "Avg Rule Pass Rate",
                "value": avg_pass,
                "unit": "%",
                "status": _pct(avg_pass) if results else "neutral",
                "formula": "mean(pass_rate) × 100 across dq_rule_run_results",
                "details": {"result_rows": len(results)},
            },
        ],
        "explainability": {
            "overview": "DQ Rules metrics track rule quality — how many are active and how well they perform when executed.",
            "improvement": "Trigger rule evaluation runs from the main app. Review rules and ensure they have Active status.",
            "low_success_rate": "dq_rule_runs is empty — no rule evaluation jobs have been run yet. Start one from the DQ Engine tab.",
        },
    }


def _tab_monitoring() -> Dict:
    # FIXED: status uppercase, no severity on drift_records
    runs  = _query("SELECT id, status, duration_ms, timestamp FROM profiling_runs ORDER BY timestamp DESC LIMIT 200")
    # FIXED: drift_records has drift_score and drift_type (no severity column)
    drift = _query("SELECT id, drift_score, drift_type, profiling_run_id FROM drift_records LIMIT 500")
    total = len(runs)
    completed = [r for r in runs if str(r.get("status", "")).upper() == "COMPLETED"]

    uptime = round(len(completed) / total * 100, 1) if total else 0.0

    # Drift precision: high drift_score (>=0.5) as signal of meaningful drift
    significant = [d for d in drift if float(d.get("drift_score") or 0) >= 0.5]
    precision   = round(len(significant) / len(drift) * 100, 1) if drift else 0.0

    # Volatility: std dev of quality scores
    snaps = _query("SELECT score FROM quality_snapshots ORDER BY snap_date DESC LIMIT 20")
    vals  = []
    for s in snaps:
        v = s.get("score")
        if v is not None:
            try: vals.append(float(v))
            except Exception: pass
    if len(vals) >= 2:
        mean_v    = sum(vals) / len(vals)
        std_v     = (sum((x - mean_v) ** 2 for x in vals) / len(vals)) ** 0.5
        volatility = round(std_v, 1)
    else:
        volatility = 0.0

    return {
        "tab": "Monitoring & Trends",
        "metrics": [
            {
                "id": "monitoring_uptime",
                "label": "Profiling Run Success Rate",
                "value": uptime,
                "unit": "%",
                "status": _pct(uptime),
                "formula": "COMPLETED_runs / total_runs × 100",
                "details": {"completed": len(completed), "total": total},
            },
            {
                "id": "drift_detection_precision",
                "label": "High-Drift Alert Rate",
                "value": precision,
                "unit": "%",
                "status": _pct(precision) if drift else "neutral",
                "formula": "drift_records_with_drift_score >= 0.5 / total_drift_records × 100",
                "details": {"significant": len(significant), "total": len(drift)},
            },
            {
                "id": "forecast_error_rate",
                "label": "Health Score Volatility",
                "value": volatility,
                "unit": "pts std",
                "status": "healthy" if volatility < 5 else "warning" if volatility < 15 else "critical",
                "formula": "stddev(quality_snapshots.score) — lower = more stable",
                "details": {"samples": len(vals)},
            },
        ],
        "explainability": {
            "overview": "Monitoring metrics reflect profiling consistency, drift signal quality, and health score stability over time.",
            "improvement": "Schedule profiling runs at regular intervals. High volatility means data quality is changing rapidly.",
            "high_drift": f"{len(drift)} drift records exist — review them in the main app's Monitoring tab.",
        },
    }


def _tab_anomalies() -> Dict:
    # FIXED: temporal_checks (aliased QualityCheck) has status="open"|"resolved", severity, llm_root_cause
    # No is_anomaly field; "open" = unresolved issue; severity column exists
    checks  = _query("SELECT id, status, severity, description, llm_root_cause, llm_remediation FROM temporal_checks LIMIT 500")
    total   = len(checks)
    # open = anomaly/issue not yet resolved
    flagged = [c for c in checks if str(c.get("status", "")).lower() == "open"]
    resolved = [c for c in checks if str(c.get("status", "")).lower() == "resolved"]

    precision = round(len(flagged) / total * 100, 1) if total else 0.0

    # RCA quality: open checks with llm_root_cause
    rca_attempts = [c for c in flagged if c.get("llm_root_cause")]
    rca_bad      = [c for c in rca_attempts if len(str(c.get("llm_root_cause") or "")) < 20]
    rca_hall     = round(len(rca_bad) / len(rca_attempts) * 100, 1) if rca_attempts else 0.0

    results   = _query("SELECT id, pass_rate FROM dq_rule_run_results LIMIT 500")
    n_results = len(results)
    recall    = round(len(flagged) / n_results * 100, 1) if n_results else (100.0 if not flagged else 0.0)
    recall    = min(100.0, recall)

    return {
        "tab": "Anomalies AI",
        "metrics": [
            {
                "id": "anomaly_precision",
                "label": "Open Temporal Check Rate",
                "value": precision,
                "unit": "%",
                "status": _status(100 - precision, 70, 50) if total else "neutral",
                "formula": "open_temporal_checks / total_temporal_checks × 100",
                "details": {"open": len(flagged), "resolved": len(resolved), "total": total},
            },
            {
                "id": "anomaly_recall",
                "label": "Detection Recall",
                "value": recall,
                "unit": "%",
                "status": _pct(recall) if n_results else "neutral",
                "formula": "open_checks / total_rule_results × 100",
                "details": {"flagged": len(flagged), "rule_results": n_results},
            },
            {
                "id": "rca_hallucination_rate",
                "label": "RCA Quality Rate",
                "value": round(100 - rca_hall, 1),
                "unit": "%",
                "status": _pct(100 - rca_hall) if rca_attempts else "neutral",
                "formula": "checks_with_meaningful_llm_root_cause / total_rca_attempts × 100",
                "details": {"good_rca": len(rca_attempts) - len(rca_bad), "attempts": len(rca_attempts)},
            },
            {
                "id": "auto_fix_success_rate",
                "label": "Check Resolution Rate",
                "value": round(len(resolved) / total * 100, 1) if total else 0.0,
                "unit": "%",
                "status": _pct(round(len(resolved) / total * 100, 1)) if total else "neutral",
                "formula": "resolved_temporal_checks / total_temporal_checks × 100",
                "details": {"resolved": len(resolved), "total": total},
            },
        ],
        "explainability": {
            "overview": "Anomaly metrics use temporal_checks — AI-generated validation checks that flag statistical anomalies in dataset columns over time.",
            "improvement": "Investigate open temporal checks in the main app. High open rates indicate volatile data pipelines.",
            "many_critical": "Many open temporal checks suggest systematic data quality issues. Review by dataset in the Monitoring tab.",
        },
    }


def _tab_lineage() -> Dict:
    # FIXED: lineage_edges has source/target as strings (dataset names), not source_dataset_id/target_dataset_id
    edges       = _query("SELECT id, source, target FROM lineage_edges LIMIT 500")
    total_edges = len(edges)
    datasets_n  = int(_scalar("SELECT COUNT(*) FROM datasets"))

    # Map dataset names to IDs
    ds_rows       = _query("SELECT id, display_name, physical_name FROM datasets")
    name_set      = set()
    for d in ds_rows:
        if d.get("display_name"): name_set.add(str(d["display_name"]).lower())
        if d.get("physical_name"): name_set.add(str(d["physical_name"]).lower())

    # Check which edges reference known datasets
    mapped_sources = set(str(e["source"]).lower() for e in edges if e.get("source"))
    mapped_targets = set(str(e["target"]).lower() for e in edges if e.get("target"))
    mapped = mapped_sources | mapped_targets
    known  = mapped & name_set
    coverage = round(len(known) / datasets_n * 100, 1) if datasets_n else 0.0

    return {
        "tab": "Data Lineage & Impact",
        "metrics": [
            {
                "id": "lineage_coverage",
                "label": "Lineage Coverage",
                "value": coverage,
                "unit": "%",
                "status": _pct(coverage) if datasets_n else "neutral",
                "formula": "known_dataset_names_in_lineage_edges / total_datasets × 100",
                "details": {"known": len(known), "total_datasets": datasets_n, "edges": total_edges},
            },
            {
                "id": "broken_edge_count",
                "label": "Unknown Source/Target Names",
                "value": len(mapped - name_set),
                "unit": "",
                "status": "healthy" if not (mapped - name_set) else "warning",
                "formula": "lineage edge endpoints not matching any known dataset name",
                "details": {"unknown": len(mapped - name_set), "total_edges": total_edges},
            },
            {
                "id": "missed_dependency_rate",
                "label": "Datasets Without Lineage",
                "value": round((datasets_n - len(known)) / datasets_n * 100, 1) if datasets_n else 0.0,
                "unit": "%",
                "status": _pct(coverage) if datasets_n else "neutral",
                "formula": "(total_datasets - datasets_with_lineage) / total_datasets × 100",
                "details": {"without_lineage": datasets_n - len(known), "total": datasets_n},
            },
        ],
        "explainability": {
            "overview": "Lineage metrics track how completely upstream/downstream dataset relationships are mapped in lineage_edges.",
            "improvement": "Create lineage edges from the Lineage tab in the main app. Run profiling on all 10 datasets first.",
            "low_coverage": f"{total_edges} lineage edges found. lineage_edges stores source/target as dataset name strings.",
        },
    }


def _tab_kg() -> Dict:
    # KnowledgeGraphEdge columns: confidence, relationship_type, source_column, target_column, invalidated
    edges     = _query("SELECT id, confidence, relationship_type, source_column, target_column, invalidated FROM knowledge_graph_edges LIMIT 1000")
    total     = len(edges)
    high_conf = [e for e in edges if float(e.get("confidence") or 0) >= 0.7]
    null_conf = [e for e in edges if e.get("confidence") is None]
    invalid   = [e for e in edges if e.get("invalidated")]

    precision  = round(len(high_conf) / total * 100, 1) if total else 0.0
    hall_rate  = round(len(null_conf) / total * 100, 1) if total else 0.0

    # Column mapping coverage
    total_cols  = int(_scalar("SELECT COUNT(DISTINCT column_name) FROM column_profiles"))
    col_edges   = set(e.get("source_column") for e in edges if e.get("source_column")) | \
                  set(e.get("target_column") for e in edges if e.get("target_column"))
    mapping_acc = round(len(col_edges) / total_cols * 100, 1) if total_cols else 0.0
    mapping_acc = min(100.0, mapping_acc)

    return {
        "tab": "Knowledge Graph AI",
        "metrics": [
            {
                "id": "kg_relationship_precision",
                "label": "Relationship Precision",
                "value": precision,
                "unit": "%",
                "status": _pct(precision) if total else "neutral",
                "formula": "knowledge_graph_edges_with_confidence >= 0.7 / total_edges × 100",
                "details": {"high_confidence": len(high_conf), "total": total},
            },
            {
                "id": "kg_column_mapping_accuracy",
                "label": "Column Relationship Coverage",
                "value": mapping_acc,
                "unit": "%",
                "status": _pct(mapping_acc) if total else "neutral",
                "formula": "distinct_columns_in_kg_edges / distinct_columns_in_profiles × 100",
                "details": {"col_edges": len(col_edges), "distinct_cols": total_cols},
            },
            {
                "id": "kg_hallucinated_relationship_rate",
                "label": "Unscored Relationship Rate",
                "value": hall_rate,
                "unit": "%",
                "status": _status(100 - hall_rate, 90, 75) if total else "neutral",
                "formula": "edges_with_no_confidence_score / total_edges × 100",
                "details": {"unscored": len(null_conf), "total": total},
            },
        ],
        "explainability": {
            "overview": "Knowledge Graph metrics reflect AI-generated semantic relationships between dataset columns via knowledge_graph_edges.",
            "improvement": "Run KG construction from the main app after profiling all datasets. Currently 0 edges — the graph hasn't been built yet.",
            "low_coverage": "0 KG edges is expected if KG construction hasn't been triggered. Use the Knowledge Graph tab in the main AI DQM app.",
        },
    }


def _tab_assistant() -> Dict:
    # NotificationInbox: type, category, message, dataset_id, severity
    inbox     = _query("SELECT id, type, category, message, dataset_id, severity FROM notification_inbox LIMIT 500")
    # GovernanceNotification: title, description, enabled, channel
    gov_notif = _query("SELECT id, title, description, enabled, channel FROM governance_notifications LIMIT 200")
    total     = len(inbox) + len(gov_notif)

    # Routing accuracy: inbox notifications with a type set
    tagged   = [n for n in inbox if n.get("type") and n["type"] != "ALERT"]  # non-default type = properly routed
    routing  = round(len(tagged) / len(inbox) * 100, 1) if inbox else 0.0

    # Quality: notifications with meaningful message (>15 chars)
    good_msg  = [n for n in inbox if n.get("message") and len(str(n["message"])) >= 15]
    hall_rate = round((len(inbox) - len(good_msg)) / len(inbox) * 100, 1) if inbox else 0.0

    # Grounding: notifications referencing a dataset
    grounded  = [n for n in inbox if n.get("dataset_id")]
    grounding = round(len(grounded) / len(inbox) * 100, 1) if inbox else 0.0

    return {
        "tab": "DQ Assistant / AI Agent",
        "metrics": [
            {
                "id": "agent_routing_accuracy",
                "label": "Notification Type Distribution",
                "value": routing,
                "unit": "%",
                "status": _pct(routing) if inbox else "neutral",
                "formula": "notifications_with_non-default_type / total × 100",
                "details": {"typed": len(tagged), "total": len(inbox)},
            },
            {
                "id": "assistant_hallucination_rate",
                "label": "Notification Quality Rate",
                "value": round(100 - hall_rate, 1),
                "unit": "%",
                "status": _pct(100 - hall_rate) if inbox else "neutral",
                "formula": "notifications_with_message_length >= 15 / total × 100",
                "details": {"good": len(good_msg), "total": len(inbox)},
            },
            {
                "id": "action_agent_success_rate",
                "label": "Governance Action Rate",
                "value": round(len(gov_notif) / max(total, 1) * 100, 1),
                "unit": "%",
                "status": _pct(round(len(gov_notif) / max(total, 1) * 100, 1)) if total else "neutral",
                "formula": "governance_notifications / total_notifications × 100",
                "details": {"governance": len(gov_notif), "total": total},
            },
            {
                "id": "retrieval_grounding_score",
                "label": "Context Grounding Score",
                "value": grounding,
                "unit": "%",
                "status": _pct(grounding) if inbox else "neutral",
                "formula": "inbox_notifications_with_dataset_id / total × 100",
                "details": {"grounded": len(grounded), "total": len(inbox)},
            },
        ],
        "explainability": {
            "overview": "Assistant metrics are derived from notification_inbox and governance_notifications — the AI agent's output trail.",
            "improvement": "More profiling runs and DQ rule evaluations generate richer notifications, improving all scores here.",
            "low_satisfaction": "Low grounding means notifications aren't linked to specific datasets. Ensure dataset IDs are passed in all agent calls.",
        },
    }


def _tab_governance() -> Dict:
    # FIXED: GovernancePolicy has status field (no is_active/enabled/source)
    # Active status check: status = "Active" (capitalised)
    policies = _query("SELECT id, name, status, policy_type FROM governance_policies LIMIT 200")
    total    = len(policies)
    active   = [p for p in policies if str(p.get("status", "")).lower() == "active"]

    # Classification: column_profiles with non-null sensitivity data
    classified = int(_scalar(
        "SELECT COUNT(*) FROM column_profiles WHERE data_type IS NOT NULL AND data_type != ''"
    ))
    total_cols = int(_scalar("SELECT COUNT(*) FROM column_profiles"))
    class_acc  = round(classified / total_cols * 100, 1) if total_cols else 0.0

    # Audit log
    audit_n  = int(_scalar("SELECT COUNT(*) FROM governance_audit_log"))
    action_n = int(_scalar("SELECT COUNT(*) FROM profiling_runs")) + int(_scalar("SELECT COUNT(*) FROM dq_rules"))
    audit_c  = round(audit_n / action_n * 100, 1) if action_n else 0.0
    audit_c  = min(100.0, audit_c)

    # Policy adoption = active / total
    adopt = round(len(active) / total * 100, 1) if total else 0.0

    return {
        "tab": "Governance & Settings",
        "metrics": [
            {
                "id": "policy_adoption_rate",
                "label": "Policy Adoption Rate",
                "value": adopt,
                "unit": "%",
                "status": _pct(adopt) if total else "neutral",
                "formula": "active_governance_policies / total_policies × 100",
                "details": {"active": len(active), "total": total},
            },
            {
                "id": "classification_accuracy",
                "label": "Column Type Classification",
                "value": class_acc,
                "unit": "%",
                "status": _pct(class_acc) if total_cols else "neutral",
                "formula": "column_profiles_with_data_type / total_column_profiles × 100",
                "details": {"classified": classified, "total": total_cols},
            },
            {
                "id": "audit_log_completeness",
                "label": "Audit Log Completeness",
                "value": audit_c,
                "unit": "%",
                "status": _pct(audit_c),
                "formula": "governance_audit_log entries / (profiling_runs + dq_rules) × 100",
                "details": {"audit_entries": audit_n, "actions": action_n},
            },
        ],
        "explainability": {
            "overview": "Governance metrics track policy adoption, data type classification, and audit trail completeness.",
            "improvement": "Classify column types in the Governance tab. Review and activate governance policies.",
            "low_adoption": f"Only {total} governance policy/policies. Use the Governance tab to create and activate policies.",
        },
    }


def _tab_system() -> Dict:
    # FIXED: status uppercase, use duration_ms, use timestamp
    runs      = _query("SELECT id, status, duration_ms, timestamp FROM profiling_runs ORDER BY timestamp DESC LIMIT 200")
    completed = [r for r in runs if str(r.get("status", "")).upper() == "COMPLETED"]
    failed    = [r for r in runs if str(r.get("status", "")).upper() == "FAILED"]
    total     = len(runs)

    uptime = round(len(completed) / (len(completed) + len(failed)) * 100, 1) if (completed or failed) else 100.0

    # Throughput: runs per hour over time range
    if total >= 2:
        sorted_r = sorted([r for r in runs if r.get("timestamp")], key=lambda x: str(x["timestamp"]))
        try:
            t0 = datetime.fromisoformat(str(sorted_r[0]["timestamp"]).replace("Z", "+00:00").replace(" ", "T"))
            t1 = datetime.fromisoformat(str(sorted_r[-1]["timestamp"]).replace("Z", "+00:00").replace(" ", "T"))
            hours = max(1, (t1 - t0).total_seconds() / 3600)
            throughput = round(total / hours, 2)
        except Exception:
            throughput = 0.0
    else:
        throughput = 0.0

    # Avg runtime from duration_ms
    durations = [r["duration_ms"] for r in completed if r.get("duration_ms") and r["duration_ms"] > 0]
    avg_ms    = round(sum(durations) / len(durations)) if durations else 0

    return {
        "tab": "System / Platform",
        "metrics": [
            {
                "id": "system_uptime",
                "label": "System Uptime",
                "value": uptime,
                "unit": "%",
                "status": _pct(uptime),
                "formula": "COMPLETED_runs / (completed + failed) × 100",
                "details": {"completed": len(completed), "failed": len(failed), "total": total},
            },
            {
                "id": "api_throughput",
                "label": "Processing Throughput",
                "value": throughput,
                "unit": "runs/hr",
                "status": "healthy" if throughput > 0 else "neutral",
                "formula": "total_profiling_runs / elapsed_hours",
                "details": {"total_runs": total},
            },
            {
                "id": "avg_job_duration_ms",
                "label": "Avg Job Duration",
                "value": avg_ms,
                "unit": "ms",
                "status": "healthy" if avg_ms < 60000 else "warning" if avg_ms < 300000 else "critical",
                "formula": "mean(duration_ms) across completed runs",
                "details": {"samples": len(durations)},
            },
        ],
        "explainability": {
            "overview": "System metrics reflect platform reliability and processing throughput based on profiling run history.",
            "improvement": f"{total} profiling runs recorded. Monitor failed runs in Render logs.",
            "low_success_rate": "Failed runs appear in profiling_runs with status='FAILED'. Check data source credentials.",
        },
    }


def _tab_feedback() -> Dict:
    # FIXED: governance_dismissed_suggestions only has 'id' column
    dismissed  = int(_scalar("SELECT COUNT(*) FROM governance_dismissed_suggestions"))
    # FIXED: governance_policies.status (no is_active)
    active_pol = int(_scalar(
        "SELECT COUNT(*) FROM governance_policies WHERE LOWER(status) = 'active'"
    ))
    total_suggestions = dismissed + active_pol
    accept_rate = round(active_pol / total_suggestions * 100, 1) if total_suggestions else 0.0

    # Satisfaction proxy: governance_audit_log actions
    audit_actions = int(_scalar("SELECT COUNT(*) FROM governance_audit_log"))
    satisfaction  = round(min(100.0, audit_actions / 10 * 100), 1)

    return {
        "tab": "Human Feedback",
        "metrics": [
            {
                "id": "ai_acceptance_rate",
                "label": "Policy Acceptance Rate",
                "value": accept_rate,
                "unit": "%",
                "status": _pct(accept_rate) if total_suggestions else "neutral",
                "formula": "active_governance_policies / (active + dismissed) × 100",
                "details": {"accepted": active_pol, "dismissed": dismissed, "total": total_suggestions},
            },
            {
                "id": "analyst_satisfaction_score",
                "label": "Platform Engagement Score",
                "value": satisfaction,
                "unit": "/100",
                "status": _pct(satisfaction) if audit_actions else "neutral",
                "formula": "min(audit_log_actions / 10 × 100, 100) — proxy for analyst engagement",
                "details": {"audit_actions": audit_actions},
            },
        ],
        "explainability": {
            "overview": "Human Feedback uses governance policy acceptance and audit log activity as proxies for analyst engagement.",
            "improvement": "Encourage analysts to review and accept/dismiss AI policy suggestions. More audit log activity = higher engagement score.",
            "low_satisfaction": f"{audit_actions} audit log entries. More active use of Governance, DQ Rules, and Policy features increases this score.",
        },
    }


# ── Main metrics endpoint ─────────────────────────────────────────────────────

@router.get("/")
@router.get("")
def get_all_metrics(dataset_id: Optional[int] = Query(None)):
    """Compute and return all health metric tabs from live DB data."""
    db_path = "unknown"
    try:
        from app.database import engine
        db_path = str(engine.url).replace("sqlite:////", "/").replace("sqlite:///", "")
    except Exception:
        pass

    tabs = [
        _safe(_tab_global_llm,   {"tab": "Global AI / LLM",          "metrics": [], "explainability": {}}),
        _safe(_tab_profiling,    {"tab": "Profiling AI",               "metrics": [], "explainability": {}}),
        _safe(_tab_dq_scores,    {"tab": "DQ Scores",                  "metrics": [], "explainability": {}}),
        _safe(_tab_dq_rules,     {"tab": "DQ Rules",                   "metrics": [], "explainability": {}}),
        _safe(_tab_monitoring,   {"tab": "Monitoring & Trends",        "metrics": [], "explainability": {}}),
        _safe(_tab_anomalies,    {"tab": "Anomalies AI",               "metrics": [], "explainability": {}}),
        _safe(_tab_lineage,      {"tab": "Data Lineage & Impact",      "metrics": [], "explainability": {}}),
        _safe(_tab_kg,           {"tab": "Knowledge Graph AI",         "metrics": [], "explainability": {}}),
        _safe(_tab_assistant,    {"tab": "DQ Assistant / AI Agent",    "metrics": [], "explainability": {}}),
        _safe(_tab_governance,   {"tab": "Governance & Settings",      "metrics": [], "explainability": {}}),
        _safe(_tab_system,       {"tab": "System / Platform",          "metrics": [], "explainability": {}}),
        _safe(_tab_feedback,     {"tab": "Human Feedback",             "metrics": [], "explainability": {}}),
    ]

    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "dataset_id": dataset_id,
        "db_path": db_path,
        "tabs": tabs,
    }
