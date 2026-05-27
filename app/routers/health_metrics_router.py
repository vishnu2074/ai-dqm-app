"""
AI DQM Health Metrics Router v3
=================================
Corrected for actual DB schema at /tmp/ai-dqm/ai_dqm.db

Real tables (from debug endpoint):
  data_sources, datasets, profiling_runs, column_profiles,
  profiling_baselines, drift_records, dq_rules, dq_rule_history,
  dq_rule_runs, dq_rule_run_results, knowledge_graph_edges,
  lineage_edges, governance_policies, governance_audit_log,
  temporal_checks, quality_snapshots, notification_inbox,
  governance_notifications, governance_users, governance_system_config,
  governance_dismissed_suggestions, dataset_versions, schema_history,
  global_context, notification_preferences

Place at: app/routers/health_metrics_router.py
Register in app/main.py (already done).
"""

import os
import re
import sqlite3
from datetime import datetime
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/api/health-metrics", tags=["health-metrics"])


# ── DB connection — uses the same SQLAlchemy engine as the main app ───────────

def _query(sql: str, params: dict = None) -> List[Dict[str, Any]]:
    """Execute SQL via SQLAlchemy engine (preferred) or sqlite3 fallback."""
    if params is None:
        params = {}
    # 1. SQLAlchemy (same engine, same connection pool as main app)
    try:
        from app.database import engine
        from sqlalchemy import text as _text
        with engine.connect() as conn:
            result = conn.execute(_text(sql), params)
            cols = list(result.keys())
            return [dict(zip(cols, row)) for row in result.fetchall()]
    except Exception:
        pass
    # 2. sqlite3 fallback — convert :name → ? for sqlite3
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
    """Call fn(), return default on any exception."""
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


# ── Datasets list ─────────────────────────────────────────────────────────────

@router.get("/datasets-list")
def get_datasets_list():
    rows = _query("SELECT id, name FROM datasets ORDER BY id LIMIT 200")
    return [{"id": r["id"], "display_name": r.get("name") or f"Dataset {r['id']}"} for r in rows]


# ── Tab builders ──────────────────────────────────────────────────────────────

def _tab_global_llm() -> Dict:
    """
    No llm_interactions table exists. Derive from profiling_runs AI fields
    and temporal_checks (which stores AI-generated check results).
    """
    runs = _query("SELECT * FROM profiling_runs ORDER BY created_at DESC LIMIT 200")
    total_runs = len(runs)

    # Runs with an AI-generated summary / description field
    ai_runs = [r for r in runs if r.get("summary") or r.get("ai_summary") or r.get("description")]
    
    # Hallucination proxy: runs that completed but have no summary despite having column_profiles
    completed = [r for r in runs if r.get("status") == "completed"]
    no_summary = [r for r in completed if not r.get("summary") and not r.get("ai_summary")]
    hallucination_rate = round(len(no_summary) / len(completed) * 100, 1) if completed else 0.0

    # Latency: time between created_at and updated_at on profiling runs (proxy for LLM call duration)
    latencies = []
    for r in completed:
        try:
            s = datetime.fromisoformat(str(r.get("created_at","")).replace("Z","+00:00").replace(" ","T"))
            e = datetime.fromisoformat(str(r.get("updated_at","")).replace("Z","+00:00").replace(" ","T"))
            ms = (e - s).total_seconds() * 1000
            if 0 < ms < 600000:
                latencies.append(ms)
        except Exception:
            pass
    avg_latency = round(sum(latencies) / len(latencies), 0) if latencies else 0

    # Relevance: profiling runs that produced column_profiles
    runs_with_profiles = _safe(lambda: int(_scalar(
        "SELECT COUNT(DISTINCT run_id) FROM column_profiles"
    )), 0)
    relevance = round(runs_with_profiles / len(completed) * 100, 1) if completed else 100.0

    consistency = round(max(0.0, 100.0 - hallucination_rate * 1.5), 1)

    return {
        "tab": "Global AI / LLM",
        "metrics": [
            {
                "id": "hallucination_rate",
                "label": "Hallucination Rate",
                "value": hallucination_rate,
                "unit": "%",
                "status": _status(100 - hallucination_rate, 95, 80),
                "formula": "completed_runs_without_ai_summary / completed_runs × 100",
                "details": {"no_summary": len(no_summary), "completed": len(completed)},
            },
            {
                "id": "avg_llm_latency_ms",
                "label": "Avg LLM Latency",
                "value": int(avg_latency),
                "unit": "ms",
                "status": "healthy" if avg_latency < 30000 else "warning" if avg_latency < 120000 else "critical",
                "formula": "mean(updated_at - created_at) across completed profiling runs",
                "details": {"samples": len(latencies)},
            },
            {
                "id": "response_relevance",
                "label": "Response Relevance",
                "value": relevance,
                "unit": "%",
                "status": _pct(relevance),
                "formula": "runs_with_column_profiles / completed_runs × 100",
                "details": {"runs_with_profiles": runs_with_profiles, "completed": len(completed)},
            },
            {
                "id": "response_consistency",
                "label": "Response Consistency",
                "value": consistency,
                "unit": "%",
                "status": _pct(consistency),
                "formula": "100 - (hallucination_rate × 1.5)",
                "details": {},
            },
        ],
        "explainability": {
            "overview": "Global AI/LLM metrics are derived from profiling run outcomes — completion rates, AI summary generation, and profile coverage.",
            "improvement": "Ensure all profiling runs complete successfully and generate AI summaries. Check LLM endpoint availability in Azure AI Foundry.",
            "low_success_rate": "High hallucination rate means completed runs aren't generating AI summaries. Check the LLM key configuration on Render.",
        },
    }


def _tab_profiling() -> Dict:
    runs = _query("SELECT * FROM profiling_runs ORDER BY created_at DESC LIMIT 200")
    total = len(runs)
    completed = [r for r in runs if r.get("status") == "completed"]
    failed = [r for r in runs if r.get("status") == "failed"]
    n_completed = len(completed)

    success_rate = round(n_completed / total * 100, 1) if total else 0.0

    # Metadata grounding: completed runs with AI summary
    grounded = [r for r in completed if r.get("summary") or r.get("ai_summary") or r.get("description")]
    grounding = round(len(grounded) / n_completed * 100, 1) if n_completed else 0.0

    # Drift coverage: runs that generated drift_records
    runs_with_drift = _safe(lambda: int(_scalar(
        "SELECT COUNT(DISTINCT run_id) FROM drift_records WHERE run_id IS NOT NULL"
    )), 0)
    drift_cov = round(runs_with_drift / n_completed * 100, 1) if n_completed else 0.0

    # Average runtime
    durations = []
    for r in completed:
        try:
            s = datetime.fromisoformat(str(r.get("created_at","")).replace("Z","+00:00").replace(" ","T"))
            e = datetime.fromisoformat(str(r.get("updated_at","")).replace("Z","+00:00").replace(" ","T"))
            sec = (e - s).total_seconds()
            if 0 < sec < 3600:
                durations.append(sec)
        except Exception:
            pass
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
                "formula": "completed_runs / total_runs × 100",
                "details": {"completed": n_completed, "total": total, "failed": len(failed)},
            },
            {
                "id": "metadata_grounding_score",
                "label": "Metadata Grounding Score",
                "value": grounding,
                "unit": "%",
                "status": _pct(grounding),
                "formula": "runs_with_ai_summary / completed_runs × 100",
                "details": {"grounded": len(grounded), "completed": n_completed},
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
                "formula": "mean(updated_at - created_at) in seconds across completed runs",
                "details": {"samples": len(durations)},
            },
        ],
        "explainability": {
            "overview": "Profiling AI metrics measure how reliably datasets are profiled, AI summaries generated, and drift detected.",
            "improvement": "Improve success rate by checking data source credentials. Improve grounding by ensuring the LLM endpoint is configured.",
            "low_success_rate": "Failed runs typically mean the data source is unreachable or credentials have expired.",
        },
    }


def _tab_dq_scores() -> Dict:
    snapshots = _query("SELECT * FROM quality_snapshots ORDER BY created_at DESC LIMIT 100")
    baselines = _query("SELECT * FROM profiling_baselines ORDER BY created_at DESC LIMIT 100")

    # Health scores from quality_snapshots
    health_scores = []
    for s in snapshots:
        v = s.get("overall_score") or s.get("health_score") or s.get("score")
        if v is not None:
            try:
                health_scores.append(float(v))
            except Exception:
                pass

    avg_health = round(sum(health_scores) / len(health_scores), 1) if health_scores else 0.0
    accuracy = round(sum(1 for s in health_scores if 0 <= s <= 100) / len(health_scores) * 100, 1) if health_scores else 100.0

    # Rule compliance from dq_rule_run_results
    rule_scores = _query("SELECT * FROM dq_rule_run_results LIMIT 200")
    passed = [r for r in rule_scores if r.get("status") == "passed" or r.get("result") == "pass"]
    compliance = round(len(passed) / len(rule_scores) * 100, 1) if rule_scores else 0.0

    # Degradation velocity: slope of quality_snapshots over time
    recent = sorted([s for s in snapshots if s.get("overall_score") or s.get("score")],
                    key=lambda x: str(x.get("created_at", "")))[-5:]
    if len(recent) >= 2:
        vals = [float(r.get("overall_score") or r.get("score") or 0) for r in recent]
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
                "label": "Rule Pass Rate",
                "value": compliance,
                "unit": "%",
                "status": _pct(compliance) if rule_scores else "neutral",
                "formula": "passed_rule_results / total_rule_results × 100",
                "details": {"passed": len(passed), "total": len(rule_scores)},
            },
            {
                "id": "avg_health_score",
                "label": "Avg Dataset Health Score",
                "value": avg_health,
                "unit": "%",
                "status": _pct(avg_health),
                "formula": "mean(overall_score) across quality_snapshots",
                "details": {"snapshots": len(health_scores)},
            },
            {
                "id": "health_degradation_velocity",
                "label": "Health Degradation Velocity",
                "value": velocity,
                "unit": "pts",
                "status": "healthy" if velocity >= -5 else "warning" if velocity >= -15 else "critical",
                "formula": "quality_snapshot[last].score - quality_snapshot[first].score over last 5 (negative = degrading)",
                "details": {"window": len(recent)},
            },
        ],
        "explainability": {
            "overview": "DQ Scores measure dataset health via quality snapshots and rule execution results.",
            "improvement": "Run DQ rule evaluation jobs regularly to populate dq_rule_run_results and get accurate compliance rates.",
            "low_success_rate": "Zero rule results means DQ rules haven't been executed yet. Trigger a rule evaluation run from the main DQ app.",
        },
    }


def _tab_dq_rules() -> Dict:
    rules = _query("SELECT * FROM dq_rules LIMIT 500")
    total = len(rules)
    active = [r for r in rules if r.get("is_active") or r.get("status") == "active" or r.get("enabled")]

    # Execution: dq_rule_runs
    rule_runs = _query("SELECT * FROM dq_rule_runs LIMIT 500")
    results   = _query("SELECT * FROM dq_rule_run_results LIMIT 500")

    executed_rule_ids = set(r.get("rule_id") for r in rule_runs if r.get("rule_id"))
    active_ids = set(r.get("id") for r in active)
    exec_rate = round(len(executed_rule_ids & active_ids) / len(active_ids) * 100, 1) if active_ids else 0.0

    # AI-suggested rules: source field
    ai_rules = [r for r in rules if str(r.get("source","")).lower() in ("llm","ai","recommended","generated")]
    accepted  = [r for r in ai_rules if r.get("is_active") or r.get("enabled")]
    accept_rate = round(len(accepted) / len(ai_rules) * 100, 1) if ai_rules else 0.0

    # Hallucinated: AI rules that have never been run
    hall_rules = [r for r in ai_rules if r.get("id") not in executed_rule_ids]
    hall_rate  = round(len(hall_rules) / len(ai_rules) * 100, 1) if ai_rules else 0.0

    return {
        "tab": "DQ Rules",
        "metrics": [
            {
                "id": "rule_execution_success_rate",
                "label": "Rule Execution Success Rate",
                "value": exec_rate,
                "unit": "%",
                "status": _pct(exec_rate) if active_ids else "neutral",
                "formula": "active_rules_with_run_history / total_active_rules × 100",
                "details": {"executed": len(executed_rule_ids & active_ids), "active": len(active_ids), "total": total},
            },
            {
                "id": "rule_recommendation_acceptance_rate",
                "label": "AI Rule Acceptance Rate",
                "value": accept_rate,
                "unit": "%",
                "status": _pct(accept_rate) if ai_rules else "neutral",
                "formula": "active_ai_rules / total_ai_suggested_rules × 100",
                "details": {"accepted": len(accepted), "suggested": len(ai_rules)},
            },
            {
                "id": "hallucinated_rule_rate",
                "label": "Hallucinated Rule Rate",
                "value": hall_rate,
                "unit": "%",
                "status": _status(100 - hall_rate, 90, 75),
                "formula": "ai_rules_never_executed / total_ai_rules × 100",
                "details": {"never_run": len(hall_rules), "ai_rules": len(ai_rules)},
            },
        ],
        "explainability": {
            "overview": "DQ Rules metrics track rule quality — how many are active, AI-recommended, and actually being executed.",
            "improvement": "Trigger rule evaluation runs from the main app. Review AI-suggested rules and activate relevant ones.",
            "low_success_rate": "dq_rule_runs is empty — no rule evaluation jobs have been run yet. Start one from the DQ Engine tab.",
        },
    }


def _tab_monitoring() -> Dict:
    runs  = _query("SELECT * FROM profiling_runs ORDER BY created_at DESC LIMIT 200")
    drift = _query("SELECT * FROM drift_records LIMIT 500")
    total = len(runs)
    completed = [r for r in runs if r.get("status") == "completed"]

    uptime = round(len(completed) / total * 100, 1) if total else 0.0

    # Drift precision: records with meaningful severity
    significant = [d for d in drift if str(d.get("severity","")).upper() in ("HIGH","CRITICAL","MEDIUM")]
    precision = round(len(significant) / len(drift) * 100, 1) if drift else 0.0

    # Volatility: std dev of health scores from quality_snapshots
    snaps = _query("SELECT overall_score, score FROM quality_snapshots ORDER BY created_at DESC LIMIT 20")
    vals  = []
    for s in snaps:
        v = s.get("overall_score") or s.get("score")
        if v is not None:
            try: vals.append(float(v))
            except Exception: pass
    if len(vals) >= 2:
        mean_v = sum(vals) / len(vals)
        std_v  = (sum((x - mean_v) ** 2 for x in vals) / len(vals)) ** 0.5
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
                "formula": "completed_runs / total_runs × 100",
                "details": {"completed": len(completed), "total": total},
            },
            {
                "id": "drift_detection_precision",
                "label": "Drift Alert Precision",
                "value": precision,
                "unit": "%",
                "status": _pct(precision) if drift else "neutral",
                "formula": "MEDIUM/HIGH/CRITICAL drift_records / total_drift_records × 100",
                "details": {"significant": len(significant), "total": len(drift)},
            },
            {
                "id": "forecast_error_rate",
                "label": "Health Score Volatility",
                "value": volatility,
                "unit": "pts std",
                "status": "healthy" if volatility < 5 else "warning" if volatility < 15 else "critical",
                "formula": "stddev(quality_snapshots.overall_score) — lower = more stable",
                "details": {"samples": len(vals)},
            },
        ],
        "explainability": {
            "overview": "Monitoring metrics reflect profiling consistency, drift signal quality, and health score stability over time.",
            "improvement": "Schedule profiling runs at regular intervals. High volatility means data quality is changing rapidly.",
            "high_drift": "208 drift records exist — review them in the main app's Monitoring tab to identify which datasets are drifting.",
        },
    }


def _tab_anomalies() -> Dict:
    """
    No anomalies table. Use temporal_checks as the closest equivalent —
    these are AI-generated checks that flag data quality issues.
    """
    checks  = _query("SELECT * FROM temporal_checks LIMIT 500")
    total   = len(checks)
    flagged = [c for c in checks if c.get("status") in ("failed","error","anomaly") or c.get("is_anomaly")]
    passed  = [c for c in checks if c.get("status") in ("passed","ok")]

    precision = round(len(flagged) / total * 100, 1) if total else 0.0

    # RCA proxy: flagged checks with an explanation/reason field
    rca_attempts = [c for c in flagged if c.get("explanation") or c.get("reason") or c.get("details")]
    rca_bad      = [c for c in rca_attempts if len(str(c.get("explanation") or c.get("reason") or "")) < 20]
    rca_hall     = round(len(rca_bad) / len(rca_attempts) * 100, 1) if rca_attempts else 0.0

    # Use dq_rule_run_results for recall
    results   = _query("SELECT * FROM dq_rule_run_results LIMIT 500")
    n_results = len(results)
    recall    = round(len(flagged) / n_results * 100, 1) if n_results else (100.0 if not flagged else 0.0)
    recall    = min(100.0, recall)

    return {
        "tab": "Anomalies AI",
        "metrics": [
            {
                "id": "anomaly_precision",
                "label": "Temporal Check Anomaly Rate",
                "value": precision,
                "unit": "%",
                "status": _status(100 - precision, 70, 50) if total else "neutral",
                "formula": "failed_temporal_checks / total_temporal_checks × 100",
                "details": {"flagged": len(flagged), "total": total, "passed": len(passed)},
            },
            {
                "id": "anomaly_recall",
                "label": "Detection Recall",
                "value": recall,
                "unit": "%",
                "status": _pct(recall) if n_results else "neutral",
                "formula": "flagged_checks / total_rule_results × 100",
                "details": {"flagged": len(flagged), "rule_results": n_results},
            },
            {
                "id": "rca_hallucination_rate",
                "label": "RCA Quality Rate",
                "value": round(100 - rca_hall, 1),
                "unit": "%",
                "status": _pct(100 - rca_hall) if rca_attempts else "neutral",
                "formula": "checks_with_meaningful_explanation / total_rca_attempts × 100",
                "details": {"good_rca": len(rca_attempts) - len(rca_bad), "attempts": len(rca_attempts)},
            },
            {
                "id": "auto_fix_success_rate",
                "label": "Check Resolution Rate",
                "value": round(len(passed) / total * 100, 1) if total else 0.0,
                "unit": "%",
                "status": _pct(round(len(passed) / total * 100, 1)) if total else "neutral",
                "formula": "passed_temporal_checks / total_temporal_checks × 100",
                "details": {"passed": len(passed), "total": total},
            },
        ],
        "explainability": {
            "overview": "Anomaly metrics use temporal_checks — AI-generated validation checks that flag statistical anomalies in dataset columns over time.",
            "improvement": "Investigate failed temporal checks in the main app. High anomaly rates indicate volatile data pipelines.",
            "many_critical": "Many failed temporal checks suggest systematic data quality issues. Review by dataset in the Monitoring tab.",
        },
    }


def _tab_lineage() -> Dict:
    edges         = _query("SELECT * FROM lineage_edges LIMIT 500")
    total_edges   = len(edges)
    broken        = [e for e in edges if e.get("status") == "broken" or e.get("is_stale")]
    low_conf      = [e for e in edges if (e.get("confidence") or 1.0) < 0.5]
    datasets_n    = int(_scalar("SELECT COUNT(*) FROM datasets"))
    mapped_ids    = set(e.get("source_dataset_id") for e in edges) | set(e.get("target_dataset_id") for e in edges)
    mapped_ids.discard(None)
    coverage = round(len(mapped_ids) / datasets_n * 100, 1) if datasets_n else 0.0

    return {
        "tab": "Data Lineage & Impact",
        "metrics": [
            {
                "id": "lineage_coverage",
                "label": "Lineage Coverage",
                "value": coverage,
                "unit": "%",
                "status": _pct(coverage) if datasets_n else "neutral",
                "formula": "datasets_with_lineage_edges / total_datasets × 100",
                "details": {"mapped": len(mapped_ids), "total": datasets_n, "edges": total_edges},
            },
            {
                "id": "broken_edge_count",
                "label": "Broken Lineage Edges",
                "value": len(broken),
                "unit": "",
                "status": "healthy" if not broken else "warning" if len(broken) < 5 else "critical",
                "formula": "COUNT(lineage_edges WHERE status='broken' OR is_stale=true)",
                "details": {"broken": len(broken), "total": total_edges},
            },
            {
                "id": "missed_dependency_rate",
                "label": "Low-Confidence Dependency Rate",
                "value": round(len(low_conf) / total_edges * 100, 1) if total_edges else 0.0,
                "unit": "%",
                "status": _pct(100 - len(low_conf) / total_edges * 100) if total_edges else "neutral",
                "formula": "edges_with_confidence < 0.5 / total_edges × 100",
                "details": {"low_confidence": len(low_conf), "total": total_edges},
            },
        ],
        "explainability": {
            "overview": "Lineage metrics track how completely upstream/downstream dataset relationships are mapped.",
            "improvement": "Lineage edges are auto-detected during profiling. Run profiling on all 10 datasets to build the lineage graph.",
            "low_coverage": "0 lineage edges currently — profiling runs haven't generated lineage data yet. Check if the lineage engine is enabled.",
        },
    }


def _tab_kg() -> Dict:
    """knowledge_graph_edges table exists (0 rows). No kg_nodes table."""
    edges      = _query("SELECT * FROM knowledge_graph_edges LIMIT 1000")
    total      = len(edges)
    high_conf  = [e for e in edges if float(e.get("confidence") or e.get("weight") or 0) >= 0.7]
    null_conf  = [e for e in edges if e.get("confidence") is None and e.get("weight") is None]

    precision  = round(len(high_conf) / total * 100, 1) if total else 0.0
    hall_rate  = round(len(null_conf) / total * 100, 1) if total else 0.0

    # Column mapping: edges where edge_type involves column relationships
    col_edges  = [e for e in edges if str(e.get("edge_type","")).lower() in
                  ("similar_to","references","maps_to","column_similarity","related_to")]
    # Count distinct column references from column_profiles
    total_cols = int(_scalar("SELECT COUNT(DISTINCT column_name) FROM column_profiles"))
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
                "formula": "knowledge_graph_edges_with_confidence ≥ 0.7 / total_edges × 100",
                "details": {"high_confidence": len(high_conf), "total": total},
            },
            {
                "id": "kg_column_mapping_accuracy",
                "label": "Column Relationship Coverage",
                "value": mapping_acc,
                "unit": "%",
                "status": _pct(mapping_acc) if total else "neutral",
                "formula": "column_relationship_edges / distinct_columns_in_profiles × 100",
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
    """
    No llm_interactions table. Derive from notification_inbox (AI-generated
    notifications) and governance_notifications as proxy for agent activity.
    """
    inbox    = _query("SELECT * FROM notification_inbox LIMIT 500")
    gov_notif = _query("SELECT * FROM governance_notifications LIMIT 200")
    total    = len(inbox) + len(gov_notif)

    # Routing accuracy: notifications with a category/type tag (agent correctly categorised)
    tagged   = [n for n in inbox if n.get("type") or n.get("category") or n.get("notification_type")]
    routing  = round(len(tagged) / len(inbox) * 100, 1) if inbox else 0.0

    # Hallucination: notifications with empty or very short message
    bad_msg  = [n for n in inbox if not n.get("message") or len(str(n.get("message",""))) < 15]
    hall_rate = round(len(bad_msg) / len(inbox) * 100, 1) if inbox else 0.0

    # Grounding: notifications referencing a dataset_id or entity
    grounded = [n for n in inbox if n.get("dataset_id") or n.get("entity_id") or n.get("run_id")]
    grounding = round(len(grounded) / len(inbox) * 100, 1) if inbox else 0.0

    return {
        "tab": "DQ Assistant / AI Agent",
        "metrics": [
            {
                "id": "agent_routing_accuracy",
                "label": "Notification Routing Accuracy",
                "value": routing,
                "unit": "%",
                "status": _pct(routing) if inbox else "neutral",
                "formula": "notifications_with_type_tag / total_notifications × 100",
                "details": {"tagged": len(tagged), "total": len(inbox)},
            },
            {
                "id": "assistant_hallucination_rate",
                "label": "Notification Quality Rate",
                "value": round(100 - hall_rate, 1),
                "unit": "%",
                "status": _pct(100 - hall_rate) if inbox else "neutral",
                "formula": "notifications_with_meaningful_message / total × 100",
                "details": {"good": len(inbox) - len(bad_msg), "total": len(inbox)},
            },
            {
                "id": "action_agent_success_rate",
                "label": "Governance Action Rate",
                "value": round(len(gov_notif) / max(total, 1) * 100, 1),
                "unit": "%",
                "status": _pct(round(len(gov_notif) / max(total, 1) * 100, 1)) if total else "neutral",
                "formula": "governance_notifications / total_agent_notifications × 100",
                "details": {"governance": len(gov_notif), "total": total},
            },
            {
                "id": "retrieval_grounding_score",
                "label": "Context Grounding Score",
                "value": grounding,
                "unit": "%",
                "status": _pct(grounding) if inbox else "neutral",
                "formula": "notifications_referencing_dataset_or_entity / total × 100",
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
    policies = _query("SELECT * FROM governance_policies LIMIT 200")
    total    = len(policies)
    active   = [p for p in policies if p.get("is_active") or p.get("status") == "active" or p.get("enabled")]
    ai_pol   = [p for p in policies if str(p.get("source","")).lower() in ("llm","ai","suggested","generated")]
    accepted = [p for p in ai_pol if p.get("is_active") or p.get("enabled")]
    adopt    = round(len(accepted) / len(ai_pol) * 100, 1) if ai_pol else 0.0

    # Classification: column_profiles with a sensitivity_label or data_type classified
    classified = int(_scalar(
        "SELECT COUNT(*) FROM column_profiles WHERE sensitivity_label IS NOT NULL AND sensitivity_label != ''"
    ))
    total_cols = int(_scalar("SELECT COUNT(*) FROM column_profiles"))
    class_acc  = round(classified / total_cols * 100, 1) if total_cols else 0.0

    # Audit log
    audit_n = int(_scalar("SELECT COUNT(*) FROM governance_audit_log"))
    action_n = int(_scalar("SELECT COUNT(*) FROM profiling_runs")) + int(_scalar("SELECT COUNT(*) FROM dq_rules"))
    audit_c  = round(audit_n / action_n * 100, 1) if action_n else 0.0
    audit_c  = min(100.0, audit_c)

    return {
        "tab": "Governance & Settings",
        "metrics": [
            {
                "id": "policy_adoption_rate",
                "label": "Policy Adoption Rate",
                "value": adopt,
                "unit": "%",
                "status": _pct(adopt) if ai_pol else "neutral",
                "formula": "active_ai_policies / total_ai_suggested_policies × 100",
                "details": {"accepted": len(accepted), "suggested": len(ai_pol), "total": total},
            },
            {
                "id": "classification_accuracy",
                "label": "Column Sensitivity Classification",
                "value": class_acc,
                "unit": "%",
                "status": _pct(class_acc) if total_cols else "neutral",
                "formula": "column_profiles_with_sensitivity_label / total_column_profiles × 100",
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
            "overview": "Governance metrics track policy adoption, data sensitivity classification, and audit trail completeness.",
            "improvement": "Classify column sensitivity in the Governance tab. Review AI policy suggestions and activate relevant ones.",
            "low_adoption": "Only 1 governance policy exists. Use the Governance tab to generate and review AI policy suggestions.",
        },
    }


def _tab_system() -> Dict:
    runs      = _query("SELECT * FROM profiling_runs ORDER BY created_at DESC LIMIT 200")
    completed = [r for r in runs if r.get("status") == "completed"]
    failed    = [r for r in runs if r.get("status") == "failed"]
    total     = len(runs)

    uptime = round(len(completed) / (len(completed) + len(failed)) * 100, 1) if (completed or failed) else 100.0

    # Throughput: runs per hour
    if total >= 2:
        sorted_r = sorted([r for r in runs if r.get("created_at")], key=lambda x: str(x["created_at"]))
        try:
            t0 = datetime.fromisoformat(str(sorted_r[0]["created_at"]).replace("Z","+00:00").replace(" ","T"))
            t1 = datetime.fromisoformat(str(sorted_r[-1]["created_at"]).replace("Z","+00:00").replace(" ","T"))
            hours = max(1, (t1 - t0).total_seconds() / 3600)
            throughput = round(total / hours, 2)
        except Exception:
            throughput = 0.0
    else:
        throughput = 0.0

    # Avg runtime
    durations = []
    for r in completed:
        try:
            s = datetime.fromisoformat(str(r.get("created_at","")).replace("Z","+00:00").replace(" ","T"))
            e = datetime.fromisoformat(str(r.get("updated_at","")).replace("Z","+00:00").replace(" ","T"))
            ms = (e - s).total_seconds() * 1000
            if 0 < ms < 3600000:
                durations.append(ms)
        except Exception:
            pass
    avg_ms = round(sum(durations) / len(durations)) if durations else 0

    return {
        "tab": "System / Platform",
        "metrics": [
            {
                "id": "system_uptime",
                "label": "System Uptime",
                "value": uptime,
                "unit": "%",
                "status": _pct(uptime),
                "formula": "completed_runs / (completed + failed) × 100",
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
                "formula": "mean(updated_at - created_at) in ms across completed runs",
                "details": {"samples": len(durations)},
            },
        ],
        "explainability": {
            "overview": "System metrics reflect platform reliability and processing throughput based on profiling run history.",
            "improvement": "35 profiling runs recorded. Monitor failed runs in Render logs. High job duration means large datasets.",
            "low_success_rate": "Failed runs appear in the profiling_runs table with status='failed'. Check data source credentials.",
        },
    }


def _tab_feedback() -> Dict:
    # governance_dismissed_suggestions = analyst explicitly dismissed → inverse of acceptance
    dismissed = int(_scalar("SELECT COUNT(*) FROM governance_dismissed_suggestions"))
    total_pol  = int(_scalar("SELECT COUNT(*) FROM governance_policies"))
    # Accepted = total policies that are active
    active_pol = int(_scalar(
        "SELECT COUNT(*) FROM governance_policies WHERE is_active = 1 OR status = 'active'"
    ))
    total_suggestions = dismissed + active_pol
    accept_rate = round(active_pol / total_suggestions * 100, 1) if total_suggestions else 0.0

    # Satisfaction proxy: governance_audit_log actions (more actions = more engagement)
    audit_actions = int(_scalar("SELECT COUNT(*) FROM governance_audit_log"))
    # Scale: 0 actions = 0, 10+ actions = 100
    satisfaction = round(min(100.0, audit_actions / 10 * 100), 1)

    return {
        "tab": "Human Feedback",
        "metrics": [
            {
                "id": "ai_acceptance_rate",
                "label": "AI Suggestion Acceptance Rate",
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
            "low_satisfaction": "3 audit log entries recorded. More active use of Governance, DQ Rules, and Policy features increases this score.",
        },
    }


# ── Main metrics endpoint ─────────────────────────────────────────────────────

@router.get("/")
@router.get("")
def get_all_metrics(dataset_id: Optional[int] = Query(None)):
    """Compute and return all health metric tabs from live DB data."""
    # Determine DB path for response metadata
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