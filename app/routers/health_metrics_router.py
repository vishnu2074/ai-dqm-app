"""
AI DQM Health Metrics Router
==============================
Drop this file into: <main-ai-dqm-app>/app/routers/health_metrics_router.py

Then in <main-ai-dqm-app>/app/main.py add:
    try:
        from app.routers.health_metrics_router import router as health_metrics_router
        app.include_router(health_metrics_router)
    except Exception as e:
        print(f"Health metrics router load error: {e}")

Also add "http://localhost:5174" (and your dashboard's deployed URL) to CORS origins in main.py.

This router reads directly from the SQLite database that the main AI DQM app uses.
It does NOT call itself — it uses SQLAlchemy sessions (or raw sqlite3 as fallback).
"""

import os
import json
import sqlite3
from datetime import datetime
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, Query

router = APIRouter(prefix="/api/health-metrics", tags=["health-metrics"])


@router.get("/debug")
def debug_info():
    """Diagnostic — shows DB path resolution and table counts."""
    import os, sqlite3
    candidates_checked = []
    for p in _DB_CANDIDATES:
        resolved = os.path.abspath(p)
        candidates_checked.append({"path": resolved, "exists": os.path.exists(resolved)})
    env_path = os.getenv("AIDQM_DB_PATH", "")
    db_path = _get_db_path()
    tables = []
    row_counts = {}
    if db_path:
        try:
            con = sqlite3.connect(db_path, timeout=5)
            tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
            for t in tables:
                try:
                    row_counts[t] = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                except:
                    row_counts[t] = "error"
            con.close()
        except Exception as e:
            tables = [f"ERROR: {e}"]
    # Also try reading from SQLAlchemy engine if available
    sa_db_url = ""
    try:
        from app.database import engine
        sa_db_url = str(engine.url)
    except Exception as e:
        sa_db_url = f"unavailable: {e}"
    return {
        "db_found": db_path,
        "env_AIDQM_DB_PATH": env_path,
        "sqlalchemy_url": sa_db_url,
        "candidates_checked": candidates_checked,
        "tables": tables,
        "row_counts": row_counts,
        "cwd": os.getcwd(),
        "__file__": os.path.abspath(__file__),
    }

# ── Database path — same DB the main AI DQM app uses ──────────────────────────
# The main AI DQM app typically stores its SQLite DB in the app root or a data/ folder.
# Adjust this path if your app stores it elsewhere.
_DB_CANDIDATES = [
    # Your actual DB location: project_root/ai_dqm.db
    # Router lives at: app/routers/health_metrics_router.py => ../../ai_dqm.db
    os.path.join(os.path.dirname(__file__), "..", "..", "ai_dqm.db"),
    os.path.join(os.path.dirname(__file__), "..", "..", "aidqm.db"),
    os.path.join(os.path.dirname(__file__), "..", "ai_dqm.db"),
    # Absolute fallback — set AIDQM_DB_PATH env var on Render/Azure
    "/var/data/ai_dqm.db",
    "/opt/render/project/src/ai_dqm.db",
    "ai_dqm.db",
]

def _get_db_path() -> Optional[str]:
    # 1. Environment variable override (highest priority)
    env_path = os.getenv("AIDQM_DB_PATH", "")
    if env_path and os.path.exists(env_path):
        return env_path

    # 2. Try to extract path from SQLAlchemy engine URL
    try:
        from app.database import engine
        url = str(engine.url)
        # sqlite:////absolute/path/to/db  or  sqlite:///relative/path
        if url.startswith("sqlite"):
            db_file = url.replace("sqlite:///", "").replace("sqlite://", "")
            if db_file and os.path.exists(db_file):
                return db_file
            # try as absolute
            if db_file.startswith("/") and os.path.exists(db_file):
                return db_file
    except Exception:
        pass

    # 3. Walk candidate paths
    for p in _DB_CANDIDATES:
        resolved = os.path.abspath(p)
        if os.path.exists(resolved):
            return resolved

    return None


def _query(sql: str, params=None) -> List[Dict[str, Any]]:
    """Run a read-only query — tries SQLAlchemy engine first, falls back to sqlite3.
    params can be a dict (named :param style) or tuple (positional ? style).
    """
    if params is None:
        params = {}

    # --- Try SQLAlchemy engine (same connection the main app uses) ---
    try:
        from app.database import engine
        from sqlalchemy import text as _text
        with engine.connect() as conn:
            # SQLAlchemy needs dict params with named placeholders
            sa_params = params if isinstance(params, dict) else {}
            result = conn.execute(_text(sql), sa_params)
            cols = list(result.keys())
            return [dict(zip(cols, row)) for row in result.fetchall()]
    except Exception:
        pass  # fall through to sqlite3

    # --- Fallback: direct sqlite3 file access ---
    db_path = _get_db_path()
    if not db_path:
        print(f"[health-metrics] DB not found. Set AIDQM_DB_PATH env var.")
        return []
    try:
        con = sqlite3.connect(db_path, timeout=10)
        con.row_factory = sqlite3.Row
        # sqlite3 needs tuple params with ? placeholders
        sqlite_params = tuple(params.values()) if isinstance(params, dict) else (params if params else ())
        # Convert named placeholders to ? for sqlite3 fallback
        import re
        sqlite_sql = re.sub(r":([a-zA-Z_][a-zA-Z0-9_]*)", "?", sql)
        cur = con.execute(sqlite_sql, sqlite_params)
        rows = [dict(r) for r in cur.fetchall()]
        con.close()
        return rows
    except Exception as e:
        print(f"[health-metrics] sqlite3 query error: {e} | sql: {sql[:80]}")
        return []


def _scalar(sql: str, params: tuple = (), default=0):
    """Run a query that returns a single value."""
    rows = _query(sql, params)
    if not rows:
        return default
    first = list(rows[0].values())
    return first[0] if first else default


def _query_sa(sql: str, **kwargs) -> List[Dict[str, Any]]:
    """SQLAlchemy named-param query helper for parameterised queries."""
    try:
        from app.database import engine
        from sqlalchemy import text as _text
        with engine.connect() as conn:
            result = conn.execute(_text(sql), kwargs)
            cols = result.keys()
            return [dict(zip(cols, row)) for row in result.fetchall()]
    except Exception:
        return []


def _status(value: float, good_threshold: float = 80, warn_threshold: float = 60) -> str:
    if value is None:
        return "neutral"
    if value >= good_threshold:
        return "healthy"
    if value >= warn_threshold:
        return "warning"
    return "critical"


def _pct_status(pct: float) -> str:
    return _status(pct, 85, 65)


# ── Data fetchers ─────────────────────────────────────────────────────────────

def _fetch_datasets(dataset_id: Optional[int] = None) -> List[Dict]:
    sql = "SELECT id, display_name, physical_name, source_id FROM datasets"
    params = ()
    if dataset_id:
        sql += " WHERE id = :dataset_id"
        params = {"dataset_id": dataset_id}
    return _query(sql, params)


def _fetch_profiling_runs(dataset_id: Optional[int] = None) -> List[Dict]:
    sql = "SELECT * FROM profiling_runs ORDER BY started_at DESC LIMIT 100"
    params = ()
    if dataset_id:
        sql = "SELECT * FROM profiling_runs WHERE dataset_id = :dataset_id ORDER BY started_at DESC LIMIT 20"
        params = {"dataset_id": dataset_id}
    return _query(sql, params)


def _fetch_latest_run(dataset_id: Optional[int] = None) -> Optional[Dict]:
    if dataset_id:
        rows = _query(
            "SELECT * FROM profiling_runs WHERE dataset_id = :dataset_id AND status = 'completed' "
            "ORDER BY completed_at DESC LIMIT 1", {"dataset_id": dataset_id}
        )
    else:
        rows = _query(
            "SELECT * FROM profiling_runs WHERE status = 'completed' "
            "ORDER BY completed_at DESC LIMIT 1"
        )
    return rows[0] if rows else None


def _fetch_column_metrics(run_id: int) -> List[Dict]:
    return _query(
        "SELECT * FROM column_metrics WHERE run_id = :run_id", {"run_id": run_id}
    )


def _fetch_quality_checks(run_id: Optional[int] = None, dataset_id: Optional[int] = None) -> List[Dict]:
    if run_id:
        return _query("SELECT * FROM quality_checks WHERE run_id = :run_id", {"run_id": run_id})
    if dataset_id:
        return _query(
            "SELECT qc.* FROM quality_checks qc "
            "JOIN profiling_runs pr ON qc.run_id = pr.id "
            "WHERE pr.dataset_id = :dataset_id", {"dataset_id": dataset_id}
        )
    return _query("SELECT * FROM quality_checks LIMIT 500")


def _fetch_dq_rules(dataset_id: Optional[int] = None) -> List[Dict]:
    if dataset_id:
        return _query("SELECT * FROM dq_rules WHERE dataset_id = :dataset_id", {"dataset_id": dataset_id})
    return _query("SELECT * FROM dq_rules LIMIT 500")


def _fetch_anomalies(dataset_id: Optional[int] = None) -> List[Dict]:
    if dataset_id:
        return _query(
            "SELECT a.* FROM anomalies a "
            "JOIN profiling_runs pr ON a.run_id = pr.id "
            "WHERE pr.dataset_id = :dataset_id", {"dataset_id": dataset_id}
        )
    return _query("SELECT * FROM anomalies LIMIT 500")


def _fetch_drift_records(run_id: Optional[int] = None) -> List[Dict]:
    if run_id:
        return _query("SELECT * FROM drift_records WHERE run_id = :run_id", {"run_id": run_id})
    return _query("SELECT * FROM drift_records LIMIT 500")


def _fetch_lineage_edges() -> List[Dict]:
    try:
        return _query("SELECT * FROM lineage_edges LIMIT 500")
    except:
        return []


def _fetch_kg_nodes() -> List[Dict]:
    try:
        return _query("SELECT * FROM kg_nodes LIMIT 1000")
    except:
        return []


def _fetch_kg_edges() -> List[Dict]:
    try:
        return _query("SELECT * FROM kg_edges LIMIT 1000")
    except:
        return []


def _fetch_governance_policies() -> List[Dict]:
    try:
        return _query("SELECT * FROM governance_policies LIMIT 200")
    except:
        return []


def _fetch_data_sources() -> List[Dict]:
    try:
        return _query("SELECT * FROM data_sources LIMIT 200")
    except:
        return []


def _fetch_llm_interactions(dataset_id: Optional[int] = None) -> List[Dict]:
    try:
        if dataset_id:
            return _query(
                "SELECT * FROM llm_interactions WHERE dataset_id = :dataset_id ORDER BY created_at DESC LIMIT 100",
                {"dataset_id": dataset_id}
            )
        return _query("SELECT * FROM llm_interactions ORDER BY created_at DESC LIMIT 200")
    except:
        return []


# ── Metric builders ───────────────────────────────────────────────────────────

def _build_global_llm_tab(runs, llm_rows) -> Dict:
    total_calls = len(llm_rows)
    # Estimate hallucination rate: rows where output JSON is malformed or empty
    bad = sum(1 for r in llm_rows if not r.get("response_text") or len(str(r.get("response_text", ""))) < 10)
    hallucination_rate = round((bad / total_calls * 100) if total_calls else 0, 1)

    # Average latency
    latencies = [r.get("duration_ms", 0) or 0 for r in llm_rows if r.get("duration_ms")]
    avg_latency_ms = round(sum(latencies) / len(latencies) if latencies else 0)

    # Relevance: approximate as % where response_text is not empty
    relevant = sum(1 for r in llm_rows if r.get("response_text") and len(str(r.get("response_text", ""))) > 20)
    relevance = round((relevant / total_calls * 100) if total_calls else 100, 1)

    # Consistency: approximate as how often same prompt_type returns similar length responses
    consistency = max(0.0, 100.0 - hallucination_rate * 1.5)

    hallucination_status = _status(100 - hallucination_rate, 95, 80)
    latency_status = "healthy" if avg_latency_ms < 5000 else "warning" if avg_latency_ms < 15000 else "critical"

    return {
        "tab": "Global AI / LLM",
        "metrics": [
            {
                "id": "hallucination_rate",
                "label": "Hallucination Rate",
                "value": hallucination_rate,
                "unit": "%",
                "status": hallucination_status,
                "formula": "hallucinated_outputs / total_llm_calls × 100",
                "details": {"total_calls": total_calls, "flagged": bad},
            },
            {
                "id": "avg_llm_latency_ms",
                "label": "Avg LLM Latency",
                "value": avg_latency_ms,
                "unit": "ms",
                "status": latency_status,
                "formula": "sum(duration_ms) / count(llm_calls)",
                "details": {"samples": len(latencies)},
            },
            {
                "id": "response_relevance",
                "label": "Response Relevance",
                "value": relevance,
                "unit": "%",
                "status": _pct_status(relevance),
                "formula": "non_empty_responses / total_calls × 100",
                "details": {"relevant": relevant, "total": total_calls},
            },
            {
                "id": "response_consistency",
                "label": "Response Consistency",
                "value": round(consistency, 1),
                "unit": "%",
                "status": _pct_status(consistency),
                "formula": "100 - (hallucination_rate × 1.5), capped at 0",
                "details": {},
            },
        ],
        "explainability": {
            "overview": "Global AI/LLM metrics track the overall quality and reliability of the Llama 3.3 70B model used across all platform features.",
            "improvement": "Reduce hallucination rate by improving prompt specificity. Monitor latency for cold-start issues on Azure AI Foundry.",
            "low_success_rate": "A high hallucination rate means model outputs are being flagged as empty or malformed. Check prompt construction and model endpoint availability.",
        },
    }


def _build_profiling_tab(runs, col_metrics_all) -> Dict:
    total_runs = len(runs)
    completed_runs = sum(1 for r in runs if r.get("status") == "completed")
    success_rate = round((completed_runs / total_runs * 100) if total_runs else 0, 1)

    # Metadata grounding: % of completed runs that have an AI description
    grounded = sum(1 for r in runs if r.get("status") == "completed" and r.get("ai_description"))
    grounding_score = round((grounded / completed_runs * 100) if completed_runs else 0, 1)

    # Drift detection accuracy: % of runs that have drift records
    runs_with_drift = _scalar("SELECT COUNT(DISTINCT run_id) FROM drift_records")
    drift_acc = round((runs_with_drift / completed_runs * 100) if completed_runs else 0, 1)

    # Average runtime
    durations = []
    for r in runs:
        if r.get("started_at") and r.get("completed_at") and r.get("status") == "completed":
            try:
                start = datetime.fromisoformat(str(r["started_at"]).replace("Z", "+00:00").replace(" ", "T"))
                end = datetime.fromisoformat(str(r["completed_at"]).replace("Z", "+00:00").replace(" ", "T"))
                durations.append((end - start).total_seconds())
            except:
                pass
    avg_runtime = round(sum(durations) / len(durations) if durations else 0)

    return {
        "tab": "Profiling AI",
        "metrics": [
            {
                "id": "profiling_success_rate",
                "label": "Profiling Success Rate",
                "value": success_rate,
                "unit": "%",
                "status": _pct_status(success_rate),
                "formula": "completed_runs / total_runs × 100",
                "details": {"completed": completed_runs, "total": total_runs},
            },
            {
                "id": "metadata_grounding_score",
                "label": "Metadata Grounding Score",
                "value": grounding_score,
                "unit": "%",
                "status": _pct_status(grounding_score),
                "formula": "runs_with_ai_description / completed_runs × 100",
                "details": {"grounded": grounded, "completed": completed_runs},
            },
            {
                "id": "drift_detection_accuracy",
                "label": "Drift Detection Coverage",
                "value": drift_acc,
                "unit": "%",
                "status": _pct_status(drift_acc),
                "formula": "runs_with_drift_records / completed_runs × 100",
                "details": {"runs_with_drift": int(runs_with_drift)},
            },
            {
                "id": "avg_profiling_runtime_s",
                "label": "Avg Profiling Runtime",
                "value": avg_runtime,
                "unit": "s",
                "status": "healthy" if avg_runtime < 120 else "warning" if avg_runtime < 600 else "critical",
                "formula": "mean(completed_at - started_at) in seconds",
                "details": {"samples": len(durations)},
            },
        ],
        "explainability": {
            "overview": "Profiling AI metrics measure how reliably the platform reads and profiles datasets, generates metadata, and detects distribution changes.",
            "improvement": "Improve metadata grounding by ensuring the LLM endpoint is responsive. Reduce runtime by optimising dataset size before profiling.",
            "low_success_rate": "Failed profiling runs usually indicate network errors reaching the data source, or data source credential expiry.",
        },
    }


def _build_dq_scores_tab(runs, col_metrics_all) -> Dict:
    # Average health score across all completed runs
    health_scores = [r.get("overall_health", 0) or 0 for r in runs if r.get("status") == "completed"]
    avg_health = round(sum(health_scores) / len(health_scores) if health_scores else 0, 1)

    # Rule compliance: average across all runs that have it
    rule_scores = [r.get("rule_compliance", None) for r in runs if r.get("rule_compliance") is not None]
    avg_rule_compliance = round(sum(rule_scores) / len(rule_scores) if rule_scores else 0, 1)

    # Health degradation velocity: slope of health over last 5 runs
    recent = sorted([r for r in runs if r.get("status") == "completed" and r.get("overall_health")],
                    key=lambda x: str(x.get("completed_at", "")))[-5:]
    if len(recent) >= 2:
        h_vals = [r.get("overall_health", 0) for r in recent]
        velocity = round(h_vals[-1] - h_vals[0], 1)  # positive = improving
    else:
        velocity = 0.0
    vel_status = "healthy" if velocity >= -5 else "warning" if velocity >= -15 else "critical"

    # Health score accuracy: % within expected range (not extreme outliers)
    accurate = sum(1 for s in health_scores if 0 <= s <= 100)
    accuracy = round((accurate / len(health_scores) * 100) if health_scores else 100, 1)

    return {
        "tab": "DQ Scores",
        "metrics": [
            {
                "id": "health_score_accuracy",
                "label": "Health Score Accuracy",
                "value": accuracy,
                "unit": "%",
                "status": _pct_status(accuracy),
                "formula": "scores_in_valid_range / total_scores × 100",
                "details": {"valid": accurate, "total": len(health_scores)},
            },
            {
                "id": "rule_compliance_accuracy",
                "label": "Avg Rule Compliance Score",
                "value": avg_rule_compliance,
                "unit": "%",
                "status": _pct_status(avg_rule_compliance),
                "formula": "mean(rule_compliance) across completed runs",
                "details": {"samples": len(rule_scores)},
            },
            {
                "id": "avg_health_score",
                "label": "Avg Dataset Health Score",
                "value": avg_health,
                "unit": "%",
                "status": _pct_status(avg_health),
                "formula": "mean(overall_health) across all completed runs",
                "details": {"runs": len(health_scores)},
            },
            {
                "id": "health_degradation_velocity",
                "label": "Health Degradation Velocity",
                "value": velocity,
                "unit": "pts",
                "status": vel_status,
                "formula": "health_score[last] - health_score[first] over last 5 runs (negative = degrading)",
                "details": {"window": len(recent)},
            },
        ],
        "explainability": {
            "overview": "DQ Scores metrics track how accurately and consistently the platform computes data quality scores across datasets.",
            "improvement": "To improve degradation velocity, investigate the dimensions (completeness, validity) that are falling in the monitoring tab.",
            "low_success_rate": "Low rule compliance accuracy suggests rules are firing frequently — review rule thresholds or address the root data quality issues.",
        },
    }


def _build_dq_rules_tab(rules) -> Dict:
    total_rules = len(rules)
    active_rules = [r for r in rules if r.get("is_active") or r.get("status") == "active"]
    n_active = len(active_rules)

    # Execution success rate: rules with recent check results
    executed = _scalar("SELECT COUNT(DISTINCT rule_id) FROM quality_checks WHERE rule_id IS NOT NULL")
    exec_rate = round((executed / n_active * 100) if n_active else 0, 1)

    # Rule recommendation acceptance rate
    ai_suggested = [r for r in rules if r.get("source") in ("llm", "ai", "recommended")]
    accepted = [r for r in ai_suggested if r.get("is_active")]
    acceptance_rate = round((len(accepted) / len(ai_suggested) * 100) if ai_suggested else 0, 1)

    # Hallucinated rule rate: AI rules referencing columns that don't exist
    # We approximate: if a rule has no associated quality_checks, it may be malformed
    all_check_rule_ids = set(
        r.get("rule_id") for r in _query("SELECT DISTINCT rule_id FROM quality_checks WHERE rule_id IS NOT NULL")
    )
    ai_never_fired = [r for r in ai_suggested if r.get("id") not in all_check_rule_ids]
    hallucinated_rate = round((len(ai_never_fired) / len(ai_suggested) * 100) if ai_suggested else 0, 1)
    hallucination_rate = round(hallucinated_rate, 1)

    return {
        "tab": "DQ Rules",
        "metrics": [
            {
                "id": "rule_execution_success_rate",
                "label": "Rule Execution Success Rate",
                "value": exec_rate,
                "unit": "%",
                "status": _pct_status(exec_rate),
                "formula": "rules_with_check_results / active_rules × 100",
                "details": {"executed": int(executed), "active": n_active, "total": total_rules},
            },
            {
                "id": "rule_recommendation_acceptance_rate",
                "label": "AI Rule Acceptance Rate",
                "value": acceptance_rate,
                "unit": "%",
                "status": _pct_status(acceptance_rate),
                "formula": "accepted_ai_rules / total_ai_suggested_rules × 100",
                "details": {"accepted": len(accepted), "suggested": len(ai_suggested)},
            },
            {
                "id": "hallucinated_rule_rate",
                "label": "Hallucinated Rule Rate",
                "value": hallucination_rate,
                "unit": "%",
                "status": _status(100 - hallucination_rate, 90, 75),
                "formula": "ai_rules_never_fired / total_ai_rules × 100 (proxy for invalid rules)",
                "details": {"never_fired": len(ai_never_fired), "ai_rules": len(ai_suggested)},
            },
        ],
        "explainability": {
            "overview": "DQ Rules metrics measure the quality and effectiveness of custom validation rules — both human-authored and AI-recommended.",
            "improvement": "Increase acceptance rate by reviewing the AI rule recommendations after each profiling run. Activate rules that are relevant to your domain.",
            "low_success_rate": "Low execution rate means rules aren't being evaluated. Run a DQ scoring job to execute all active rules.",
        },
    }


def _build_monitoring_tab(runs, drift_records) -> Dict:
    total = len(runs)
    completed = sum(1 for r in runs if r.get("status") == "completed")

    # Monitoring uptime = % of scheduled time slots with a completed run
    uptime = round((completed / total * 100) if total else 0, 1)

    # Drift detection precision: % of drift records with severity HIGH or CRITICAL
    # (proxy: real drifts tend to be marked high/critical)
    real_drifts = [d for d in drift_records if d.get("severity") in ("HIGH", "CRITICAL", "MEDIUM")]
    precision = round((len(real_drifts) / len(drift_records) * 100) if drift_records else 0, 1)

    # Forecast error rate: approximate using actual vs predicted health slope
    # Use std deviation of health scores as proxy for forecast accuracy
    health_vals = [r.get("overall_health", 0) or 0 for r in runs if r.get("overall_health")]
    if len(health_vals) >= 2:
        mean_h = sum(health_vals) / len(health_vals)
        variance = sum((v - mean_h) ** 2 for v in health_vals) / len(health_vals)
        std_dev = variance ** 0.5
        forecast_error = round(min(100, std_dev), 1)
    else:
        forecast_error = 0.0

    return {
        "tab": "Monitoring & Trends",
        "metrics": [
            {
                "id": "monitoring_uptime",
                "label": "Monitoring Uptime",
                "value": uptime,
                "unit": "%",
                "status": _pct_status(uptime),
                "formula": "completed_runs / scheduled_runs × 100",
                "details": {"completed": completed, "total": total},
            },
            {
                "id": "drift_detection_precision",
                "label": "Drift Alert Precision",
                "value": precision,
                "unit": "%",
                "status": _pct_status(precision),
                "formula": "significant_drift_records (MEDIUM+) / total_drift_records × 100",
                "details": {"significant": len(real_drifts), "total": len(drift_records)},
            },
            {
                "id": "forecast_error_rate",
                "label": "Health Score Volatility",
                "value": forecast_error,
                "unit": "pts std",
                "status": "healthy" if forecast_error < 5 else "warning" if forecast_error < 15 else "critical",
                "formula": "stddev(overall_health) across recent runs — lower is more predictable",
                "details": {"samples": len(health_vals)},
            },
        ],
        "explainability": {
            "overview": "Monitoring metrics reflect how consistently the platform tracks dataset health and surfaces real quality signals vs noise.",
            "improvement": "Schedule profiling runs at regular intervals using the built-in scheduler to improve monitoring uptime.",
            "high_drift": "High volatility in health scores suggests unstable data quality. Check source data pipeline stability.",
        },
    }


def _build_anomalies_tab(anomalies, quality_checks) -> Dict:
    total_checks = len(quality_checks)
    open_anomalies = [a for a in anomalies if a.get("status") in ("open", "investigating")]
    resolved = [a for a in anomalies if a.get("status") == "resolved"]

    # Precision: resolved / (resolved + still open) — proxy for resolution rate
    total_detected = len(anomalies)
    precision = round((len(resolved) / total_detected * 100) if total_detected else 100, 1)

    # Recall: open_anomalies against quality_checks count
    open_checks = [c for c in quality_checks if c.get("status") in ("open", "investigating")]
    recall = round((len(open_anomalies) / len(open_checks) * 100) if open_checks else 100, 1)
    recall = min(100.0, recall)

    # RCA hallucination rate: anomalies with empty or short RCA text
    rca_attempts = [a for a in anomalies if a.get("root_cause_analysis")]
    rca_bad = [a for a in rca_attempts if len(str(a.get("root_cause_analysis", ""))) < 30]
    rca_hall = round((len(rca_bad) / len(rca_attempts) * 100) if rca_attempts else 0, 1)

    # Auto-fix success rate
    auto_fixes = [a for a in anomalies if a.get("auto_fix_applied")]
    fix_success = [a for a in auto_fixes if a.get("status") == "resolved"]
    fix_rate = round((len(fix_success) / len(auto_fixes) * 100) if auto_fixes else 0, 1)

    return {
        "tab": "Anomalies AI",
        "metrics": [
            {
                "id": "anomaly_precision",
                "label": "Detection Precision",
                "value": precision,
                "unit": "%",
                "status": _pct_status(precision),
                "formula": "resolved_anomalies / total_detected × 100",
                "details": {"resolved": len(resolved), "total": total_detected, "open": len(open_anomalies)},
            },
            {
                "id": "anomaly_recall",
                "label": "Detection Recall",
                "value": recall,
                "unit": "%",
                "status": _pct_status(recall),
                "formula": "anomaly_records / open_quality_checks × 100",
                "details": {"anomalies": len(open_anomalies), "open_checks": len(open_checks)},
            },
            {
                "id": "rca_hallucination_rate",
                "label": "RCA Hallucination Rate",
                "value": rca_hall,
                "unit": "%",
                "status": _status(100 - rca_hall, 90, 75),
                "formula": "malformed_rca_outputs / total_rca_attempts × 100",
                "details": {"bad": len(rca_bad), "attempts": len(rca_attempts)},
            },
            {
                "id": "auto_fix_success_rate",
                "label": "Auto-Fix Success Rate",
                "value": fix_rate,
                "unit": "%",
                "status": _pct_status(fix_rate) if auto_fixes else "neutral",
                "formula": "successful_auto_fixes / total_auto_fix_attempts × 100",
                "details": {"success": len(fix_success), "attempts": len(auto_fixes)},
            },
        ],
        "explainability": {
            "overview": "Anomaly AI metrics measure how accurately the platform detects, diagnoses, and resolves data quality violations using AI-powered root cause analysis.",
            "improvement": "Improve precision by tuning anomaly severity thresholds. Improve RCA quality by ensuring the LLM endpoint is available.",
            "many_critical": "High open anomaly count indicates active data quality issues. Prioritize CRITICAL severity anomalies and apply auto-fix where available.",
        },
    }


def _build_lineage_tab() -> Dict:
    edges = _fetch_lineage_edges()
    total_edges = len(edges)
    broken_edges = [e for e in edges if e.get("status") == "broken" or e.get("is_stale")]
    missed_dep_edges = [e for e in edges if e.get("confidence", 1.0) < 0.5]

    datasets_count = _scalar("SELECT COUNT(*) FROM datasets")
    datasets_with_lineage = len(set(e.get("source_dataset_id") for e in edges) |
                                 set(e.get("target_dataset_id") for e in edges))
    coverage = round((datasets_with_lineage / datasets_count * 100) if datasets_count else 0, 1)

    return {
        "tab": "Data Lineage & Impact",
        "metrics": [
            {
                "id": "lineage_coverage",
                "label": "Lineage Coverage",
                "value": coverage,
                "unit": "%",
                "status": _pct_status(coverage),
                "formula": "datasets_with_lineage_mapped / total_datasets × 100",
                "details": {"mapped": datasets_with_lineage, "total": int(datasets_count)},
            },
            {
                "id": "broken_edge_count",
                "label": "Broken Lineage Edges",
                "value": len(broken_edges),
                "unit": "",
                "status": "healthy" if not broken_edges else "warning" if len(broken_edges) < 5 else "critical",
                "formula": "count(lineage_edges WHERE status = 'broken' OR is_stale = true)",
                "details": {"broken": len(broken_edges), "total": total_edges},
            },
            {
                "id": "missed_dependency_rate",
                "label": "Low-Confidence Dependency Rate",
                "value": round((len(missed_dep_edges) / total_edges * 100) if total_edges else 0, 1),
                "unit": "%",
                "status": _pct_status(100 - (len(missed_dep_edges) / total_edges * 100) if total_edges else 100),
                "formula": "edges_with_confidence < 0.5 / total_edges × 100",
                "details": {"low_confidence": len(missed_dep_edges), "total": total_edges},
            },
        ],
        "explainability": {
            "overview": "Lineage metrics track how completely and accurately the platform maps data flow between datasets.",
            "improvement": "Improve lineage coverage by registering all datasets and running profiling — lineage edges are auto-detected during profiling.",
            "low_coverage": "Low lineage coverage means most datasets don't have upstream/downstream mapping. Run profiling on all registered datasets.",
        },
    }


def _build_kg_tab() -> Dict:
    nodes = _fetch_kg_nodes()
    edges = _fetch_kg_edges()

    total_nodes = len(nodes)
    total_edges = len(edges)
    # Relationship precision: edges with high confidence
    high_conf = [e for e in edges if (e.get("confidence") or e.get("weight") or 0) >= 0.7]
    precision = round((len(high_conf) / total_edges * 100) if total_edges else 0, 1)

    # Column mapping accuracy: node pairs that are column-type and have edges
    col_nodes = [n for n in nodes if n.get("node_type") in ("column", "COLUMN")]
    col_edges = [e for e in edges if e.get("edge_type") in ("similar_to", "references", "MAPS_TO")]
    mapping_acc = round((len(col_edges) / len(col_nodes) * 100) if col_nodes else 0, 1)
    mapping_acc = min(100.0, mapping_acc)

    # Hallucinated relationship rate: edges with no confidence score
    null_conf = [e for e in edges if e.get("confidence") is None and e.get("weight") is None]
    hall_rate = round((len(null_conf) / total_edges * 100) if total_edges else 0, 1)

    return {
        "tab": "Knowledge Graph AI",
        "metrics": [
            {
                "id": "kg_relationship_precision",
                "label": "Relationship Precision",
                "value": precision,
                "unit": "%",
                "status": _pct_status(precision),
                "formula": "edges_with_confidence ≥ 0.7 / total_edges × 100",
                "details": {"high_confidence": len(high_conf), "total": total_edges, "nodes": total_nodes},
            },
            {
                "id": "kg_column_mapping_accuracy",
                "label": "Column Mapping Coverage",
                "value": mapping_acc,
                "unit": "%",
                "status": _pct_status(mapping_acc),
                "formula": "column_mapping_edges / column_nodes × 100",
                "details": {"column_nodes": len(col_nodes), "mapping_edges": len(col_edges)},
            },
            {
                "id": "kg_hallucinated_relationship_rate",
                "label": "Hallucinated Relationship Rate",
                "value": hall_rate,
                "unit": "%",
                "status": _status(100 - hall_rate, 90, 75),
                "formula": "edges_with_no_confidence_score / total_edges × 100",
                "details": {"unscored_edges": len(null_conf), "total": total_edges},
            },
        ],
        "explainability": {
            "overview": "Knowledge Graph metrics measure how accurately the AI-powered graph construction identifies real semantic relationships between dataset columns.",
            "improvement": "Run knowledge graph construction after profiling to increase node and edge counts. Higher confidence scores emerge from more column data.",
            "low_coverage": "Low relationship precision often means the LLM is generating connections without enough column metadata context.",
        },
    }


def _build_dq_assistant_tab(llm_rows) -> Dict:
    # Agent routing accuracy: LLM calls that hit the correct agent
    # We approximate using calls tagged with agent_type
    calls_with_agent = [r for r in llm_rows if r.get("agent_type") or r.get("feature")]
    routing_acc = round((len(calls_with_agent) / len(llm_rows) * 100) if llm_rows else 0, 1)

    # Hallucination rate in assistant responses
    bad = sum(1 for r in llm_rows if not r.get("response_text") or len(str(r.get("response_text", ""))) < 10)
    hall_rate = round((bad / len(llm_rows) * 100) if llm_rows else 0, 1)

    # ActionAgent success rate: calls where action was applied
    action_calls = [r for r in llm_rows if r.get("action_taken") or r.get("action_type")]
    action_success = [r for r in action_calls if r.get("action_success") or r.get("status") == "success"]
    action_rate = round((len(action_success) / len(action_calls) * 100) if action_calls else 0, 1)

    # Retrieval grounding score: calls with context provided
    grounded_calls = [r for r in llm_rows if r.get("context") or r.get("has_context")]
    grounding = round((len(grounded_calls) / len(llm_rows) * 100) if llm_rows else 0, 1)

    return {
        "tab": "DQ Assistant / AI Agent",
        "metrics": [
            {
                "id": "agent_routing_accuracy",
                "label": "Agent Routing Accuracy",
                "value": routing_acc,
                "unit": "%",
                "status": _pct_status(routing_acc),
                "formula": "calls_with_agent_tag / total_llm_calls × 100",
                "details": {"tagged": len(calls_with_agent), "total": len(llm_rows)},
            },
            {
                "id": "assistant_hallucination_rate",
                "label": "Assistant Hallucination Rate",
                "value": hall_rate,
                "unit": "%",
                "status": _status(100 - hall_rate, 90, 75),
                "formula": "empty_or_malformed_responses / total_calls × 100",
                "details": {"bad": bad, "total": len(llm_rows)},
            },
            {
                "id": "action_agent_success_rate",
                "label": "ActionAgent Success Rate",
                "value": action_rate,
                "unit": "%",
                "status": _pct_status(action_rate) if action_calls else "neutral",
                "formula": "successful_agentic_actions / total_action_attempts × 100",
                "details": {"success": len(action_success), "attempts": len(action_calls)},
            },
            {
                "id": "retrieval_grounding_score",
                "label": "Retrieval Grounding Score",
                "value": grounding,
                "unit": "%",
                "status": _pct_status(grounding),
                "formula": "calls_with_context / total_calls × 100",
                "details": {"grounded": len(grounded_calls), "total": len(llm_rows)},
            },
        ],
        "explainability": {
            "overview": "DQ Assistant metrics measure the quality and effectiveness of the conversational AI agent and its sub-agents across all platform interactions.",
            "improvement": "Improve routing accuracy by ensuring the MasterAgent prompt includes clear feature descriptions. Improve grounding by passing dataset context with every query.",
            "low_satisfaction": "Low grounding score means the assistant is answering from model knowledge rather than actual dataset data. Add context injection to prompts.",
        },
    }


def _build_governance_tab() -> Dict:
    policies = _fetch_governance_policies()
    total = len(policies)
    active = [p for p in policies if p.get("is_active") or p.get("status") == "active"]
    ai_suggested = [p for p in policies if p.get("source") in ("llm", "ai")]
    accepted = [p for p in ai_suggested if p.get("is_active") or p.get("status") == "active"]

    adoption_rate = round((len(accepted) / len(ai_suggested) * 100) if ai_suggested else 0, 1)

    # Classification accuracy: AI-tagged columns with sensitivity labels
    classified = _scalar(
        "SELECT COUNT(*) FROM column_metrics WHERE sensitivity_label IS NOT NULL AND sensitivity_label != ''"
    )
    total_cols = _scalar("SELECT COUNT(*) FROM column_metrics")
    class_acc = round((classified / total_cols * 100) if total_cols else 0, 1)

    # Audit log completeness: system_events or audit_log completeness
    try:
        audit_entries = _scalar("SELECT COUNT(*) FROM audit_log")
        total_actions = _scalar("SELECT COUNT(*) FROM profiling_runs") + _scalar("SELECT COUNT(*) FROM dq_rules")
        audit_completeness = round((audit_entries / total_actions * 100) if total_actions else 100, 1)
        audit_completeness = min(100.0, audit_completeness)
    except:
        audit_completeness = 0.0

    return {
        "tab": "Governance & Settings",
        "metrics": [
            {
                "id": "policy_adoption_rate",
                "label": "Policy Adoption Rate",
                "value": adoption_rate,
                "unit": "%",
                "status": _pct_status(adoption_rate) if ai_suggested else "neutral",
                "formula": "accepted_ai_policies / total_ai_suggested_policies × 100",
                "details": {"accepted": len(accepted), "suggested": len(ai_suggested), "total_policies": total},
            },
            {
                "id": "classification_accuracy",
                "label": "Column Classification Coverage",
                "value": class_acc,
                "unit": "%",
                "status": _pct_status(class_acc),
                "formula": "columns_with_sensitivity_label / total_columns × 100",
                "details": {"classified": int(classified), "total": int(total_cols)},
            },
            {
                "id": "audit_log_completeness",
                "label": "Audit Log Completeness",
                "value": audit_completeness,
                "unit": "%",
                "status": _pct_status(audit_completeness),
                "formula": "audit_entries / (profiling_runs + dq_rules) × 100 (proxy)",
                "details": {},
            },
        ],
        "explainability": {
            "overview": "Governance metrics track how well the platform enforces data policies, classifies sensitive data, and maintains an auditable activity trail.",
            "improvement": "Increase policy adoption by reviewing AI-suggested governance policies after each profiling run. Activate the ones relevant to your data domain.",
            "low_adoption": "Low adoption rate means AI governance recommendations are not being reviewed or activated. Schedule a governance review cycle.",
        },
    }


def _build_system_tab(runs) -> Dict:
    # API throughput: approximate from profiling run frequency
    total_runs = len(runs)
    if total_runs >= 2:
        try:
            sorted_runs = sorted([r for r in runs if r.get("started_at")], key=lambda x: str(x["started_at"]))
            first = datetime.fromisoformat(str(sorted_runs[0]["started_at"]).replace("Z", "+00:00").replace(" ", "T"))
            last  = datetime.fromisoformat(str(sorted_runs[-1]["started_at"]).replace("Z", "+00:00").replace(" ", "T"))
            hours = max(1, (last - first).total_seconds() / 3600)
            throughput = round(total_runs / hours, 2)
        except:
            throughput = 0.0
    else:
        throughput = 0.0

    # Uptime approximation: runs completing without error
    completed = sum(1 for r in runs if r.get("status") == "completed")
    failed = sum(1 for r in runs if r.get("status") == "failed")
    uptime = round(((completed) / (completed + failed) * 100) if (completed + failed) else 100, 1)

    # Average API response time — not directly available, use profiling runtime as proxy
    durations = []
    for r in runs:
        if r.get("started_at") and r.get("completed_at") and r.get("status") == "completed":
            try:
                start = datetime.fromisoformat(str(r["started_at"]).replace("Z", "+00:00").replace(" ", "T"))
                end = datetime.fromisoformat(str(r["completed_at"]).replace("Z", "+00:00").replace(" ", "T"))
                durations.append((end - start).total_seconds() * 1000)  # ms
            except:
                pass
    avg_response_ms = round(sum(durations) / len(durations) if durations else 0)

    return {
        "tab": "System / Platform",
        "metrics": [
            {
                "id": "system_uptime",
                "label": "System Uptime",
                "value": uptime,
                "unit": "%",
                "status": _pct_status(uptime),
                "formula": "completed_runs / (completed + failed_runs) × 100",
                "details": {"completed": completed, "failed": failed},
            },
            {
                "id": "api_throughput",
                "label": "Processing Throughput",
                "value": throughput,
                "unit": "runs/hr",
                "status": "healthy" if throughput > 0 else "neutral",
                "formula": "total_runs / time_span_hours",
                "details": {"total_runs": total_runs},
            },
            {
                "id": "avg_job_duration_ms",
                "label": "Avg Job Duration",
                "value": avg_response_ms,
                "unit": "ms",
                "status": "healthy" if avg_response_ms < 60000 else "warning" if avg_response_ms < 300000 else "critical",
                "formula": "mean(completed_at - started_at) in milliseconds across completed runs",
                "details": {"samples": len(durations)},
            },
        ],
        "explainability": {
            "overview": "System metrics reflect the operational health of the AI DQM platform infrastructure — throughput, reliability, and responsiveness.",
            "improvement": "Monitor failed runs via the Databricks app logs. High job duration usually means large datasets — consider sampling or pagination.",
            "low_success_rate": "Low uptime indicates frequent profiling failures. Check data source connectivity and credentials.",
        },
    }


def _build_feedback_tab() -> Dict:
    # Human feedback from quality checks that were manually reviewed
    try:
        reviewed = _scalar(
            "SELECT COUNT(*) FROM quality_checks WHERE reviewed_by IS NOT NULL OR acknowledged_at IS NOT NULL"
        )
        total_checks = _scalar("SELECT COUNT(*) FROM quality_checks")
        ai_acceptance = round((reviewed / total_checks * 100) if total_checks else 0, 1)
    except:
        ai_acceptance = 0.0
        total_checks = 0

    # Satisfaction score — if there's a feedback table
    try:
        avg_rating = _scalar("SELECT AVG(rating) FROM user_feedback WHERE rating IS NOT NULL")
        satisfaction = round(float(avg_rating or 0) * 20, 1)  # assume 1-5 scale → 0-100
    except:
        satisfaction = 0.0

    return {
        "tab": "Human Feedback",
        "metrics": [
            {
                "id": "ai_acceptance_rate",
                "label": "AI Acceptance Rate",
                "value": ai_acceptance,
                "unit": "%",
                "status": _pct_status(ai_acceptance),
                "formula": "reviewed_quality_checks / total_quality_checks × 100",
                "details": {"reviewed": int(reviewed if 'reviewed' in dir() else 0), "total": int(total_checks)},
            },
            {
                "id": "analyst_satisfaction_score",
                "label": "Analyst Satisfaction Score",
                "value": satisfaction,
                "unit": "/100",
                "status": _pct_status(satisfaction) if satisfaction > 0 else "neutral",
                "formula": "mean(user_feedback.rating) × 20 (1–5 scale normalised to 0–100)",
                "details": {},
            },
        ],
        "explainability": {
            "overview": "Human Feedback metrics capture how analysts are engaging with and accepting AI-generated outputs across the platform.",
            "improvement": "Encourage analysts to review and acknowledge quality checks and anomalies. This data drives acceptance rate.",
            "low_satisfaction": "Low satisfaction may indicate UI issues, irrelevant recommendations, or LLM responses that don't match analyst expectations.",
        },
    }


# ── API Endpoints ─────────────────────────────────────────────────────────────

@router.get("/datasets-list")
def get_datasets_list():
    """Return list of all registered datasets for the dataset selector."""
    # Query all possible name columns — handles different schema versions
    rows = _query("SELECT id, name, display_name, physical_name FROM datasets LIMIT 200")
    if not rows:
        # Fallback: minimal query
        rows = _query("SELECT id, name FROM datasets LIMIT 200")
    result = []
    for d in rows:
        display = (
            d.get("display_name")
            or d.get("name")
            or d.get("physical_name")
            or f"Dataset {d.get('id', '?')}"
        )
        result.append({"id": d["id"], "display_name": display})
    return result


@router.get("/")
@router.get("")
def get_all_metrics(dataset_id: Optional[int] = Query(None)):
    """
    Compute and return all health metrics tabs.
    This is the main endpoint consumed by the health dashboard backend.
    """
    runs           = _fetch_profiling_runs(dataset_id)
    latest_run     = _fetch_latest_run(dataset_id)
    col_metrics    = _fetch_column_metrics(latest_run["id"]) if latest_run else []
    quality_checks = _fetch_quality_checks(dataset_id=dataset_id)
    dq_rules       = _fetch_dq_rules(dataset_id)
    anomalies      = _fetch_anomalies(dataset_id)
    drift_records  = _fetch_drift_records(latest_run["id"] if latest_run else None)
    llm_rows       = _fetch_llm_interactions(dataset_id)

    tabs = [
        _build_global_llm_tab(runs, llm_rows),
        _build_profiling_tab(runs, col_metrics),
        _build_dq_scores_tab(runs, col_metrics),
        _build_dq_rules_tab(dq_rules),
        _build_monitoring_tab(runs, drift_records),
        _build_anomalies_tab(anomalies, quality_checks),
        _build_lineage_tab(),
        _build_kg_tab(),
        _build_dq_assistant_tab(llm_rows),
        _build_governance_tab(),
        _build_system_tab(runs),
        _build_feedback_tab(),
    ]

    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "dataset_id": dataset_id,
        "db_path": _get_db_path() or "NOT FOUND",
        "tabs": tabs,
    }