import os
import sqlite3
import logging
from datetime import datetime, timezone, timedelta
from statistics import mean, stdev as pstdev
from typing import Optional
from fastapi import APIRouter, Query

logger = logging.getLogger("ai_dqm.health_metrics")
router = APIRouter()
DB_PATH = os.getenv("DB_PATH", "/tmp/ai-dqm/ai_dqm.db")

def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn

def _columns(conn: sqlite3.Connection, table: str) -> set:
    try:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        return {r["name"] for r in rows}
    except Exception:
        return set()

def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    try:
        conn.execute(f"SELECT 1 FROM {table} LIMIT 1")
        return True
    except Exception:
        return False

def _one(conn: sqlite3.Connection, sql: str, params=()) -> sqlite3.Row | None:
    try:
        return conn.execute(sql, params).fetchone()
    except Exception as e:
        logger.warning(f"Query failed: {e} | SQL: {sql[:120]}")
        return None

def _all(conn: sqlite3.Connection, sql: str, params=()) -> list:
    try:
        return conn.execute(sql, params).fetchall()
    except Exception as e:
        logger.warning(f"Query failed: {e} | SQL: {sql[:120]}")
        return []

def _scalar(conn, sql, params=(), default=None):
    row = _one(conn, sql, params)
    if row is None: return default
    v = row[0]
    return default if v is None else v

def _esc(s: str) -> str:
    return str(s).replace("'", "''")

def _in_clause(col: str, values: set | list) -> str:
    if not values: return "1=0"
    escaped = [_esc(v) for v in values]
    parts = ["'" + v + "'" for v in escaped]
    return f"{col} IN ({','.join(parts)})"

def safe_pct(num: float | int, den: float | int) -> float | None:
    if not den: return None
    return round((num / den) * 100, 2)

def _status(value, *, healthy_ge=None, critical_lt=None, healthy_le=None, critical_gt=None) -> str:
    if value is None: return "neutral"
    if healthy_ge is not None and critical_lt is not None:
        if value >= healthy_ge: return "healthy"
        if value < critical_lt: return "critical"
        return "warning"
    if healthy_le is not None and critical_gt is not None:
        if value <= healthy_le: return "healthy"
        if value > critical_gt: return "critical"
        return "warning"
    return "neutral"

def M(id_, label, value, unit, status, formula, details=None, **extra) -> dict:
    return {"id": id_, "label": label, "value": value, "unit": unit, "status": status, "formula": formula, "details": details or {}, **extra}

def _introspect(conn) -> dict:
    pr_cols = _columns(conn, "profiling_runs")
    tc_cols = _columns(conn, "temporal_checks")
    cp_cols = _columns(conn, "column_profiles")
    dr_cols = _columns(conn, "drift_records")
    dq_cols = _columns(conn, "dq_rules")

    pr_statuses = set()
    if _table_exists(conn, "profiling_runs"):
        rows = _all(conn, "SELECT DISTINCT status FROM profiling_runs WHERE status IS NOT NULL")
        pr_statuses = {r["status"] for r in rows}
    completed_keywords = {'completed', 'success', 'finished', 'done', 'succeeded', 'pass', 'passed'}
    failed_keywords = {'failed', 'error', 'failure', 'cancelled', 'aborted', 'fail'}
    pr_completed = [s for s in pr_statuses if s.lower() in completed_keywords]
    pr_failed = [s for s in pr_statuses if s.lower() in failed_keywords]
    if not pr_completed and pr_statuses: pr_completed = [s for s in pr_statuses if s not in pr_failed]
    # If the table is empty OR all rows have NULL status, fall back to COUNT(*) so
    # metrics still reflect the real row counts (0) instead of silently 1=0-ing.
    if not pr_completed and not pr_statuses:
        # Table is empty or all statuses are NULL — treat every row as "completed"
        pr_completed_filter = "1=1"
    elif not pr_completed:
        # Statuses exist but none matched known completed keywords — use all non-failed
        pr_completed_filter = _in_clause("status", list(pr_statuses - set(pr_failed)))
    else:
        pr_completed_filter = _in_clause("status", pr_completed)
    pr_failed_filter = _in_clause("status", pr_failed) if pr_failed else "1=0"

    dq_statuses = set()
    if _table_exists(conn, "dq_rules"):
        rows = _all(conn, "SELECT DISTINCT status FROM dq_rules WHERE status IS NOT NULL")
        dq_statuses = {r["status"] for r in rows}
    dq_active = [s for s in dq_statuses if s.lower() in {'active', 'enabled', 'on', 'true', '1'}]
    if not dq_active and dq_statuses: dq_active = [s for s in dq_statuses if s.lower() not in {'inactive', 'disabled', 'off', 'false', '0', 'deleted', 'archived'}]
    if not dq_active and not dq_statuses:
        dq_active_filter = "1=1"
    elif not dq_active:
        dq_active_filter = "1=1"  # no inactive detected either — treat all as active
    else:
        dq_active_filter = _in_clause("status", dq_active)

    severity_dist = {}
    drift_has_severity = False
    drift_magnitude_col = None
    if _table_exists(conn, "drift_records"):
        if "severity" in dr_cols:
            rows = _all(conn, "SELECT severity, COUNT(*) as cnt FROM drift_records GROUP BY severity")
            severity_dist = {r["severity"]: r["cnt"] for r in rows if r["severity"]}
            drift_has_severity = len(severity_dist) > 0
        if not drift_has_severity:
            for candidate in ("magnitude", "change_value", "delta", "value", "score"):
                if candidate in dr_cols: drift_magnitude_col = candidate; break

    tc_statuses = set()
    tc_context_col = None
    if _table_exists(conn, "temporal_checks"):
        rows = _all(conn, "SELECT DISTINCT status FROM temporal_checks")
        tc_statuses = {r["status"] for r in rows if r["status"]}
        for candidate in ("explanation", "root_cause", "analysis", "notes", "details", "description"):
            if candidate in tc_cols: tc_context_col = candidate; break

    pr_timestamp_col = None
    for candidate in ("created_at", "started_at", "timestamp"):
        if candidate in pr_cols: pr_timestamp_col = candidate; break

    policy_table = "ai_policies" if _table_exists(conn, "ai_policies") else "governance_policies"
    policy_sources = set()
    if _table_exists(conn, policy_table):
        rows = _all(conn, f"SELECT DISTINCT source FROM {policy_table}")
        policy_sources = {r["source"] for r in rows if r["source"]}

    return {
        "pr_has_started_at": "started_at" in pr_cols, "pr_has_completed_at": "completed_at" in pr_cols,
        "pr_has_duration_ms": "duration_ms" in pr_cols, "pr_has_ai_summary": "ai_summary" in pr_cols,
        "pr_has_dataset_id": "dataset_id" in pr_cols, "pr_completed_filter": pr_completed_filter,
        "pr_failed_filter": pr_failed_filter, "pr_timestamp_col": pr_timestamp_col, "pr_statuses": pr_statuses,
        "tc_has_explanation": tc_context_col is not None, "tc_context_col": tc_context_col,
        "tc_has_dataset_id": "dataset_id" in tc_cols, "tc_statuses": tc_statuses,
        "cp_has_ai_description": "ai_description" in cp_cols, "cp_has_sensitivity_label": "sensitivity_label" in cp_cols,
        "dr_has_dataset_id": "dataset_id" in dr_cols, "dr_has_run_id": any(c in dr_cols for c in ("profiling_run_id","run_id")),
        "severity_dist": severity_dist, "drift_has_severity": drift_has_severity, "drift_magnitude_col": drift_magnitude_col,
        "dq_active_filter": dq_active_filter, "dq_statuses": dq_statuses,
        "policy_table": policy_table, "policy_sources": policy_sources,
    }

def _duration_stats(conn, schema, dataset_id=None) -> dict:
    pr = schema
    ds_filter = ""
    params: list = []
    if dataset_id and pr["pr_has_dataset_id"]: ds_filter = "AND dataset_id = ?"; params.append(dataset_id)
    comp_filter = pr["pr_completed_filter"]
    if pr["pr_has_duration_ms"]:
        row = _one(conn, f"SELECT AVG(duration_ms), COUNT(*) FROM profiling_runs WHERE {comp_filter} AND duration_ms IS NOT NULL {ds_filter}", params)
        if row and row[1]: return {"avg_ms": round(row[0] or 0), "avg_s": round((row[0] or 0)/1000, 2), "samples": row[1]}
    if pr["pr_has_started_at"] and pr["pr_has_completed_at"]:
        rows = _all(conn, f"SELECT started_at, completed_at FROM profiling_runs WHERE {comp_filter} AND started_at IS NOT NULL AND completed_at IS NOT NULL {ds_filter}", params)
        if rows:
            durations_ms = []
            for r in rows:
                try:
                    s = datetime.fromisoformat(r["started_at"].replace("Z", "+00:00"))
                    c = datetime.fromisoformat(r["completed_at"].replace("Z", "+00:00"))
                    durations_ms.append((c - s).total_seconds() * 1000)
                except Exception: pass
            if durations_ms:
                avg_ms = mean(durations_ms)
                return {"avg_ms": round(avg_ms), "avg_s": round(avg_ms / 1000, 2), "samples": len(durations_ms)}
    return {"avg_ms": 0, "avg_s": 0.0, "samples": 0}

def _tab_global_ai_llm(conn, schema, dataset_id) -> dict:
    ds_pr = "AND dataset_id = ?" if (dataset_id and schema["pr_has_dataset_id"]) else ""
    ds_cp = "AND dataset_id = ?" if dataset_id else ""
    pr_p = [dataset_id] if ds_pr else []
    cp_p = [dataset_id] if ds_cp else []
    comp_filter = schema["pr_completed_filter"]
    if schema["pr_has_ai_summary"]:
        row = _one(conn, f"SELECT COUNT(CASE WHEN ai_summary IS NULL OR TRIM(ai_summary)='' THEN 1 END) as no_s, COUNT(*) as total FROM profiling_runs WHERE {comp_filter} {ds_pr}", pr_p)
        no_s = row["no_s"] if row else 0; comp = row["total"] if row else 0
    else:
        row = _one(conn, f"SELECT COUNT(*) FROM profiling_runs WHERE {comp_filter} {ds_pr}", pr_p)
        comp = row[0] if row else 0; no_s = comp
    hal_val = safe_pct(no_s, comp); hal_status = _status(hal_val, healthy_le=10, critical_gt=50)
    dur = _duration_stats(conn, schema, dataset_id)
    if schema["cp_has_ai_description"]:
        row_cp = _one(conn, f"SELECT COUNT(CASE WHEN ai_description IS NOT NULL AND TRIM(ai_description)!='' THEN 1 END) as has_ai, COUNT(*) as total FROM column_profiles WHERE 1=1 {ds_cp}", cp_p)
        has_ai = row_cp["has_ai"] if row_cp else 0; cp_total = row_cp["total"] if row_cp else 0
    else:
        # No ai_description column — we can't measure response relevance meaningfully
        # but at least report the real profile count
        row_cp = _one(conn, f"SELECT COUNT(*) as total FROM column_profiles WHERE 1=1 {ds_cp}", cp_p)
        cp_total = row_cp["total"] if row_cp else 0
        has_ai = 0  # genuinely unknown without the column
    rr_val = safe_pct(has_ai, cp_total); rr_status = _status(rr_val, healthy_ge=80, critical_lt=20)
    if schema["pr_has_ai_summary"]:
        row_sc = _one(conn, f"SELECT COUNT(CASE WHEN ai_summary IS NOT NULL AND LENGTH(TRIM(ai_summary))>50 THEN 1 END) as good, COUNT(CASE WHEN ai_summary IS NOT NULL AND TRIM(ai_summary)!='' THEN 1 END) as attempted FROM profiling_runs WHERE {comp_filter} {ds_pr}", pr_p)
        good = row_sc["good"] if row_sc else 0; attempted = row_sc["attempted"] if row_sc else 0
    else: good, attempted = 0, 0
    comp_val = safe_pct(good, attempted); comp_status = _status(comp_val, healthy_ge=90, critical_lt=50)
    return {"tab": "Global AI / LLM", "metrics": [
        M("hallucination_rate", "Hallucination Rate", hal_val, "%", hal_status, "completed_runs_without_ai_summary / completed_runs × 100", {"no_summary": no_s, "completed": comp}),
        M("avg_llm_latency_ms", "Avg LLM Latency", dur["avg_ms"], "ms", "healthy" if dur["avg_ms"] < 3000 else "warning", "mean(completed_at - started_at) across completed profiling runs", {"samples": dur["samples"]}),
        M("response_relevance", "Response Relevance", rr_val, "%", rr_status, "column_profiles_with_ai_content / total_column_profiles × 100", {"with_ai_content": has_ai, "total_profiles": cp_total}),
        M("llm_output_schema_compliance_rate", "LLM Output Quality", comp_val, "%", comp_status, "ai_summaries_with_length>50chars / total_ai_summaries × 100", {"substantive": good, "attempted": attempted}),
    ], "explainability": {"overview": "Global AI/LLM metrics measure whether LLM outputs are actually being produced and are substantive.", "improvement": "If Hallucination Rate is 100%, the Azure LLM key is missing or the endpoint URL format is wrong."}}

def _tab_profiling_ai(conn, schema, dataset_id) -> dict:
    ds_pr = "AND dataset_id = ?" if (dataset_id and schema["pr_has_dataset_id"]) else ""
    pr_p = [dataset_id] if ds_pr else []
    ds_dr = "AND dataset_id = ?" if (dataset_id and schema["dr_has_dataset_id"]) else ""
    dr_p = [dataset_id] if ds_dr else []
    comp_filter = schema["pr_completed_filter"]; fail_filter = schema["pr_failed_filter"]
    row = _one(conn, f"SELECT COUNT(CASE WHEN {comp_filter} THEN 1 END) as comp, COUNT(CASE WHEN {fail_filter} THEN 1 END) as fail, COUNT(*) as total FROM profiling_runs WHERE 1=1 {ds_pr}", pr_p)
    comp = row["comp"] if row else 0; fail = row["fail"] if row else 0; total = row["total"] if row else 0
    psr_val = safe_pct(comp, total); psr_status = _status(psr_val, healthy_ge=95, critical_lt=70)
    if schema["pr_has_ai_summary"]:
        row_g = _one(conn, f"SELECT COUNT(CASE WHEN ai_summary IS NOT NULL AND TRIM(ai_summary)!='' THEN 1 END) as grounded, COUNT(*) as comp FROM profiling_runs WHERE {comp_filter} {ds_pr}", pr_p)
        grounded = row_g["grounded"] if row_g else 0; comp2 = row_g["comp"] if row_g else 0
    else: grounded, comp2 = 0, comp
    mg_val = safe_pct(grounded, comp2); mg_status = _status(mg_val, healthy_ge=80, critical_lt=20)
    dr_link_col = None; dr_cols = _columns(conn, "drift_records")
    for candidate in ("profiling_run_id", "run_id", "source_run_id"):
        if candidate in dr_cols: dr_link_col = candidate; break
    if dr_link_col:
        row_d = _one(conn, f"SELECT COUNT(DISTINCT {dr_link_col}) as runs_with_drift FROM drift_records WHERE 1=1 {ds_dr}", dr_p)
        runs_with_drift = row_d["runs_with_drift"] if row_d else 0
    else:
        row_d = _one(conn, f"SELECT COUNT(*) as cnt FROM drift_records WHERE 1=1 {ds_dr}", dr_p)
        drift_cnt = row_d["cnt"] if row_d else 0; runs_with_drift = min(drift_cnt, comp) if drift_cnt > 0 else 0
    dd_val = safe_pct(runs_with_drift, comp); dd_status = _status(dd_val, healthy_ge=50, critical_lt=10)
    dur = _duration_stats(conn, schema, dataset_id)
    return {"tab": "Profiling AI", "metrics": [
        M("profiling_success_rate", "Profiling Success Rate", psr_val, "%", psr_status, "completed_runs / total_runs × 100", {"completed": comp, "failed": fail, "total": total}),
        M("metadata_grounding_score", "Metadata Grounding Score", mg_val, "%", mg_status, "runs_with_non_empty_ai_summary / completed_runs × 100", {"grounded": grounded, "completed": comp2}),
        M("drift_detection_accuracy", "Drift Detection Coverage", dd_val, "%", dd_status, "profiling_runs_that_produced_drift_records / completed_runs × 100", {"runs_with_drift": runs_with_drift, "completed": comp}),
        M("avg_profiling_runtime_s", "Avg Profiling Runtime", dur["avg_s"], "s", "healthy" if dur["avg_s"] < 120 else "warning", "mean(completed_at - started_at) in seconds across completed runs", {"samples": dur["samples"]}),
    ], "explainability": {"overview": "Profiling AI metrics measure dataset profiling reliability, AI summary generation, and drift detection."}}

def _tab_dq_scores(conn, schema, dataset_id) -> dict:
    ds_qs = "AND dataset_id = ?" if dataset_id else ""; qs_p = [dataset_id] if dataset_id else []
    ds_rr = "AND dataset_id = ?" if dataset_id else ""; rr_p = [dataset_id] if dataset_id else []
    qs_cols = _columns(conn, "quality_snapshots")
    score_col = next((c for c in ("score", "health_score", "quality_score") if c in qs_cols), None)
    if score_col:
        row_qs = _one(conn, f"SELECT COUNT(CASE WHEN {score_col} BETWEEN 0 AND 100 THEN 1 END) as valid, COUNT(*) as total FROM quality_snapshots WHERE 1=1 {ds_qs}", qs_p)
        valid_snaps = row_qs["valid"] if row_qs else 0; total_snaps = row_qs["total"] if row_qs else 0
    else: valid_snaps, total_snaps = 0, 0
    hsa_val = safe_pct(valid_snaps, total_snaps); hsa_status = "neutral" if hsa_val is None else _status(hsa_val, healthy_ge=95, critical_lt=70)
    rr_cols = _columns(conn, "dq_rule_run_results")
    result_col = next((c for c in ("result", "status", "passed") if c in rr_cols), None)
    pass_val_candidates = ["passed", "pass", "PASSED", "success"]
    if result_col:
        passed_cond = " OR ".join(f"LOWER({result_col})='{v.lower()}'" for v in pass_val_candidates)
        row_rc = _one(conn, f"SELECT COUNT(CASE WHEN {passed_cond} THEN 1 END) as passed, COUNT(*) as total FROM dq_rule_run_results WHERE 1=1 {ds_rr}", rr_p)
        passed = row_rc["passed"] if row_rc else 0; rr_total = row_rc["total"] if row_rc else 0
    else: passed, rr_total = 0, 0
    rca_val = safe_pct(passed, rr_total); rca_status = _status(rca_val, healthy_ge=90, critical_lt=60)
    scores_rows = _all(conn, f"SELECT {score_col} FROM quality_snapshots WHERE {score_col} IS NOT NULL {ds_qs}" if score_col else f"SELECT NULL FROM quality_snapshots WHERE 1=0", qs_p)
    scores = [r[0] for r in scores_rows if r[0] is not None]
    ahs_val = round(mean(scores), 2) if scores else None; ahs_status = _status(ahs_val, healthy_ge=80, critical_lt=50)
    if score_col and scores:
        recent_rows = _all(conn, f"SELECT {score_col} FROM quality_snapshots WHERE {score_col} IS NOT NULL {ds_qs} ORDER BY rowid DESC LIMIT 5", qs_p)
        recent = [r[0] for r in recent_rows if r[0] is not None]
        velocity = round(recent[0] - recent[-1], 4) if len(recent) >= 2 else None
    else:
        recent = []
        velocity = None
    hdv_status = ("neutral" if velocity is None else "healthy" if velocity >= 0 else "warning" if velocity > -10 else "critical")
    return {"tab": "DQ Scores", "metrics": [
        M("health_score_accuracy", "Health Score Accuracy", hsa_val, "%", hsa_status, "quality_snapshots WHERE score BETWEEN 0 AND 100 / total_snapshots × 100", {"valid": valid_snaps, "total": total_snaps}),
        M("rule_compliance_accuracy", "Rule Pass Rate", rca_val, "%", rca_status, "passed_rule_results / total_rule_results × 100", {"passed": passed, "total": rr_total}),
        M("avg_health_score", "Avg Dataset Health Score", ahs_val, "%", ahs_status, f"mean({score_col or 'score'}) from quality_snapshots", {"snapshots": len(scores)}),
        M("health_degradation_velocity", "Health Degradation Velocity", velocity, "pts", hdv_status, "most_recent_score - oldest_score in last 5 snapshots", {"window": len(recent) if score_col and scores else 0}),
    ], "explainability": {"overview": "DQ Scores use quality_snapshots and dq_rule_run_results for health and compliance tracking."}}

def _tab_dq_rules(conn, schema, dataset_id) -> dict:
    ds_r = "AND dataset_id = ?" if dataset_id else ""; r_p = [dataset_id] if dataset_id else []
    active_filter = schema["dq_active_filter"]
    row_rules = _one(conn, f"SELECT COUNT(CASE WHEN {active_filter} THEN 1 END) as active, COUNT(*) as total FROM dq_rules WHERE 1=1 {ds_r}", r_p)
    active_rules = row_rules["active"] if row_rules else 0; total_rules = row_rules["total"] if row_rules else 0
    rr_rule_col = "rule_id" if "rule_id" in _columns(conn, "dq_rule_runs") else None
    if rr_rule_col:
        row_exec = _one(conn, "SELECT COUNT(DISTINCT rule_id) as executed FROM dq_rule_runs")
        executed = row_exec["executed"] if row_exec else 0
    else: executed = 0
    resr_val = safe_pct(executed, active_rules); resr_status = _status(resr_val, healthy_ge=80, critical_lt=30)
    ai_sources = schema["policy_sources"]; ai_source_vals = ai_sources or {"llm", "ai", "gpt", "openai"}
    in_clause = _in_clause("source", ai_source_vals)
    row_ai = _one(conn, f"SELECT COUNT(CASE WHEN {active_filter} THEN 1 END) as accepted, COUNT(*) as suggested FROM dq_rules WHERE {in_clause} {ds_r}", r_p)
    ai_accepted = row_ai["accepted"] if row_ai else 0; ai_suggested = row_ai["suggested"] if row_ai else 0
    rrar_val = safe_pct(ai_accepted, ai_suggested); rrar_status = _status(rrar_val, healthy_ge=70, critical_lt=20)
    if ai_suggested > 0 and rr_rule_col:
        row_never = _one(conn, f"SELECT COUNT(*) as never_run FROM dq_rules r WHERE {in_clause} AND NOT EXISTS (SELECT 1 FROM dq_rule_runs rr WHERE rr.rule_id = r.id) {ds_r}", r_p)
        never_run = row_never["never_run"] if row_never else 0
    else: never_run = 0
    hrr_val = safe_pct(never_run, ai_suggested); hrr_status = _status(hrr_val, healthy_le=10, critical_gt=50) if hrr_val is not None else "neutral"
    total_datasets = _scalar(conn, "SELECT COUNT(*) FROM datasets", default=0)
    row_cov = _one(conn, f"SELECT COUNT(DISTINCT dataset_id) as covered FROM dq_rules WHERE {active_filter}")
    datasets_with_rules = row_cov["covered"] if row_cov else 0
    rcr_val = safe_pct(datasets_with_rules, total_datasets); rcr_status = _status(rcr_val, healthy_ge=80, critical_lt=30)
    return {"tab": "DQ Rules", "metrics": [
        M("rule_execution_success_rate", "Rule Execution Rate", resr_val, "%", resr_status, "active_rules_with_at_least_one_run / total_active_rules × 100", {"executed": executed, "active": active_rules, "total": total_rules}),
        M("rule_recommendation_acceptance_rate", "AI Rule Acceptance Rate", rrar_val, "%", rrar_status, "active_ai_suggested_rules / total_ai_suggested_rules × 100", {"accepted": ai_accepted, "suggested": ai_suggested}),
        M("hallucinated_rule_rate", "Hallucinated Rule Rate", hrr_val, "%", hrr_status, "ai_rules_with_zero_run_history / total_ai_rules × 100", {"never_run": never_run, "ai_rules": ai_suggested}),
        M("rule_coverage_rate", "Dataset Rule Coverage", rcr_val, "%", rcr_status, "datasets_with_at_least_one_active_rule / total_datasets × 100", {"covered": datasets_with_rules, "total_datasets": total_datasets}),
    ], "explainability": {"overview": "DQ Rules tracks rule execution rates, AI-suggested rule adoption, and dataset coverage."}}

def _tab_monitoring_trends(conn, schema, dataset_id) -> dict:
    ds_pr = "AND dataset_id = ?" if (dataset_id and schema["pr_has_dataset_id"]) else ""
    pr_p = [dataset_id] if ds_pr else []
    ds_dr = "AND dataset_id = ?" if (dataset_id and schema["dr_has_dataset_id"]) else ""
    dr_p = [dataset_id] if ds_dr else []
    cutoff_7d = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
    cutoff_14d = (datetime.now(timezone.utc) - timedelta(days=14)).isoformat()
    ts_col = schema["pr_timestamp_col"]
    if ts_col:
        row_7 = _one(conn, f"SELECT COUNT(*) as cnt FROM profiling_runs WHERE {ts_col} >= ? {ds_pr}", [cutoff_7d] + pr_p)
        runs_last_7 = row_7["cnt"] if row_7 else 0; avg_per_day = round(runs_last_7 / 7, 2)
    else:
        row_total = _one(conn, f"SELECT COUNT(*) FROM profiling_runs WHERE 1=1 {ds_pr}", pr_p)
        runs_last_7 = row_total[0] if row_total else 0
        avg_per_day = round(runs_last_7 / 7, 2) if runs_last_7 > 0 else 0
    art_status = "healthy" if runs_last_7 > 0 else "critical"
    
    # FIX: Check for SIGNIFICANT drift type (from dq_scores.py) instead of MAJOR/CRITICAL
    dr_cols = _columns(conn, "drift_records")
    if schema["drift_has_severity"]:
        row_drift = _one(conn, f"SELECT COUNT(CASE WHEN LOWER(severity) IN ('medium','high','critical') THEN 1 END) as significant, COUNT(*) as total FROM drift_records WHERE 1=1 {ds_dr}", dr_p)
        significant = row_drift["significant"] if row_drift else 0; drift_total = row_drift["total"] if row_drift else 0
        ddp_val = safe_pct(significant, drift_total); ddp_unit = "%"; ddp_formula = "LOWER(severity) IN ('medium','high','critical') drift_records / total × 100"
        ddp_status = _status(ddp_val, healthy_ge=1, critical_lt=1) if drift_total > 0 else "neutral"
    elif "drift_type" in dr_cols:
        row_drift = _one(conn, f"SELECT COUNT(CASE WHEN UPPER(drift_type) IN ('SIGNIFICANT','TYPE_CHANGE') THEN 1 END) as significant, COUNT(*) as total FROM drift_records WHERE 1=1 {ds_dr}", dr_p)
        significant = row_drift["significant"] if row_drift else 0; drift_total = row_drift["total"] if row_drift else 0
        ddp_val = safe_pct(significant, drift_total); ddp_unit = "%"; ddp_formula = "UPPER(drift_type) IN ('SIGNIFICANT','TYPE_CHANGE') drift_records / total × 100"
        ddp_status = _status(ddp_val, healthy_ge=1, critical_lt=1) if drift_total > 0 else "neutral"
    elif schema["drift_magnitude_col"]:
        mag_col = schema["drift_magnitude_col"]
        row_drift = _one(conn, f"SELECT COUNT(CASE WHEN ABS({mag_col}) > 0.1 THEN 1 END) as significant, COUNT(*) as total FROM drift_records WHERE 1=1 {ds_dr}", dr_p)
        significant = row_drift["significant"] if row_drift else 0; drift_total = row_drift["total"] if row_drift else 0
        ddp_val = safe_pct(significant, drift_total); ddp_unit = "%"; ddp_formula = f"ABS({mag_col}) > 0.1 drift_records / total × 100"
        ddp_status = _status(ddp_val, healthy_ge=1, critical_lt=1) if drift_total > 0 else "neutral"
    else:
        row_drift = _one(conn, f"SELECT COUNT(*) as total FROM drift_records WHERE 1=1 {ds_dr}", dr_p)
        drift_total = row_drift["total"] if row_drift else 0
        ddp_val = float(drift_total); ddp_unit = "events"; ddp_formula = "Total drift records (no severity/type data available)"
        ddp_status = "warning" if drift_total > 100 else "healthy" if drift_total > 0 else "neutral"
        significant = drift_total

    dr_has_created = "created_at" in dr_cols
    if ts_col and _table_exists(conn, "drift_records") and dr_has_created:
        row_last7 = _one(conn, f"SELECT COUNT(*) as cnt FROM drift_records WHERE created_at >= ? {ds_dr}", [cutoff_7d] + dr_p)
        row_prev7 = _one(conn, f"SELECT COUNT(*) as cnt FROM drift_records WHERE created_at >= ? AND created_at < ? {ds_dr}", [cutoff_14d, cutoff_7d] + dr_p)
        last7 = row_last7["cnt"] if row_last7 else 0; prev7 = row_prev7["cnt"] if row_prev7 else 0
        trend = round(((last7 - prev7) / max(prev7, 1)) * 100, 1) if prev7 > 0 else None
        dvt_status = ("healthy" if trend is not None and trend < 0 else "warning" if trend is not None and trend > 50 else "neutral")
    else: last7, prev7, trend = 0, 0, None; dvt_status = "neutral"
    qs_cols = _columns(conn, "quality_snapshots")
    score_col = next((c for c in ("score", "health_score", "quality_score") if c in qs_cols), None)
    ds_qs = "AND dataset_id = ?" if dataset_id else ""; qs_p = [dataset_id] if dataset_id else []
    if score_col:
        score_rows = _all(conn, f"SELECT {score_col} FROM quality_snapshots WHERE {score_col} IS NOT NULL {ds_qs}", qs_p)
        score_vals = [r[0] for r in score_rows if r[0] is not None]
    else: score_vals = []
    volatility = round(pstdev(score_vals), 4) if len(score_vals) >= 2 else None
    vol_status = ("healthy" if volatility is not None and volatility < 5 else "warning" if volatility is not None and volatility < 15 else "critical" if volatility is not None else "neutral")
    return {"tab": "Monitoring & Trends", "metrics": [
        M("avg_runs_last_7_days", "Profiling Activity (7-day)", avg_per_day, "runs/day", art_status, "total_profiling_runs in last 7 days / 7", {"runs_last_7d": runs_last_7}),
        M("drift_detection_precision", "Drift Alert Precision" if ddp_unit == "%" else "Total Drift Events", ddp_val, ddp_unit, ddp_status, ddp_formula, {"significant": significant, "total": drift_total, "severity_distribution": schema["severity_dist"]}),
        M("drift_volume_trend", "Drift Volume Trend (7-day vs prev 7-day)", trend, "%", dvt_status, "(last_7d_drift_count - prev_7d_drift_count) / prev_7d × 100", {"last_7d": last7, "prev_7d": prev7}),
        M("forecast_error_rate", "Health Score Volatility", volatility, "pts std", vol_status, "stdev(quality_snapshots.score)", {"samples": len(score_vals)}),
    ], "explainability": {"overview": "Monitoring tracks profiling activity, drift signal quality, and health score stability over time."}}

def _tab_anomalies_ai(conn, schema, dataset_id) -> dict:
    ds_tc = "AND dataset_id = ?" if (dataset_id and schema["tc_has_dataset_id"]) else ""
    tc_p = [dataset_id] if ds_tc else []
    tc_statuses = schema["tc_statuses"]
    row_tc = _one(conn, f"SELECT COUNT(CASE WHEN LOWER(status)='open' THEN 1 END) as open_cnt, COUNT(CASE WHEN LOWER(status)='resolved' THEN 1 END) as resolved_cnt, COUNT(*) as total FROM temporal_checks WHERE 1=1 {ds_tc}", tc_p)
    open_cnt = row_tc["open_cnt"] if row_tc else 0; resolved_cnt = row_tc["resolved_cnt"] if row_tc else 0; tc_total = row_tc["total"] if row_tc else 0
    ap_val = safe_pct(open_cnt, tc_total); ap_status = _status(ap_val, healthy_le=10, critical_gt=50) if ap_val is not None else "neutral"
    ao_val = safe_pct(open_cnt, tc_total); ao_status = _status(ao_val, healthy_le=20, critical_gt=60) if ao_val is not None else "neutral"
    ctx_col = schema["tc_context_col"]
    if ctx_col:
        row_ctx = _one(conn, f"SELECT COUNT(CASE WHEN {ctx_col} IS NOT NULL AND TRIM({ctx_col})!='' THEN 1 END) as has_ctx, COUNT(*) as total FROM temporal_checks WHERE 1=1 {ds_tc}", tc_p)
        has_ctx = row_ctx["has_ctx"] if row_ctx else 0; ctx_total = row_ctx["total"] if row_ctx else 0
        cwc_val = safe_pct(has_ctx, ctx_total)
    else: has_ctx, ctx_total = 0, tc_total; cwc_val = None
    cwc_status = _status(cwc_val, healthy_ge=70, critical_lt=30) if cwc_val is not None else "neutral"
    afsr_val = safe_pct(resolved_cnt, tc_total); afsr_status = _status(afsr_val, healthy_ge=70, critical_lt=30)
    return {"tab": "Anomalies AI", "metrics": [
        M("anomaly_precision", "Temporal Check Anomaly Rate", ap_val, "%", ap_status, "temporal_checks WHERE LOWER(status)='open' / total × 100", {"open": open_cnt, "resolved": resolved_cnt, "total": tc_total}),
        M("anomaly_open_rate", "Open Anomaly Rate", ao_val, "%", ao_status, "open_temporal_checks / total_temporal_checks × 100", {"open": open_cnt, "total": tc_total}),
        M("checks_with_context", "Checks With Context", cwc_val, "%", cwc_status, f"temporal_checks WHERE {ctx_col or 'explanation'} IS NOT NULL AND LEN>0 / total × 100", {"with_context": has_ctx, "total": ctx_total, "context_column_used": ctx_col or "none found"}),
        M("auto_fix_success_rate", "Check Resolution Rate", afsr_val, "%", afsr_status, "temporal_checks WHERE LOWER(status)='resolved' / total × 100", {"resolved": resolved_cnt, "total": tc_total}),
    ], "explainability": {"overview": "Anomaly metrics use temporal_checks table. Status values are 'open' and 'resolved'."}}

def _tab_data_lineage(conn, schema, dataset_id) -> dict:
    le_cols = _columns(conn, "lineage_edges")
    ds_le = ""; le_p: list = []
    if dataset_id and ("source_dataset_id" in le_cols or "dataset_id" in le_cols):
        id_col = "source_dataset_id" if "source_dataset_id" in le_cols else "dataset_id"
        ds_le = f"AND ({id_col} = ? OR target_dataset_id = ?)"; le_p = [dataset_id, dataset_id]
    total_datasets = _scalar(conn, "SELECT COUNT(*) FROM datasets", default=0)
    id_col = "source_dataset_id" if "source_dataset_id" in le_cols else "dataset_id"
    row_lc = _one(conn, f"SELECT COUNT(DISTINCT {id_col}) as mapped, COUNT(*) as edges FROM lineage_edges WHERE 1=1 {ds_le}", le_p) if _table_exists(conn, "lineage_edges") else None
    mapped = row_lc["mapped"] if row_lc else 0; edges = row_lc["edges"] if row_lc else 0
    lc_val = safe_pct(mapped, total_datasets); lc_status = _status(lc_val, healthy_ge=80, critical_lt=1)
    status_col = "status" if "status" in le_cols else None
    if status_col:
        row_be = _one(conn, f"SELECT COUNT(CASE WHEN LOWER({status_col})='broken' THEN 1 END) as broken, COUNT(*) as total FROM lineage_edges WHERE 1=1 {ds_le}", le_p)
        broken = row_be["broken"] if row_be else 0; le_total = row_be["total"] if row_be else 0
    else: broken, le_total = 0, edges
    bec_status = "healthy" if broken == 0 else "critical"
    conf_col = "confidence" if "confidence" in le_cols else None
    if conf_col:
        row_md = _one(conn, f"SELECT COUNT(CASE WHEN {conf_col} < 0.5 THEN 1 END) as low_conf, COUNT(*) as total FROM lineage_edges WHERE 1=1 {ds_le}", le_p)
        low_conf = row_md["low_conf"] if row_md else 0; md_total = row_md["total"] if row_md else 0
    else: low_conf, md_total = 0, edges
    mdr_val = safe_pct(low_conf, md_total); mdr_status = _status(mdr_val, healthy_le=20, critical_gt=60)
    comp_filter = schema["pr_completed_filter"]; ds_pr = "AND dataset_id = ?" if dataset_id else ""
    row_prof = _one(conn, f"SELECT COUNT(DISTINCT dataset_id) as profiled FROM profiling_runs WHERE {comp_filter} {ds_pr}", [dataset_id] if dataset_id else [])
    profiled = row_prof["profiled"] if row_prof else 0
    dpr_val = safe_pct(profiled, total_datasets); dpr_status = _status(dpr_val, healthy_ge=80, critical_lt=30)
    return {"tab": "Data Lineage & Impact", "metrics": [
        M("lineage_coverage", "Lineage Coverage", lc_val, "%", lc_status, "datasets_with_lineage_edges / total_datasets × 100", {"mapped": mapped, "total": total_datasets, "edges": edges}),
        M("broken_edge_count", "Broken Lineage Edges", broken, "", bec_status, "COUNT(lineage_edges WHERE LOWER(status)='broken')", {"broken": broken, "total": le_total}),
        M("missed_dependency_rate", "Low-Confidence Dependency Rate", mdr_val, "%", mdr_status, "edges_with_confidence < 0.5 / total × 100", {"low_confidence": low_conf, "total": md_total}),
        M("datasets_profiled_rate", "Datasets Profiled", dpr_val, "%", dpr_status, "datasets_with_at_least_one_completed_profiling_run / total_datasets × 100", {"profiled": profiled, "total_datasets": total_datasets}),
    ], "explainability": {"overview": "Lineage tracks upstream/downstream dataset relationships."}}

def _tab_knowledge_graph(conn, schema, dataset_id) -> dict:
    kg_cols = _columns(conn, "knowledge_graph_edges")
    row_kg = _one(conn, "SELECT COUNT(*) as total FROM knowledge_graph_edges") if _table_exists(conn, "knowledge_graph_edges") else None
    kg_total = row_kg["total"] if row_kg else 0
    conf_col = "confidence" if "confidence" in kg_cols else None
    if conf_col and kg_total > 0:
        row_p = _one(conn, f"SELECT COUNT(CASE WHEN {conf_col} >= 0.7 THEN 1 END) as high_conf FROM knowledge_graph_edges")
        high_conf = row_p["high_conf"] if row_p else 0
    else: high_conf = 0
    rp_val = safe_pct(high_conf, kg_total); rp_status = _status(rp_val, healthy_ge=70, critical_lt=30)
    cp_distinct = _scalar(conn, "SELECT COUNT(DISTINCT column_name) FROM column_profiles", default=0)
    edge_type_col = next((c for c in ("edge_type","type","relationship_type") if c in kg_cols), None)
    if edge_type_col and kg_total > 0:
        row_col = _one(conn, f"SELECT COUNT(*) as col_edges FROM knowledge_graph_edges WHERE LOWER({edge_type_col}) LIKE '%column%'")
        col_edges = row_col["col_edges"] if row_col else 0
    else: col_edges = 0
    kgcma_val = safe_pct(col_edges, cp_distinct); kgcma_status = _status(kgcma_val, healthy_ge=50, critical_lt=10)
    if conf_col and kg_total > 0:
        row_unc = _one(conn, f"SELECT COUNT(CASE WHEN {conf_col} IS NULL THEN 1 END) as unscored FROM knowledge_graph_edges")
        unscored = row_unc["unscored"] if row_unc else 0
    else: unscored = 0
    kghrr_val = safe_pct(unscored, kg_total); kghrr_status = _status(kghrr_val, healthy_le=10, critical_gt=50)
    return {"tab": "Knowledge Graph AI", "metrics": [
        M("kg_relationship_precision", "Relationship Precision", rp_val, "%", rp_status, "KG_edges_with_confidence >= 0.7 / total_KG_edges × 100", {"high_confidence": high_conf, "total": kg_total}),
        M("kg_column_mapping_accuracy", "Column Relationship Coverage", kgcma_val, "%", kgcma_status, "column_relationship_edges / distinct_columns × 100", {"col_edges": col_edges, "distinct_cols": cp_distinct}),
        M("kg_hallucinated_relationship_rate", "Unscored Relationship Rate", kghrr_val, "%", kghrr_status, "edges_with_no_confidence_score / total × 100", {"unscored": unscored, "total": kg_total}),
    ], "explainability": {"overview": "KG metrics use knowledge_graph_edges. Graph construction hasn't been triggered yet — all 0 is expected."}}

def _tab_dq_assistant(conn, schema, dataset_id) -> dict:
    ni_cols = _columns(conn, "notification_inbox")
    gn_cols = _columns(conn, "governance_notifications")
    type_col = next((c for c in ("notification_type","type","category") if c in ni_cols), None)
    if type_col:
        row_ni = _one(conn, f"SELECT COUNT(CASE WHEN {type_col} IS NOT NULL AND TRIM({type_col})!='' THEN 1 END) as tagged, COUNT(*) as total FROM notification_inbox")
        tagged = row_ni["tagged"] if row_ni else 0; ni_total = row_ni["total"] if row_ni else 0
    else: tagged = 0; row_ni = _one(conn, "SELECT COUNT(*) FROM notification_inbox"); ni_total = row_ni[0] if row_ni else 0
    ara_val = safe_pct(tagged, ni_total); ara_status = _status(ara_val, healthy_ge=90, critical_lt=50)
    msg_col = next((c for c in ("message","content","body","text") if c in ni_cols), None)
    if msg_col:
        row_msg = _one(conn, f"SELECT COUNT(CASE WHEN {msg_col} IS NOT NULL AND TRIM({msg_col})!='' THEN 1 END) as good, COUNT(*) as total FROM notification_inbox")
        good_msgs = row_msg["good"] if row_msg else 0; msg_total = row_msg["total"] if row_msg else 0
    else: good_msgs, msg_total = 0, ni_total
    ncr_val = safe_pct(good_msgs, msg_total); ncr_status = _status(ncr_val, healthy_ge=90, critical_lt=50)
    action_col = next((c for c in ("action_taken","actioned","resolved") if c in gn_cols), None)
    row_gn = _one(conn, "SELECT COUNT(*) FROM governance_notifications") if _table_exists(conn, "governance_notifications") else None
    gn_total = row_gn[0] if row_gn else 0
    all_notif = _scalar(conn, "SELECT COUNT(*) FROM (SELECT id FROM notification_inbox UNION ALL SELECT id FROM governance_notifications)", default=gn_total)
    if action_col:
        row_act = _one(conn, f"SELECT COUNT(CASE WHEN {action_col}=1 OR LOWER({action_col})='true' THEN 1 END) as actioned FROM governance_notifications")
        actioned = row_act["actioned"] if row_act else 0; aas_val = safe_pct(actioned, gn_total)
    else: aas_val = safe_pct(gn_total, max(all_notif, 1)); actioned = gn_total
    aas_status = _status(aas_val, healthy_ge=60, critical_lt=20)
    
    # FIX: Check 'dataset' column first (TEXT, has values like 'test2.csv'), fallback to 'dataset_id' (Integer FK, often NULL)
    ds_id_col = None
    for candidate in ("dataset", "dataset_id"):
        if candidate in ni_cols:
            row_check = _one(conn, f"SELECT COUNT(*) as cnt FROM notification_inbox WHERE {candidate} IS NOT NULL AND TRIM({candidate})!=''")
            if row_check and row_check["cnt"] > 0:
                ds_id_col = candidate; break
    if not ds_id_col:
        for candidate in ("dataset", "dataset_id"):
            if candidate in ni_cols: ds_id_col = candidate; break
            
    if ds_id_col:
        row_gs = _one(conn, f"SELECT COUNT(CASE WHEN {ds_id_col} IS NOT NULL AND TRIM({ds_id_col})!='' THEN 1 END) as grounded, COUNT(*) as total FROM notification_inbox")
        grounded = row_gs["grounded"] if row_gs else 0; gs_total = row_gs["total"] if row_gs else 0
    else: grounded, gs_total = 0, ni_total
    rgs_val = safe_pct(grounded, gs_total); rgs_status = _status(rgs_val, healthy_ge=70, critical_lt=20)
    total_datasets = _scalar(conn, "SELECT COUNT(*) FROM datasets", default=1)
    avg_notif = round(ni_total / max(total_datasets, 1), 2); anpd_status = "healthy" if avg_notif >= 1 else "neutral"
    return {"tab": "DQ Assistant / AI Agent", "metrics": [
        M("agent_routing_accuracy", "Notification Routing Accuracy", ara_val, "%", ara_status, "notifications_with_non_null_type / total_inbox × 100", {"tagged": tagged, "total": ni_total}),
        M("notification_content_rate", "Notification Content Rate", ncr_val, "%", ncr_status, "notifications_with_non_empty_message / total × 100", {"good": good_msgs, "total": msg_total}),
        M("action_agent_success_rate", "Governance Action Rate", aas_val, "%", aas_status, "governance_notifications_actioned / total_governance × 100", {"actioned": actioned, "governance_total": gn_total, "all_notifications": all_notif}),
        M("retrieval_grounding_score", "Context Grounding Score", rgs_val, "%", rgs_status, f"notifications_with_{ds_id_col or 'dataset'} / total × 100", {"grounded": grounded, "total": gs_total, "dataset_column": ds_id_col or "not found"}),
        M("avg_notifications_per_dataset", "Notifications Per Dataset", avg_notif, "", anpd_status, "total_notification_inbox / total_datasets", {"total_notifications": ni_total, "total_datasets": total_datasets}),
    ], "explainability": {"overview": "Assistant metrics derived from notification_inbox and governance_notifications tables."}}

def _tab_governance(conn, schema, dataset_id) -> dict:
    policy_table = schema["policy_table"]; policy_sources = schema["policy_sources"]
    ai_candidates = policy_sources or {"llm", "ai", "gpt", "openai", "generated"}
    in_cl = _in_clause("source", ai_candidates)
    if _table_exists(conn, policy_table):
        row_pol = _one(conn, f"SELECT COUNT(CASE WHEN LOWER(status) IN ('active','accepted','enabled') THEN 1 END) as accepted, COUNT(*) as suggested FROM {policy_table} WHERE {in_cl}")
        accepted = row_pol["accepted"] if row_pol else 0; suggested = row_pol["suggested"] if row_pol else 0
        row_all = _one(conn, f"SELECT COUNT(*) FROM {policy_table}"); pol_total = row_all[0] if row_all else 0
    else: accepted, suggested, pol_total = 0, 0, 0
    par_val = safe_pct(accepted, suggested); par_status = _status(par_val, healthy_ge=70, critical_lt=20)
    ds_cp = "AND dataset_id = ?" if dataset_id else ""; cp_p = [dataset_id] if dataset_id else []
    cp_cols = _columns(conn, "column_profiles")
    sens_col = next((c for c in ("sensitivity_label","sensitivity","label") if c in cp_cols), None)
    if sens_col:
        row_ca = _one(conn, f"SELECT COUNT(CASE WHEN {sens_col} IS NOT NULL AND TRIM({sens_col})!='' THEN 1 END) as classified, COUNT(*) as total FROM column_profiles WHERE 1=1 {ds_cp}", cp_p)
        classified = row_ca["classified"] if row_ca else 0; ca_total = row_ca["total"] if row_ca else 0
    else: classified, ca_total = 0, 0
    ca_val = safe_pct(classified, ca_total); ca_status = _status(ca_val, healthy_ge=70, critical_lt=20)
    al_cols = _columns(conn, "governance_audit_log")
    if _table_exists(conn, "governance_audit_log"):
        row_al = _one(conn, "SELECT COUNT(*) as entries FROM governance_audit_log"); audit_entries = row_al["entries"] if row_al else 0
    else: audit_entries = 0
    total_rules_created = _scalar(conn, "SELECT COUNT(*) FROM dq_rules", default=0)
    total_labels = classified; total_pol_actions = accepted + (pol_total - suggested)
    expected_audit = max(total_rules_created + total_labels + total_pol_actions, 1)
    alc_val = safe_pct(audit_entries, expected_audit); alc_val = min(alc_val, 100.0) if alc_val is not None else None
    alc_status = _status(alc_val, healthy_ge=70, critical_lt=20)
    return {"tab": "Governance & Settings", "metrics": [
        M("policy_adoption_rate", "Policy Adoption Rate", par_val, "%", par_status, "active_ai_policies / total_ai_suggested × 100", {"accepted": accepted, "suggested": suggested, "total_all_policies": pol_total}),
        M("classification_accuracy", "Column Sensitivity Classification", ca_val, "%", ca_status, "columns_with_sensitivity_label / total_columns × 100", {"classified": classified, "total": ca_total, "sensitivity_col": sens_col}),
        M("audit_log_completeness", "Audit Log Completeness", alc_val, "%", alc_status, "audit_log_entries / (rules_created + labels_applied + policy_actions) × 100", {"audit_entries": audit_entries, "expected_actions": expected_audit}),
    ], "explainability": {"overview": "Governance: policy adoption, column sensitivity labeling, and audit trail completeness."}}

def _tab_system_platform(conn, schema, dataset_id) -> dict:
    comp_filter = schema["pr_completed_filter"]; fail_filter = schema["pr_failed_filter"]
    row_su = _one(conn, f"SELECT COUNT(CASE WHEN {comp_filter} THEN 1 END) as comp, COUNT(CASE WHEN {fail_filter} THEN 1 END) as fail, COUNT(*) as total FROM profiling_runs")
    comp = row_su["comp"] if row_su else 0; fail = row_su["fail"] if row_su else 0; total = row_su["total"] if row_su else 0
    su_val = safe_pct(comp, comp + fail) if (comp + fail) > 0 else None; su_status = _status(su_val, healthy_ge=95, critical_lt=70)
    ts_col = schema["pr_timestamp_col"]
    if ts_col and total >= 2:
        rows_ts = _all(conn, f"SELECT {ts_col} FROM profiling_runs WHERE {ts_col} IS NOT NULL ORDER BY {ts_col}")
        if rows_ts and len(rows_ts) >= 2:
            try:
                t_first = datetime.fromisoformat(rows_ts[0][0].replace("Z", "+00:00"))
                t_last = datetime.fromisoformat(rows_ts[-1][0].replace("Z", "+00:00"))
                elapsed_h = (t_last - t_first).total_seconds() / 3600
                throughput = round(len(rows_ts) / elapsed_h, 3) if elapsed_h > 0 else 0
            except Exception: throughput = 0
        else: throughput = 0
    else: throughput = 0
    at_status = "neutral" if throughput == 0 else ("healthy" if throughput >= 0.5 else "warning")
    dur = _duration_stats(conn, schema)
    table_counts = {}
    for tbl in ("profiling_runs", "column_profiles", "drift_records", "temporal_checks", "dq_rules", "dq_rule_runs", "lineage_edges", "knowledge_graph_edges", "notification_inbox", "governance_audit_log"):
        if _table_exists(conn, tbl): n = _scalar(conn, f"SELECT COUNT(*) FROM {tbl}", default=0); table_counts[tbl] = n
    return {"tab": "System / Platform", "metrics": [
        M("system_uptime", "System Uptime", su_val, "%", su_status, "completed_runs / (completed + failed) × 100", {"completed": comp, "failed": fail, "total": total}),
        M("api_throughput", "Processing Throughput", throughput, "runs/hr", at_status, "total_profiling_runs / elapsed_hours", {"total_runs": total}),
        M("avg_job_duration_ms", "Avg Job Duration", dur["avg_ms"], "ms", "healthy" if dur["avg_ms"] < 60000 else "warning", "mean(completed_at - started_at) in ms across completed runs", {"samples": dur["samples"]}),
    ], "explainability": {"overview": "System metrics based on profiling_runs history.", "db_table_counts": table_counts}}

def _tab_human_feedback(conn, schema, dataset_id) -> dict:
    policy_table = schema["policy_table"]
    if _table_exists(conn, policy_table):
        row_pol = _one(conn, f"SELECT COUNT(CASE WHEN LOWER(status) IN ('active','accepted') THEN 1 END) as accepted, COUNT(CASE WHEN LOWER(status) IN ('dismissed','rejected','inactive') THEN 1 END) as dismissed, COUNT(*) as total FROM {policy_table}")
        accepted = row_pol["accepted"] if row_pol else 0; dismissed = row_pol["dismissed"] if row_pol else 0; pol_total = row_pol["total"] if row_pol else 0
    else: accepted, dismissed, pol_total = 0, 0, 0
    aar_val = safe_pct(accepted, accepted + dismissed); aar_status = _status(aar_val, healthy_ge=60, critical_lt=20)
    has_created = "created_at" in _columns(conn, "governance_audit_log") if _table_exists(conn, "governance_audit_log") else False
    cutoff_30d = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    if has_created:
        row_gai = _one(conn, "SELECT COUNT(*) as actions FROM governance_audit_log WHERE created_at >= ?", [cutoff_30d])
        governance_actions_30d = row_gai["actions"] if row_gai else 0
    else:
        row_gai = _one(conn, "SELECT COUNT(*) FROM governance_audit_log") if _table_exists(conn, "governance_audit_log") else None
        governance_actions_30d = row_gai[0] if row_gai else 0
    gai_status = ("healthy" if governance_actions_30d >= 10 else "warning" if governance_actions_30d >= 3 else "critical")
    return {"tab": "Human Feedback", "metrics": [
        M("ai_acceptance_rate", "AI Suggestion Acceptance Rate", aar_val, "%", aar_status, "active_policies / (active + dismissed) × 100", {"accepted": accepted, "dismissed": dismissed, "total": pol_total}),
        M("governance_activity_index", "Governance Activity (30-day)", governance_actions_30d, "actions", gai_status, "COUNT(governance_audit_log entries in last 30 days)", {"audit_actions_30d": governance_actions_30d}),
    ], "explainability": {"overview": "Feedback uses governance policy acceptance and audit log activity as engagement signals."}}

@router.get("/api/health-metrics")
async def get_health_metrics(dataset_id: Optional[str] = Query(None)):
    generated_at = datetime.now(timezone.utc).isoformat()
    try: conn = _conn()
    except Exception as e:
        logger.error(f"DB connection failed: {e}")
        return {"generated_at": generated_at, "dataset_id": dataset_id, "db_path": DB_PATH, "error": str(e), "tabs": []}
    try:
        schema = _introspect(conn)
        logger.info(f"Schema introspected. PR statuses: {schema['pr_statuses']}")
        logger.info(f"Drift severity: {schema['severity_dist']}")
        tab_fns = [_tab_global_ai_llm, _tab_profiling_ai, _tab_dq_scores, _tab_dq_rules, _tab_monitoring_trends, _tab_anomalies_ai, _tab_data_lineage, _tab_knowledge_graph, _tab_dq_assistant, _tab_governance, _tab_system_platform, _tab_human_feedback]
        tabs = []
        for fn in tab_fns:
            try: tabs.append(fn(conn, schema, dataset_id))
            except Exception as e:
                tab_name = fn.__name__.replace("_tab_", "").replace("_", " ").title()
                logger.error(f"Tab '{tab_name}' failed: {e}", exc_info=True)
                tabs.append({"tab": tab_name, "metrics": [], "error": str(e), "explainability": {"overview": f"Tab computation failed: {e}"}})
        return {"generated_at": generated_at, "dataset_id": dataset_id, "db_path": DB_PATH,
            "schema_info": {"pr_timing_available": schema["pr_has_started_at"] and schema["pr_has_completed_at"], "tc_status_values": sorted(schema["tc_statuses"]), "drift_severity_dist": schema["severity_dist"], "policy_table": schema["policy_table"], "policy_source_values": sorted(schema["policy_sources"]), "detected_pr_completed": schema["pr_completed_filter"], "detected_dq_active": schema["dq_active_filter"]}, "tabs": tabs}
    except Exception as e:
        logger.error(f"health-metrics catastrophic failure: {e}", exc_info=True)
        return {"generated_at": generated_at, "dataset_id": dataset_id, "db_path": DB_PATH, "error": str(e), "tabs": []}
    finally: conn.close()

@router.get("/debug/schema-inspection")
async def debug_schema_inspection():
    try:
        conn = _conn(); schema = _introspect(conn)
        samples = {}
        for tbl in ["profiling_runs", "dq_rules", "drift_records", "temporal_checks", "notification_inbox"]:
            if _table_exists(conn, tbl):
                rows = _all(conn, f"SELECT * FROM {tbl} LIMIT 3")
                samples[tbl] = [{k: r[k] for k in r.keys()} for r in rows]
        conn.close()
        return {"schema": schema, "samples": samples}
    except Exception as e: return {"error": str(e)}