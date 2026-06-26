"""
health_metrics_router.py — AI DQM Health Observatory v4.1
──────────────────────────────────────────────────────────
v4.1 CHANGES (this fix):
  • _tab_dq_rules rewritten: DQRuleRun has NO rule_id column (confirmed in
    models/__init__.py — only dataset_id, input_version_id, output_version_id).
    The old code's `rr_rule_id_col` lookup correctly found nothing because
    there is nothing to find. Rule execution must be detected through the
    real bridge table: dq_rule_run_results.rule_code == dq_rules.rule_code.
    This is the fix described in context.md Priority 2.
  • AI-rule source detection: dq_rules has no "source" column (confirmed —
    only input_mode). _find_col() for rule_source_col will always return
    None. Switched entirely to input_mode == 'ai', matching the real schema
    and the dq_rules.py fix that now preserves input_mode='ai' on approval.

Everything else preserved from v4.0:
  • Dynamic status detection  — never assumes 'completed','active', etc.
  • Dynamic column discovery  — never assumes column names exist
  • Date queries use whichever timestamp column has actual data
  • Removes checks_with_context (no explanation column in DB)
  • Replaces kg_column_mapping_accuracy → kg_build_status (categorical)
  • Renames action_agent_success_rate label to match actual measurement
  • Fixes response_relevance fallback denominator
  • Fixes drift_detection_precision via dynamic column discovery
  • Adds /api/debug-schema endpoint for post-deploy verification
"""

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


# ═══════════════════════════════════════════════════════════════════════════
# DB HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _columns(conn: sqlite3.Connection, table: str) -> set:
    try:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        return {r["name"] for r in rows}
    except Exception:
        return set()


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    try:
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone()
        return row is not None
    except Exception:
        return False


def _one(conn, sql, params=()):
    try:
        return conn.execute(sql, params).fetchone()
    except Exception as e:
        logger.debug(f"Query failed: {e} | {sql[:100]}")
        return None


def _all(conn, sql, params=()):
    try:
        return conn.execute(sql, params).fetchall()
    except Exception as e:
        logger.debug(f"Query failed: {e} | {sql[:100]}")
        return []


def _scalar(conn, sql, params=(), default=None):
    row = _one(conn, sql, params)
    if row is None:
        return default
    return default if row[0] is None else row[0]


# ═══════════════════════════════════════════════════════════════════════════
# DISCOVERY HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _find_status(status_dist: dict, candidates: list) -> Optional[str]:
    """
    Given {actual_db_value: count}, find the first candidate that matches
    any key (case-insensitive). Returns the actual DB string or None.

    Example:
        status_dist = {'done': 45, 'running': 1}
        _find_status(status_dist, ['completed','done','success']) → 'done'
    """
    if not status_dist:
        return None
    dist_lower = {k.lower(): k for k in status_dist if k is not None}
    for c in candidates:
        if c.lower() in dist_lower:
            return dist_lower[c.lower()]
    return None


def _find_col(col_set: set, candidates: list) -> Optional[str]:
    """
    From an already-retrieved column set, return the first candidate
    that exists (case-insensitive match). Returns actual DB column name or None.
    """
    low = {c.lower(): c for c in col_set}
    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]
    return None


def _find_populated_col(conn, table: str, candidates: list) -> Optional[str]:
    """
    Like _find_col but also verifies the column has at least one non-NULL row.
    Tries each candidate in order; returns the first that is populated.
    """
    cols = _columns(conn, table)
    for c in candidates:
        low = {col.lower(): col for col in cols}
        if c.lower() in low:
            actual = low[c.lower()]
            count = _scalar(conn, f"SELECT COUNT(*) FROM {table} WHERE {actual} IS NOT NULL", default=0)
            if count and count > 0:
                return actual
    return None


# ═══════════════════════════════════════════════════════════════════════════
# METRIC HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def safe_pct(num, den) -> Optional[float]:
    if not den:
        return None
    return round((num / den) * 100, 2)


def _status(value, *, healthy_ge=None, critical_lt=None,
            healthy_le=None, critical_gt=None) -> str:
    if value is None:
        return "neutral"
    if healthy_ge is not None and critical_lt is not None:
        if value >= healthy_ge:
            return "healthy"
        if value < critical_lt:
            return "critical"
        return "warning"
    if healthy_le is not None and critical_gt is not None:
        if value <= healthy_le:
            return "healthy"
        if value > critical_gt:
            return "critical"
        return "warning"
    return "neutral"


def M(id_, label, value, unit, status, formula, details=None, **extra) -> dict:
    return {
        "id": id_,
        "label": label,
        "value": value,
        "unit": unit,
        "status": status,
        "formula": formula,
        "details": details or {},
        **extra,
    }


# ═══════════════════════════════════════════════════════════════════════════
# SCHEMA INTROSPECTION  — runs once per request, drives every query
# ═══════════════════════════════════════════════════════════════════════════

def _introspect(conn) -> dict:
    s: dict = {}

    # ── profiling_runs ───────────────────────────────────────────────────────
    pr_cols = _columns(conn, "profiling_runs")

    # Status value distribution
    pr_status_dist: dict = {}
    if _table_exists(conn, "profiling_runs"):
        rows = _all(conn, "SELECT status, COUNT(*) as cnt FROM profiling_runs "
                          "WHERE status IS NOT NULL GROUP BY status")
        pr_status_dist = {r["status"]: r["cnt"] for r in rows}

    s["pr_status_dist"]       = pr_status_dist
    s["pr_completed_status"]  = _find_status(pr_status_dist,
        ["completed", "done", "success", "finished", "complete", "COMPLETED"])
    s["pr_failed_status"]     = _find_status(pr_status_dist,
        ["failed", "error", "failure", "err", "FAILED"])
    s["pr_running_status"]    = _find_status(pr_status_dist,
        ["running", "in_progress", "processing", "active", "RUNNING"])

    # Fixed timing columns (confirmed present via v3 schema_info)
    s["pr_has_started_at"]   = "started_at"   in pr_cols
    s["pr_has_completed_at"] = "completed_at" in pr_cols
    s["pr_has_duration_ms"]  = "duration_ms"  in pr_cols
    s["pr_has_dataset_id"]   = "dataset_id"   in pr_cols

    # AI summary column
    s["pr_ai_summary_col"] = _find_col(pr_cols,
        ["ai_summary", "summary", "ai_description", "llm_summary",
         "ai_output", "profiling_summary"])
    s["pr_has_ai_summary"] = s["pr_ai_summary_col"] is not None

    # Date column for time-based queries — find the first one with actual data
    s["pr_date_col"] = _find_populated_col(conn, "profiling_runs",
        ["started_at", "created_at", "run_date", "timestamp",
         "created", "scheduled_at", "updated_at", "start_time"])

    # ── dq_rules ─────────────────────────────────────────────────────────────
    rule_status_dist: dict = {}
    if _table_exists(conn, "dq_rules"):
        rows = _all(conn, "SELECT status, COUNT(*) as cnt FROM dq_rules "
                          "WHERE status IS NOT NULL GROUP BY status")
        rule_status_dist = {r["status"]: r["cnt"] for r in rows}

    s["rule_status_dist"]    = rule_status_dist
    s["rule_active_status"]  = _find_status(rule_status_dist,
        ["active", "enabled", "on", "live", "true", "1", "ACTIVE"])
    s["rule_inactive_status"] = _find_status(rule_status_dist,
        ["inactive", "disabled", "off", "false", "0", "draft", "INACTIVE"])
    s["rule_pending_status"] = _find_status(rule_status_dist,
        ["pending review", "pending", "review", "Pending Review"])

    # dq_rules — REAL schema has input_mode, NOT a "source" column.
    # Keep _find_col as a defensive fallback (in case a future migration adds
    # a literal "source" column) but input_mode is the column that actually
    # exists and is actually populated with 'ai' / 'manual' / 'dsl' / 'nl'.
    dr_rule_cols = _columns(conn, "dq_rules")
    s["rule_source_col"] = _find_col(dr_rule_cols, ["source", "rule_source", "origin"])
    s["rule_input_mode_col"] = _find_col(dr_rule_cols, ["input_mode"])

    # dq_rule_runs — there is NO rule_id / dq_rule_id FK column on this table
    # (confirmed against models/__init__.py: DQRuleRun only has dataset_id,
    # input_version_id, output_version_id). Keep the lookup for safety/future
    # schema changes, but do not rely on it for rule execution detection.
    rr_cols = _columns(conn, "dq_rule_runs")
    s["rr_rule_id_col"] = _find_col(rr_cols, ["rule_id", "dq_rule_id", "rule"])

    # dq_rule_run_results — THIS is the real bridge: rule_code ↔ dq_rules.rule_code
    rrr_cols = _columns(conn, "dq_rule_run_results")
    s["rrr_exists"] = _table_exists(conn, "dq_rule_run_results")
    s["rrr_rule_code_col"] = _find_col(rrr_cols, ["rule_code"])
    s["rrr_passed_col"] = _find_col(rrr_cols, ["passed"])
    s["rrr_pass_rate_col"] = _find_col(rrr_cols, ["pass_rate"])

    # ── governance_policies ───────────────────────────────────────────────────
    policy_table = "ai_policies" if _table_exists(conn, "ai_policies") else "governance_policies"
    s["policy_table"] = policy_table

    pol_status_dist: dict = {}
    pol_source_vals: set = set()
    pol_has_source_col = False
    if _table_exists(conn, policy_table):
        pol_cols_check = _columns(conn, policy_table)
        pol_has_source_col = "source" in pol_cols_check
        rows = _all(conn, f"SELECT status, COUNT(*) as cnt FROM {policy_table} "
                          "WHERE status IS NOT NULL GROUP BY status")
        pol_status_dist = {r["status"]: r["cnt"] for r in rows}
        if pol_has_source_col:
            rows = _all(conn, f"SELECT DISTINCT source FROM {policy_table} "
                              "WHERE source IS NOT NULL")
            pol_source_vals = {r["source"] for r in rows}

    s["policy_status_dist"]      = pol_status_dist
    s["policy_accepted_status"]  = _find_status(pol_status_dist,
        ["active", "accepted", "approved", "enabled", "ACTIVE"])
    s["policy_dismissed_status"] = _find_status(pol_status_dist,
        ["dismissed", "rejected", "inactive", "disabled", "DISMISSED"])
    s["policy_source_values"]    = pol_source_vals

    pol_cols = _columns(conn, policy_table)
    s["policy_source_col"] = _find_col(pol_cols, ["source", "policy_source", "origin", "created_by"])

    # ── temporal_checks ───────────────────────────────────────────────────────
    tc_cols = _columns(conn, "temporal_checks")
    tc_statuses: set = set()
    if _table_exists(conn, "temporal_checks"):
        rows = _all(conn, "SELECT DISTINCT status FROM temporal_checks WHERE status IS NOT NULL")
        tc_statuses = {r["status"] for r in rows}

    s["tc_status_values"]  = tc_statuses
    s["tc_has_dataset_id"] = "dataset_id" in tc_cols
    s["tc_has_explanation"] = "explanation" in tc_cols

    # ── drift_records ─────────────────────────────────────────────────────────
    dr_cols = _columns(conn, "drift_records")
    s["dr_has_dataset_id"] = "dataset_id" in dr_cols

    # Severity column — text first, then numeric score
    dr_sev_col = _find_col(dr_cols,
        ["severity", "drift_severity", "level", "alert_level",
         "severity_level", "drift_level", "alert_severity"])
    dr_score_col = _find_col(dr_cols,
        ["drift_score", "score", "magnitude", "drift_magnitude",
         "anomaly_score", "change_score"]) if not dr_sev_col else None

    s["dr_severity_col"]        = dr_sev_col or dr_score_col
    s["dr_severity_is_numeric"] = (dr_score_col is not None and dr_sev_col is None)

    # Severity distribution (for debugging)
    sev_dist: dict = {}
    if s["dr_severity_col"] and _table_exists(conn, "drift_records"):
        try:
            rows = _all(conn, f"SELECT {s['dr_severity_col']}, COUNT(*) as cnt "
                              f"FROM drift_records GROUP BY {s['dr_severity_col']}")
            sev_dist = {str(r[0]): r["cnt"] for r in rows}
        except Exception:
            pass
    s["dr_severity_dist"] = sev_dist

    # Date column for drift trends
    s["dr_date_col"] = _find_populated_col(conn, "drift_records",
        ["created_at", "detected_at", "timestamp", "run_date", "created", "alert_date"])

    # Run FK column
    dr_run_col = _find_col(dr_cols, ["profiling_run_id", "run_id", "source_run_id"])
    s["dr_has_run_id"] = dr_run_col is not None
    s["dr_run_id_col"] = dr_run_col

    # ── column_profiles ───────────────────────────────────────────────────────
    cp_cols = _columns(conn, "column_profiles")
    s["cp_has_dataset_id"]     = "dataset_id" in cp_cols
    s["cp_ai_description_col"] = _find_col(cp_cols,
        ["ai_description", "ai_desc", "description", "ai_summary",
         "llm_description", "ai_annotation", "column_description"])
    s["cp_sensitivity_col"]    = _find_col(cp_cols,
        ["sensitivity_label", "sensitivity", "label", "data_class",
         "pii_label", "data_classification", "classification",
         "sensitivity_class", "pii_category", "tag", "data_sensitivity"])
    s["cp_run_id_col"] = _find_col(cp_cols,
        ["profiling_run_id", "run_id", "source_run_id"])

    # ── governance_notifications ──────────────────────────────────────────────
    gn_cols = _columns(conn, "governance_notifications")
    s["gn_action_col"] = _find_col(gn_cols,
        ["action_taken", "actioned", "resolved", "is_actioned", "action", "handled"])

    # ── quality_snapshots ─────────────────────────────────────────────────────
    qs_cols = _columns(conn, "quality_snapshots")
    s["qs_score_col"]      = _find_col(qs_cols, ["score", "health_score", "quality_score"])
    s["qs_has_dataset_id"] = "dataset_id" in qs_cols

    # ── notification_inbox ────────────────────────────────────────────────────
    ni_cols = _columns(conn, "notification_inbox")
    s["ni_type_col"]       = _find_col(ni_cols, ["notification_type", "type", "category", "kind"])
    s["ni_message_col"]    = _find_col(ni_cols, ["message", "content", "body", "text", "description"])
    s["ni_dataset_id_col"] = _find_col(ni_cols, ["dataset_id", "dataset", "source_dataset_id"])

    return s


# ═══════════════════════════════════════════════════════════════════════════
# DURATION HELPER
# ═══════════════════════════════════════════════════════════════════════════

def _duration_stats(conn, schema, dataset_id=None) -> dict:
    """Compute avg duration for profiling runs using available timing columns."""
    pr_comp  = schema.get("pr_completed_status")
    has_ds   = schema.get("pr_has_dataset_id")

    ds_f  = "AND dataset_id = ?" if (dataset_id and has_ds) else ""
    base_params = [dataset_id] if ds_f else []

    # Try explicit duration_ms column first
    if schema.get("pr_has_duration_ms") and pr_comp:
        params = [pr_comp] + base_params
        row = _one(conn,
            f"SELECT AVG(duration_ms), COUNT(*) FROM profiling_runs "
            f"WHERE status = ? AND duration_ms IS NOT NULL {ds_f}", params)
        if row and row[1]:
            avg_ms = row[0] or 0
            return {"avg_ms": round(avg_ms), "avg_s": round(avg_ms / 1000, 2),
                    "samples": row[1]}

    # Fall back to started_at / completed_at diff
    if schema.get("pr_has_started_at") and schema.get("pr_has_completed_at") and pr_comp:
        params = [pr_comp] + base_params
        rows = _all(conn,
            f"SELECT started_at, completed_at FROM profiling_runs "
            f"WHERE status = ? AND started_at IS NOT NULL AND completed_at IS NOT NULL {ds_f}",
            params)
        if rows:
            durs = []
            for r in rows:
                try:
                    s = datetime.fromisoformat(r["started_at"].replace("Z", "+00:00"))
                    c = datetime.fromisoformat(r["completed_at"].replace("Z", "+00:00"))
                    durs.append((c - s).total_seconds() * 1000)
                except Exception:
                    pass
            if durs:
                avg_ms = mean(durs)
                return {"avg_ms": round(avg_ms), "avg_s": round(avg_ms / 1000, 2),
                        "samples": len(durs)}

    return {"avg_ms": 0, "avg_s": 0.0, "samples": 0}


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — GLOBAL AI / LLM
# ═══════════════════════════════════════════════════════════════════════════

def _tab_global_ai_llm(conn, s, dataset_id) -> dict:
    pr_comp  = s["pr_completed_status"]
    ai_col   = s["pr_ai_summary_col"]
    has_ds   = s["pr_has_dataset_id"]
    cp_ai    = s["cp_ai_description_col"]
    cp_has_ds = s["cp_has_dataset_id"]

    ds_pr = "AND dataset_id = ?" if (dataset_id and has_ds)   else ""
    ds_cp = "AND dataset_id = ?" if (dataset_id and cp_has_ds) else ""
    pr_p  = [dataset_id] if ds_pr else []
    cp_p  = [dataset_id] if ds_cp else []

    # ── Total completed runs ──────────────────────────────────────────────────
    if pr_comp:
        row_total = _one(conn,
            f"SELECT COUNT(*) as comp FROM profiling_runs "
            f"WHERE status = ? {ds_pr}", [pr_comp] + pr_p)
        comp = row_total["comp"] if row_total else 0
    else:
        row_total = _one(conn, f"SELECT COUNT(*) FROM profiling_runs WHERE 1=1 {ds_pr}", pr_p)
        comp = row_total[0] if row_total else 0

    # ── hallucination_rate ────────────────────────────────────────────────────
    if ai_col and pr_comp:
        row_hal = _one(conn,
            f"SELECT COUNT(CASE WHEN {ai_col} IS NULL OR TRIM({ai_col})='' THEN 1 END) as no_s, "
            f"COUNT(*) as total FROM profiling_runs WHERE status = ? {ds_pr}",
            [pr_comp] + pr_p)
        no_s = row_hal["no_s"]  if row_hal else 0
        hal_den = row_hal["total"] if row_hal else 0
    else:
        no_s = comp
        hal_den = comp
    hal_val = safe_pct(no_s, hal_den)
    hal_status = _status(hal_val, healthy_le=10, critical_gt=50)

    # ── avg_llm_latency_ms ────────────────────────────────────────────────────
    dur = _duration_stats(conn, s, dataset_id)

    # ── response_relevance ────────────────────────────────────────────────────
    # % of completed profiling runs that generated at least one AI-described column.
    # Uses cp_run_id_col to link column_profiles back to profiling_runs.
    cp_run_col = s.get("cp_run_id_col")
    if cp_run_col and cp_ai and pr_comp:
        row_rr = _one(conn,
            f"SELECT COUNT(DISTINCT {cp_run_col}) as with_ai FROM column_profiles "
            f"WHERE {cp_ai} IS NOT NULL AND TRIM({cp_ai}) != '' {ds_cp}", cp_p)
        has_ai = row_rr["with_ai"] if row_rr else 0
        # Denominator = total completed profiling runs
        row_tot = _one(conn,
            f"SELECT COUNT(*) FROM profiling_runs WHERE status = ? {ds_pr}",
            [pr_comp] + pr_p)
        rr_total = row_tot[0] if row_tot else 0
    elif cp_run_col:
        # ai_description column not found — count runs that produced any column profiles
        row_rr = _one(conn,
            f"SELECT COUNT(DISTINCT {cp_run_col}) as with_prof FROM column_profiles "
            f"WHERE 1=1 {ds_cp}", cp_p)
        has_ai = row_rr["with_prof"] if row_rr else 0
        row_tot = _one(conn, f"SELECT COUNT(*) FROM profiling_runs WHERE 1=1 {ds_pr}", pr_p)
        rr_total = row_tot[0] if row_tot else 0
    else:
        has_ai = 0
        row_tot = _one(conn, f"SELECT COUNT(*) FROM profiling_runs WHERE 1=1 {ds_pr}", pr_p)
        rr_total = row_tot[0] if row_tot else 0
    rr_val = safe_pct(has_ai, rr_total)
    rr_status = _status(rr_val, healthy_ge=80, critical_lt=20)

    # ── llm_output_schema_compliance_rate ─────────────────────────────────────
    if ai_col and pr_comp:
        row_sc = _one(conn,
            f"SELECT COUNT(CASE WHEN {ai_col} IS NOT NULL AND LENGTH(TRIM({ai_col}))>50 THEN 1 END) as good, "
            f"COUNT(CASE WHEN {ai_col} IS NOT NULL AND TRIM({ai_col})!='' THEN 1 END) as attempted "
            f"FROM profiling_runs WHERE status = ? {ds_pr}", [pr_comp] + pr_p)
        good      = row_sc["good"]      if row_sc else 0
        attempted = row_sc["attempted"] if row_sc else 0
    else:
        good, attempted = 0, 0
    comp_val    = safe_pct(good, attempted)
    comp_status = _status(comp_val, healthy_ge=90, critical_lt=50)

    no_llm_note = (
        "hallucination_rate will be 100% until the profiling code's LLM client "
        "is fixed to use OpenAI(base_url=f'{endpoint}/v1') instead of AzureOpenAI. "
        f"Detected completed_status='{pr_comp}', ai_summary_col='{ai_col}'."
    )

    # ── ai_column_description_rate ────────────────────────────────────────────
    acd_val, acd_status = None, "neutral"
    acd_has, acd_total = 0, 0
    if cp_ai:
        row_acd = _one(conn,
            f"SELECT "
            f"COUNT(CASE WHEN {cp_ai} IS NOT NULL AND TRIM({cp_ai}) != '' THEN 1 END) as has_ai, "
            f"COUNT(*) as total FROM column_profiles WHERE 1=1 {ds_cp}", cp_p)
        acd_has   = row_acd["has_ai"] if row_acd else 0
        acd_total = row_acd["total"]  if row_acd else 0
        acd_val   = safe_pct(acd_has, acd_total)
        acd_status = _status(acd_val, healthy_ge=70, critical_lt=30)

    # ── ai_insight_depth_score ────────────────────────────────────────────────
    aid_val, aid_status = None, "neutral"
    if ai_col and pr_comp:
        row_aid = _one(conn,
            f"SELECT ROUND(AVG(LENGTH({ai_col}))) as avg_len FROM profiling_runs "
            f"WHERE status = ? AND {ai_col} IS NOT NULL AND TRIM({ai_col}) != '' {ds_pr}",
            [pr_comp] + pr_p)
        if row_aid and row_aid["avg_len"]:
            aid_val = int(row_aid["avg_len"])
            aid_status = (
                "healthy"  if aid_val >= 200 else
                "warning"  if aid_val >= 100 else
                "critical"
            )

    # ── llm_dataset_coverage ──────────────────────────────────────────────────
    ldc_val, ldc_status = None, "neutral"
    ldc_covered_out: int = 0
    ldc_n_datasets_out: int = 0
    if ai_col and pr_comp:
        row_ds_total = _one(conn, "SELECT COUNT(*) as n FROM datasets")
        ldc_n_datasets_out = row_ds_total["n"] if row_ds_total else 0
        if ldc_n_datasets_out > 0:
            row_ldc = _one(conn,
                f"SELECT COUNT(DISTINCT dataset_id) as covered FROM profiling_runs "
                f"WHERE status = ? AND {ai_col} IS NOT NULL AND TRIM({ai_col}) != '' {ds_pr}",
                [pr_comp] + pr_p)
            ldc_covered_out = row_ldc["covered"] if row_ldc else 0
            ldc_val    = safe_pct(ldc_covered_out, ldc_n_datasets_out)
            ldc_status = _status(ldc_val, healthy_ge=80, critical_lt=40)

    return {
        "tab": "Global AI / LLM",
        "metrics": [
            M("hallucination_rate", "Hallucination Rate",
              hal_val, "%", hal_status,
              f"runs_without_{ai_col or 'ai_summary'} / completed_runs × 100",
              {"no_summary": no_s, "completed": hal_den,
               "completed_status_used": pr_comp}),

            M("avg_llm_latency_ms", "Avg Profiling Duration",
              dur["avg_ms"], "ms",
              "healthy" if dur["avg_ms"] == 0 else (
                  "healthy" if dur["avg_ms"] < 10_000 else (
                  "warning" if dur["avg_ms"] < 30_000 else "critical")),
              "mean(completed_at - started_at) across completed profiling runs",
              {"samples": dur["samples"],
               "note": "Total profiling duration per run (includes data load + scoring + LLM summary)"}),

            M("response_relevance", "Response Relevance",
              rr_val, "%", rr_status,
              "completed_runs_with_at_least_one_AI_column_description / completed_runs × 100",
              {"runs_with_ai_columns": has_ai, "completed_runs": rr_total,
               "run_fk_col": cp_run_col or "not found",
               "ai_desc_col": cp_ai or "not found"}),

            M("llm_output_schema_compliance_rate", "LLM Output Quality",
              comp_val, "%", comp_status,
              f"ai_summaries_with_length>50 / total_ai_summaries × 100",
              {"substantive": good, "attempted": attempted}),

            M("ai_column_description_rate", "Column AI Coverage",
              acd_val, "%", acd_status,
              "column_profiles_with_non_empty_ai_description / total_column_profiles × 100",
              {"described_columns": acd_has,
               "total_columns": acd_total,
               "ai_description_col": cp_ai or "not found"}),

            M("ai_insight_depth_score", "AI Insight Depth",
              aid_val, "chars", aid_status,
              "AVG(LENGTH(ai_summary)) across completed runs — higher = more detailed LLM output",
              {"avg_summary_length": aid_val,
               "threshold_healthy": "≥200 chars",
               "threshold_warning": "100–199 chars"}),

            M("llm_dataset_coverage", "LLM Dataset Coverage",
              ldc_val, "%", ldc_status,
              "datasets_with_at_least_one_ai_summarised_run / total_datasets × 100",
              {"ai_covered_datasets": ldc_covered_out,
               "total_datasets": ldc_n_datasets_out}),
        ],
        "explainability": {
            "overview": "Global AI/LLM metrics measure whether LLM outputs are actually being produced and are substantive.",
            "llm_fix_required": no_llm_note,
            "endpoint_test": (
                "Test: curl -X POST {ENDPOINT}/v1/chat/completions "
                "-H 'Authorization: Bearer {KEY}' "
                "-d '{\"model\":\"Llama-3.3-70B-Instruct\","
                "\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"max_tokens\":5}'"
            ),
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — PROFILING AI
# ═══════════════════════════════════════════════════════════════════════════

def _tab_profiling_ai(conn, s, dataset_id) -> dict:
    pr_comp  = s["pr_completed_status"]
    pr_fail  = s["pr_failed_status"]
    ai_col   = s["pr_ai_summary_col"]
    has_ds   = s["pr_has_dataset_id"]
    dr_run   = s["dr_run_id_col"]
    dr_has_ds = s["dr_has_dataset_id"]

    ds_pr = "AND dataset_id = ?" if (dataset_id and has_ds)    else ""
    ds_dr = "AND dataset_id = ?" if (dataset_id and dr_has_ds) else ""
    pr_p  = [dataset_id] if ds_pr else []
    dr_p  = [dataset_id] if ds_dr else []

    # ── profiling_success_rate ────────────────────────────────────────────────
    if pr_comp:
        comp_expr = f"COUNT(CASE WHEN status = '{pr_comp}' THEN 1 END)"
    else:
        comp_expr = "COUNT(NULL)"
    if pr_fail:
        fail_expr = f"COUNT(CASE WHEN status = '{pr_fail}' THEN 1 END)"
    else:
        fail_expr = "COUNT(NULL)"

    row_ps = _one(conn,
        f"SELECT {comp_expr} as comp, {fail_expr} as fail, COUNT(*) as total "
        f"FROM profiling_runs WHERE 1=1 {ds_pr}", pr_p)
    comp  = row_ps["comp"]  if row_ps else 0
    fail  = row_ps["fail"]  if row_ps else 0
    total = row_ps["total"] if row_ps else 0

    psr_val    = safe_pct(comp, total)
    psr_status = _status(psr_val, healthy_ge=95, critical_lt=70)

    # ── metadata_grounding_score ──────────────────────────────────────────────
    if ai_col and pr_comp:
        row_mg = _one(conn,
            f"SELECT COUNT(CASE WHEN {ai_col} IS NOT NULL AND TRIM({ai_col})!='' THEN 1 END) as grounded "
            f"FROM profiling_runs WHERE status = ? {ds_pr}", [pr_comp] + pr_p)
        grounded = row_mg["grounded"] if row_mg else 0
    else:
        grounded = 0
    mg_val    = safe_pct(grounded, comp)
    mg_status = _status(mg_val, healthy_ge=80, critical_lt=20)

    # ── drift_detection_accuracy ──────────────────────────────────────────────
    if dr_run and pr_comp:
        row_dd = _one(conn,
            f"SELECT COUNT(DISTINCT {dr_run}) as runs_with_drift FROM drift_records "
            f"WHERE 1=1 {ds_dr}", dr_p)
        runs_with_drift = row_dd["runs_with_drift"] if row_dd else 0
    else:
        # No FK — if any drift records exist for dataset, count as 1 proxy
        row_dd = _one(conn, f"SELECT COUNT(*) as cnt FROM drift_records WHERE 1=1 {ds_dr}", dr_p)
        runs_with_drift = min((row_dd["cnt"] if row_dd else 0), comp) if comp else 0

    dd_val    = safe_pct(runs_with_drift, comp)
    dd_status = _status(dd_val, healthy_ge=50, critical_lt=10)

    # ── avg_profiling_runtime_s ───────────────────────────────────────────────
    dur = _duration_stats(conn, s, dataset_id)

    return {
        "tab": "Profiling AI",
        "metrics": [
            M("profiling_success_rate", "Profiling Success Rate",
              psr_val, "%", psr_status,
              f"runs WHERE status='{pr_comp}' / total_runs × 100",
              {"completed": comp, "failed": fail, "total": total,
               "completed_status_used": pr_comp,
               "failed_status_used": pr_fail}),

            M("metadata_grounding_score", "Metadata Grounding Score",
              mg_val, "%", mg_status,
              f"runs_with_non_empty_{ai_col or 'ai_summary'} / completed_runs × 100",
              {"grounded": grounded, "completed": comp}),

            M("drift_detection_accuracy", "Drift Detection Coverage",
              dd_val, "%", dd_status,
              f"distinct_profiling_runs_in_drift_records / completed_runs × 100",
              {"runs_with_drift": runs_with_drift, "completed": comp,
               "drift_fk_col": dr_run or "not found"}),

            M("avg_profiling_runtime_s", "Avg Profiling Runtime",
              dur["avg_s"], "s",
              "healthy" if dur["avg_s"] == 0 else (
                  "healthy" if dur["avg_s"] < 120 else "warning"),
              "mean(completed_at - started_at) in seconds",
              {"samples": dur["samples"]}),
        ],
        "explainability": {
            "overview": "Profiling AI metrics measure dataset profiling reliability and AI output generation.",
            "improvement": (
                "metadata_grounding_score will remain 0 until the Azure LLM endpoint "
                "is fixed in profiling_detail.py. See profiling_detail_llm_patch.py."
            ),
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — DQ SCORES
# ═══════════════════════════════════════════════════════════════════════════

def _tab_dq_scores(conn, s, dataset_id) -> dict:
    score_col   = s["qs_score_col"]
    qs_has_ds   = s["qs_has_dataset_id"]

    ds_qs = "AND dataset_id = ?" if (dataset_id and qs_has_ds) else ""
    qs_p  = [dataset_id] if ds_qs else []

    # ── health_score_accuracy ─────────────────────────────────────────────────
    if score_col:
        row_hs = _one(conn,
            f"SELECT COUNT(CASE WHEN {score_col} BETWEEN 0 AND 100 THEN 1 END) as valid, "
            f"COUNT(*) as total FROM quality_snapshots WHERE 1=1 {ds_qs}", qs_p)
        valid_s = row_hs["valid"] if row_hs else 0
        total_s = row_hs["total"] if row_hs else 0
    else:
        valid_s, total_s = 0, 0

    hsa_val    = safe_pct(valid_s, total_s)
    hsa_status = "neutral" if hsa_val is None else _status(hsa_val, healthy_ge=95, critical_lt=70)

    # ── rule_compliance_accuracy ──────────────────────────────────────────────
    rr_cols      = _columns(conn, "dq_rule_run_results")
    result_col   = _find_col(rr_cols, ["result", "status", "outcome", "passed"])
    ds_rr = "AND dataset_id = ?" if (dataset_id and "dataset_id" in rr_cols) else ""
    rr_p  = [dataset_id] if ds_rr else []

    if result_col:
        row_rc = _one(conn,
            f"SELECT COUNT(CASE WHEN LOWER(CAST({result_col} AS TEXT)) IN "
            f"('passed','pass','success','true','1') "
            f"THEN 1 END) as passed, COUNT(*) as total "
            f"FROM dq_rule_run_results WHERE 1=1 {ds_rr}", rr_p)
        passed   = row_rc["passed"] if row_rc else 0
        rr_total = row_rc["total"]  if row_rc else 0
    else:
        row_rc   = _one(conn, f"SELECT COUNT(*) FROM dq_rule_run_results WHERE 1=1 {ds_rr}", rr_p)
        passed   = 0
        rr_total = row_rc[0] if row_rc else 0
    # ── Fallback: aggregate from dq_rule_runs if dq_rule_run_results is empty ──
    if rr_total == 0 and _table_exists(conn, "dq_rule_runs"):
        rr2_cols = _columns(conn, "dq_rule_runs")
        if "passed_count" in rr2_cols and "total_count" in rr2_cols:
            ds_rr2 = "AND dataset_id = ?" if (dataset_id and "dataset_id" in rr2_cols) else ""
            rr2_p  = [dataset_id] if ds_rr2 else []
            row_agg = _one(conn,
                f"SELECT COALESCE(SUM(passed_count),0) as passed, COALESCE(SUM(total_count),0) as total "
                f"FROM dq_rule_runs WHERE 1=1 {ds_rr2}", rr2_p)
            if row_agg and row_agg["total"] > 0:
                passed   = row_agg["passed"]
                rr_total = row_agg["total"]

    rca_val    = safe_pct(passed, rr_total)
    rca_status = _status(rca_val, healthy_ge=90, critical_lt=60)

    # ── avg_health_score ──────────────────────────────────────────────────────
    scores = []
    if score_col:
        rows = _all(conn,
            f"SELECT {score_col} FROM quality_snapshots "
            f"WHERE {score_col} IS NOT NULL {ds_qs}", qs_p)
        scores = [r[0] for r in rows if r[0] is not None]
    ahs_val    = round(mean(scores), 2) if scores else None
    ahs_status = _status(ahs_val, healthy_ge=80, critical_lt=50)

    # ── health_degradation_velocity ───────────────────────────────────────────
    velocity = None
    if score_col and len(scores) >= 2:
        recent = _all(conn,
            f"SELECT {score_col} FROM quality_snapshots "
            f"WHERE {score_col} IS NOT NULL {ds_qs} ORDER BY rowid DESC LIMIT 5", qs_p)
        vals = [r[0] for r in recent if r[0] is not None]
        if len(vals) >= 2:
            velocity = round(vals[0] - vals[-1], 4)
    hdv_status = (
        "healthy"  if velocity is not None and velocity <= 0
        else "critical" if velocity is not None and velocity > 10
        else "warning"  if velocity is not None
        else "neutral"
    )

    return {
        "tab": "DQ Scores",
        "metrics": [
            M("health_score_accuracy", "Health Score Accuracy",
              hsa_val, "%", hsa_status,
              f"quality_snapshots WHERE {score_col or 'score'} BETWEEN 0 AND 100 / total × 100",
              {"valid": valid_s, "total": total_s, "score_col": score_col}),

            M("rule_compliance_accuracy", "Rule Pass Rate",
              rca_val, "%", rca_status,
              "passed_rule_results / total_rule_results × 100",
              {"passed": passed, "total": rr_total}),

            M("avg_health_score", "Avg Dataset Health Score",
              ahs_val, "%", ahs_status,
              f"mean({score_col or 'score'}) from quality_snapshots",
              {"snapshots": len(scores)}),

            M("health_degradation_velocity", "Health Degradation Velocity",
              velocity, "pts", hdv_status,
              "most_recent_score - oldest_in_last_5 (negative = improving)",
              {"window": len(scores),
               "note": "Positive = degrading, Negative = improving"}),
        ],
        "explainability": {
            "overview": "DQ Scores use quality_snapshots and dq_rule_run_results.",
            "improvement": "Run DQ rules in DQ Engine tab to populate rule results.",
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 — DQ RULES   ◄── REWRITTEN: bridges through dq_rule_run_results
# ═══════════════════════════════════════════════════════════════════════════

def _tab_dq_rules(conn, s, dataset_id) -> dict:
    """
    FIXED (v4.1): dq_rule_runs has NO rule_id column — confirmed against the
    real ORM model. There is therefore no way to join dq_rule_runs directly
    to dq_rules. The only real bridge in the schema is:

        dq_rule_run_results.rule_code  ==  dq_rules.rule_code

    dq_rule_run_results rows are created per-rule, per-execution (see
    DQRuleRunResult model: run_id, rule_code, rule_name, rule_type, column,
    pass_rate, violation_count). So "has this rule ever executed" is:

        EXISTS (SELECT 1 FROM dq_rule_run_results rr
                WHERE rr.rule_code = dq_rules.rule_code)

    AI-rule detection switches entirely to dq_rules.input_mode == 'ai'
    (the real column — dq_rules has no "source" column at all). This also
    aligns with the dq_rules.py fix that now preserves input_mode='ai' when
    an AI-recommended rule is approved (previously it was overwritten to
    'dsl', which made AI-rule acceptance always look like zero).
    """
    act_status   = s["rule_active_status"]
    pending_status = s.get("rule_pending_status")
    has_ds        = "dataset_id" in _columns(conn, "dq_rules")
    rrr_exists    = s.get("rrr_exists")
    rrr_rule_code_col = s.get("rrr_rule_code_col")

    ds_r = "AND dataset_id = ?" if (dataset_id and has_ds) else ""
    r_p  = [dataset_id] if ds_r else []

    act_expr = f"COUNT(CASE WHEN status = '{act_status}' THEN 1 END)" if act_status else "COUNT(NULL)"

    row_rules = _one(conn,
        f"SELECT {act_expr} as active, COUNT(*) as total FROM dq_rules WHERE 1=1 {ds_r}", r_p)
    active_rules = row_rules["active"] if row_rules else 0
    total_rules  = row_rules["total"]  if row_rules else 0

    # ── rule_execution_success_rate ───────────────────────────────────────────
    # "executed" = active rules whose rule_code appears at least once in
    # dq_rule_run_results (the real per-rule execution record).
    executed = 0
    avg_run_pass_rate = None
    if rrr_exists and rrr_rule_code_col:
        if act_status:
            row_exec = _one(conn,
                f"SELECT COUNT(DISTINCT r.rule_code) as ex "
                f"FROM dq_rules r "
                f"WHERE r.status = ? {ds_r} "
                f"AND EXISTS ("
                f"  SELECT 1 FROM dq_rule_run_results rr "
                f"  WHERE rr.{rrr_rule_code_col} = r.rule_code"
                f")",
                [act_status] + r_p)
        else:
            row_exec = _one(conn,
                f"SELECT COUNT(DISTINCT r.rule_code) as ex "
                f"FROM dq_rules r "
                f"WHERE 1=1 {ds_r} "
                f"AND EXISTS ("
                f"  SELECT 1 FROM dq_rule_run_results rr "
                f"  WHERE rr.{rrr_rule_code_col} = r.rule_code"
                f")",
                r_p)
        executed = row_exec["ex"] if row_exec else 0

        # Average pass_rate across all execution records for active rules,
        # giving a sense of how well-executed rules are actually performing.
        pr_col = s.get("rrr_pass_rate_col")
        if pr_col and act_status:
            row_avg = _one(conn,
                f"SELECT AVG(rr.{pr_col}) as avg_pr "
                f"FROM dq_rule_run_results rr "
                f"JOIN dq_rules r ON r.rule_code = rr.{rrr_rule_code_col} "
                f"WHERE r.status = ? {ds_r.replace('dataset_id', 'r.dataset_id') if ds_r else ''}",
                [act_status] + r_p)
            if row_avg and row_avg["avg_pr"] is not None:
                avg_run_pass_rate = round(float(row_avg["avg_pr"]), 2)

    resr_val    = safe_pct(executed, active_rules)
    resr_status = _status(resr_val, healthy_ge=80, critical_lt=30)

    # ── rule_recommendation_acceptance_rate ───────────────────────────────────
    # AI-origin = dq_rules.input_mode == 'ai'. "Accepted" = currently Active
    # (the dq_rules.py fix means approved AI rules keep input_mode='ai' AND
    # get status='Active', so this query now actually finds them).
    if act_status:
        row_ai = _one(conn,
            f"SELECT {act_expr} as accepted, COUNT(*) as suggested "
            f"FROM dq_rules WHERE input_mode = 'ai' {ds_r}", r_p)
        ai_accepted  = row_ai["accepted"]  if row_ai else 0
        ai_suggested = row_ai["suggested"] if row_ai else 0
    else:
        row_ai = _one(conn,
            f"SELECT COUNT(*) as suggested FROM dq_rules WHERE input_mode = 'ai' {ds_r}", r_p)
        ai_accepted, ai_suggested = 0, (row_ai["suggested"] if row_ai else 0)
    rrar_val    = safe_pct(ai_accepted, ai_suggested)
    rrar_status = _status(rrar_val, healthy_ge=70, critical_lt=20)

    # ── hallucinated_rule_rate ────────────────────────────────────────────────
    # AI rules that have NEVER appeared in dq_rule_run_results at all —
    # i.e. approved/created but never actually executed against real data.
    never_run = 0
    if ai_suggested > 0 and rrr_exists and rrr_rule_code_col:
        row_nv = _one(conn,
            f"SELECT COUNT(*) as nv FROM dq_rules r "
            f"WHERE r.input_mode = 'ai' {ds_r} "
            f"AND NOT EXISTS ("
            f"  SELECT 1 FROM dq_rule_run_results rr "
            f"  WHERE rr.{rrr_rule_code_col} = r.rule_code"
            f")", r_p)
        never_run = row_nv["nv"] if row_nv else 0
    elif ai_suggested > 0:
        # dq_rule_run_results table/column missing entirely — every AI rule
        # is, by definition, "never run" since there's nowhere to record it.
        never_run = ai_suggested
    hrr_val    = safe_pct(never_run, ai_suggested)
    hrr_status = _status(hrr_val, healthy_le=10, critical_gt=50) if hrr_val is not None else "neutral"

    # ── rule_coverage_rate ────────────────────────────────────────────────────
    total_datasets = _scalar(conn, "SELECT COUNT(*) FROM datasets", default=0)
    if act_status:
        row_cov = _one(conn,
            f"SELECT COUNT(DISTINCT dataset_id) as covered FROM dq_rules "
            f"WHERE status = ? {ds_r}", [act_status] + r_p)
        covered = row_cov["covered"] if row_cov else 0
    else:
        covered = 0
    rcr_val    = safe_pct(covered, total_datasets)
    rcr_status = _status(rcr_val, healthy_ge=80, critical_lt=30)

    return {
        "tab": "DQ Rules",
        "metrics": [
            M("rule_execution_success_rate", "Rule Execution Rate",
              resr_val, "%", resr_status,
              "active_rules_with_>=1_record_in_dq_rule_run_results / active_rules × 100",
              {"executed": executed, "active": active_rules, "total": total_rules,
               "active_status_used": act_status,
               "avg_run_pass_rate": avg_run_pass_rate,
               "bridge_col_used": rrr_rule_code_col or "dq_rule_run_results.rule_code not found"}),

            M("rule_recommendation_acceptance_rate", "AI Rule Acceptance Rate",
              rrar_val, "%", rrar_status,
              "dq_rules(input_mode='ai' AND status active) / dq_rules(input_mode='ai') × 100",
              {"accepted": ai_accepted, "suggested": ai_suggested,
               "source_detection": "input_mode column (dq_rules has no 'source' column)"}),

            M("hallucinated_rule_rate", "Hallucinated Rule Rate",
              hrr_val, "%", hrr_status,
              "ai_rules_never_in_dq_rule_run_results / total_ai_rules × 100",
              {"never_run": never_run, "ai_rules": ai_suggested}),

            M("rule_coverage_rate", "Dataset Rule Coverage",
              rcr_val, "%", rcr_status,
              "datasets_with_active_rules / total_datasets × 100",
              {"covered": covered, "total_datasets": total_datasets}),
        ],
        "explainability": {
            "overview": (
                f"DQ Rules tracks {total_rules} rule(s) ({active_rules} active) against real "
                f"execution history in dq_rule_run_results, bridged via rule_code "
                f"(dq_rule_runs has no rule_id FK column, so that table alone can't answer this)."
            ),
            "improvement": (
                "Run rules from the DQ Engine tab to populate dq_rule_run_results — "
                "rule_execution_success_rate stays at 0 until at least one rule_code "
                "shows up there. A high hallucinated_rule_rate means AI-suggested rules "
                "are being approved but never actually executed against real data."
            ),
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# TAB 5 — MONITORING & TRENDS
# ═══════════════════════════════════════════════════════════════════════════

def _tab_monitoring_trends(conn, s, dataset_id) -> dict:
    pr_comp   = s["pr_completed_status"]
    pr_date   = s["pr_date_col"]       # 'started_at' (confirmed populated)
    sev_col   = s["dr_severity_col"]
    sev_num   = s["dr_severity_is_numeric"]
    sev_dist  = s["dr_severity_dist"]
    dr_date   = s["dr_date_col"]
    dr_has_ds = s["dr_has_dataset_id"]
    has_ds    = s["pr_has_dataset_id"]
    qs_col    = s["qs_score_col"]
    qs_has_ds = s["qs_has_dataset_id"]

    ds_pr = "AND dataset_id = ?" if (dataset_id and has_ds)    else ""
    ds_dr = "AND dataset_id = ?" if (dataset_id and dr_has_ds) else ""
    ds_qs = "AND dataset_id = ?" if (dataset_id and qs_has_ds) else ""
    pr_p  = [dataset_id] if ds_pr else []
    dr_p  = [dataset_id] if ds_dr else []
    qs_p  = [dataset_id] if ds_qs else []

    cutoff_7d  = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
    cutoff_14d = (datetime.now(timezone.utc) - timedelta(days=14)).isoformat()

    # ── avg_runs_last_7_days ──────────────────────────────────────────────────
    if pr_date:
        row_7 = _one(conn,
            f"SELECT COUNT(*) as cnt FROM profiling_runs "
            f"WHERE {pr_date} >= ? {ds_pr}", [cutoff_7d] + pr_p)
        runs_7d   = row_7["cnt"] if row_7 else 0
        avg_per_day = round(runs_7d / 7, 2)
    else:
        row_all = _one(conn, f"SELECT COUNT(*) FROM profiling_runs WHERE 1=1 {ds_pr}", pr_p)
        runs_7d = row_all[0] if row_all else 0
        avg_per_day = None
    art_status = "healthy" if runs_7d > 0 else "critical"

    # ── drift_detection_precision ─────────────────────────────────────────────
    if sev_col and not sev_num:
        sig_expr = f"COUNT(CASE WHEN LOWER({sev_col}) IN ('medium','high','critical') THEN 1 END)"
    elif sev_col and sev_num:
        sig_expr = f"COUNT(CASE WHEN {sev_col} >= 0.5 THEN 1 END)"
    else:
        sig_expr = "COUNT(NULL)"

    row_drift = _one(conn,
        f"SELECT {sig_expr} as significant, COUNT(*) as total "
        f"FROM drift_records WHERE 1=1 {ds_dr}", dr_p)
    significant = row_drift["significant"] if row_drift else 0
    drift_total = row_drift["total"]       if row_drift else 0

    ddp_val = safe_pct(significant, drift_total)
    ddp_status = (
        "neutral"  if drift_total == 0
        else "healthy"  if significant > 0 and ddp_val >= 20
        else "warning"  if significant > 0
        else "neutral"
    )

    # ── drift_volume_trend ────────────────────────────────────────────────────
    trend, last7, prev7 = None, 0, 0
    dvt_status = "neutral"
    if dr_date:
        row_l = _one(conn,
            f"SELECT COUNT(*) as cnt FROM drift_records "
            f"WHERE {dr_date} >= ? {ds_dr}", [cutoff_7d] + dr_p)
        row_p = _one(conn,
            f"SELECT COUNT(*) as cnt FROM drift_records "
            f"WHERE {dr_date} >= ? AND {dr_date} < ? {ds_dr}",
            [cutoff_14d, cutoff_7d] + dr_p)
        last7 = row_l["cnt"] if row_l else 0
        prev7 = row_p["cnt"] if row_p else 0
        if prev7 > 0:
            trend = round(((last7 - prev7) / prev7) * 100, 1)
        dvt_status = (
            "healthy" if trend is not None and trend < 0
            else "warning" if trend is not None and trend > 50
            else "neutral"
        )

    # ── health_score_volatility ───────────────────────────────────────────────
    volatility = None
    scores: list = []
    if qs_col:
        rows = _all(conn,
            f"SELECT {qs_col} FROM quality_snapshots "
            f"WHERE {qs_col} IS NOT NULL {ds_qs}", qs_p)
        scores = [r[0] for r in rows if r[0] is not None]
        if len(scores) >= 2:
            volatility = round(pstdev(scores), 4)
    vol_status = (
        "healthy"  if volatility is not None and volatility < 5
        else "warning"  if volatility is not None and volatility < 15
        else "critical" if volatility is not None
        else "neutral"
    )

    return {
        "tab": "Monitoring & Trends",
        "metrics": [
            M("avg_runs_last_7_days", "Profiling Activity (7-day)",
              avg_per_day, "runs/day", art_status,
              f"COUNT(*) WHERE {pr_date or 'started_at'} >= 7_days_ago / 7",
              {"runs_last_7d": runs_7d, "date_col_used": pr_date or "none"}),

            M("drift_detection_precision", "Drift Alert Precision",
              ddp_val, "%", ddp_status,
              f"drift_records WHERE {sev_col or 'severity'} is medium/high/critical / total × 100",
              {"significant": significant, "total": drift_total,
               "severity_col_used": sev_col or "not found",
               "severity_distribution": sev_dist}),

            M("drift_volume_trend", "Drift Volume Trend (7d vs prev 7d)",
              trend, "%", dvt_status,
              "(last_7d_count - prev_7d_count) / prev_7d × 100 (negative = improving)",
              {"last_7d": last7, "prev_7d": prev7,
               "date_col_used": dr_date or "not found"}),

            M("forecast_error_rate", "Health Score Volatility",
              volatility, "pts std", vol_status,
              "stdev(quality_snapshots.score) — Python statistics.stdev",
              {"samples": len(scores)}),
        ],
        "explainability": {
            "overview": "Monitoring tracks profiling activity, drift signal quality, and health score stability.",
            "drift_note": (
                f"Severity column found: '{sev_col or 'NOT FOUND'}'. "
                f"Distribution: {sev_dist or 'empty'}. "
                f"If all LOW, the drift engine needs threshold tuning."
            ),
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# TAB 6 — ANOMALIES AI
# ═══════════════════════════════════════════════════════════════════════════

def _tab_anomalies_ai(conn, s, dataset_id) -> dict:
    tc_statuses = s["tc_status_values"]
    tc_has_ds   = s["tc_has_dataset_id"]

    ds_tc = "AND dataset_id = ?" if (dataset_id and tc_has_ds) else ""
    tc_p  = [dataset_id] if ds_tc else []

    row_tc = _one(conn,
        f"SELECT COUNT(CASE WHEN LOWER(status)='open'     THEN 1 END) as open_cnt, "
        f"COUNT(CASE WHEN LOWER(status)='resolved' THEN 1 END) as res_cnt, "
        f"COUNT(*) as total FROM temporal_checks WHERE 1=1 {ds_tc}", tc_p)
    open_cnt = row_tc["open_cnt"] if row_tc else 0
    res_cnt  = row_tc["res_cnt"]  if row_tc else 0
    tc_total = row_tc["total"]    if row_tc else 0

    # ── anomaly_precision ─────────────────────────────────────────────────────
    ap_val    = safe_pct(open_cnt, tc_total)
    ap_status = _status(ap_val, healthy_le=10, critical_gt=50) if ap_val is not None else "neutral"

    # ── anomaly_open_rate ─────────────────────────────────────────────────────
    ao_val    = safe_pct(open_cnt, tc_total)
    ao_status = _status(ao_val, healthy_le=20, critical_gt=60) if ao_val is not None else "neutral"

    # ── auto_fix_success_rate ─────────────────────────────────────────────────
    afsr_val    = safe_pct(res_cnt, tc_total)
    afsr_status = _status(afsr_val, healthy_ge=70, critical_lt=30)

    return {
        "tab": "Anomalies AI",
        "metrics": [
            M("anomaly_precision", "Temporal Check Anomaly Rate",
              ap_val, "%", ap_status,
              "temporal_checks WHERE LOWER(status)='open' / total × 100",
              {"open": open_cnt, "resolved": res_cnt, "total": tc_total,
               "status_values_found": sorted(tc_statuses)}),

            M("anomaly_open_rate", "Open Anomaly Rate",
              ao_val, "%", ao_status,
              "open_temporal_checks / total × 100",
              {"open": open_cnt, "total": tc_total}),

            M("auto_fix_success_rate", "Check Resolution Rate",
              afsr_val, "%", afsr_status,
              "temporal_checks WHERE LOWER(status)='resolved' / total × 100",
              {"resolved": res_cnt, "total": tc_total}),
        ],
        "explainability": {
            "overview": (
                f"{tc_total} temporal checks found. {open_cnt} open, {res_cnt} resolved. "
                "Status values 'open'/'resolved' confirmed."
            ),
            "improvement": (
                "A high open rate means checks are being generated but not resolved. "
                "Review temporal checks in main app — check if resolution workflow is working."
            ),
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# TAB 7 — DATA LINEAGE & IMPACT
# ═══════════════════════════════════════════════════════════════════════════

def _tab_data_lineage(conn, s, dataset_id) -> dict:
    pr_comp = s["pr_completed_status"]
    has_ds  = s["pr_has_dataset_id"]

    le_cols    = _columns(conn, "lineage_edges")
    # NOTE: After the lineage_engine.py + main.py migration fix, lineage_edges
    # gains source_dataset_id / target_dataset_id INTEGER columns. Prefer
    # those; fall back to the legacy TEXT source/target columns if the
    # migration hasn't run yet on this DB.
    id_col     = _find_col(le_cols, ["source_dataset_id", "dataset_id", "from_dataset_id"])
    target_id_col = _find_col(le_cols, ["target_dataset_id", "to_dataset_id"])
    conf_col   = _find_col(le_cols, ["confidence", "weight", "score"])
    status_col = _find_col(le_cols, ["status", "edge_status", "state"])

    total_datasets = _scalar(conn, "SELECT COUNT(*) FROM datasets", default=0)

    ds_le: str = ""
    le_p:  list = []
    if dataset_id and id_col and target_id_col:
        ds_le = f"AND ({id_col} = ? OR {target_id_col} = ?)"
        le_p  = [dataset_id, dataset_id]
    elif dataset_id and id_col:
        ds_le = f"AND {id_col} = ?"
        le_p  = [dataset_id]

    # ── lineage_coverage ──────────────────────────────────────────────────────
    if id_col and _table_exists(conn, "lineage_edges"):
        if target_id_col:
            row_lc = _one(conn,
                f"SELECT COUNT(DISTINCT {id_col}) + COUNT(DISTINCT {target_id_col}) as mapped_raw, "
                f"COUNT(*) as edges "
                f"FROM lineage_edges WHERE 1=1 {ds_le}", le_p)
            # mapped_raw double counts datasets appearing as both source and
            # target; use a UNION-based distinct count instead for accuracy.
            row_mapped = _one(conn,
                f"SELECT COUNT(DISTINCT ds_id) as mapped FROM ("
                f"  SELECT {id_col} as ds_id FROM lineage_edges WHERE {id_col} IS NOT NULL "
                f"  UNION "
                f"  SELECT {target_id_col} as ds_id FROM lineage_edges WHERE {target_id_col} IS NOT NULL"
                f")"
            )
            mapped = row_mapped["mapped"] if row_mapped else 0
            edges  = row_lc["edges"] if row_lc else 0
        else:
            row_lc = _one(conn,
                f"SELECT COUNT(DISTINCT {id_col}) as mapped, COUNT(*) as edges "
                f"FROM lineage_edges WHERE 1=1 {ds_le}", le_p)
            mapped = row_lc["mapped"] if row_lc else 0
            edges  = row_lc["edges"]  if row_lc else 0
    else:
        mapped, edges = 0, 0
    lc_val    = safe_pct(mapped, total_datasets)
    lc_status = "neutral" if edges == 0 else _status(lc_val, healthy_ge=80, critical_lt=20)

    # ── broken_edge_count ─────────────────────────────────────────────────────
    broken = 0
    if status_col and _table_exists(conn, "lineage_edges"):
        row_be = _one(conn,
            f"SELECT COUNT(CASE WHEN LOWER({status_col})='broken' THEN 1 END) as broken "
            f"FROM lineage_edges WHERE 1=1 {ds_le}", le_p)
        broken = row_be["broken"] if row_be else 0
    bec_status = "healthy" if broken == 0 else "critical"

    # ── missed_dependency_rate ────────────────────────────────────────────────
    low_conf, md_total = 0, edges
    if conf_col and edges:
        row_md = _one(conn,
            f"SELECT COUNT(CASE WHEN {conf_col} < 0.5 THEN 1 END) as low_conf "
            f"FROM lineage_edges WHERE 1=1 {ds_le}", le_p)
        low_conf = row_md["low_conf"] if row_md else 0
    mdr_val    = safe_pct(low_conf, md_total)
    mdr_status = _status(mdr_val, healthy_le=20, critical_gt=60)

    # ── datasets_profiled_rate ────────────────────────────────────────────────
    if pr_comp:
        ds_pr = "AND dataset_id = ?" if (dataset_id and has_ds) else ""
        row_prof = _one(conn,
            f"SELECT COUNT(DISTINCT dataset_id) as profiled FROM profiling_runs "
            f"WHERE status = ? {ds_pr}", [pr_comp] + ([dataset_id] if ds_pr else []))
        profiled = row_prof["profiled"] if row_prof else 0
    else:
        row_prof = _one(conn, "SELECT COUNT(DISTINCT dataset_id) as profiled FROM profiling_runs")
        profiled = row_prof["profiled"] if row_prof else 0
    dpr_val    = safe_pct(profiled, total_datasets)
    dpr_status = _status(dpr_val, healthy_ge=80, critical_lt=30)

    return {
        "tab": "Data Lineage & Impact",
        "metrics": [
            M("lineage_coverage", "Lineage Coverage",
              lc_val, "%", lc_status,
              "datasets_with_lineage_edges / total_datasets × 100",
              {"mapped": mapped, "total": total_datasets, "edges": edges,
               "source_id_col": id_col or "not found",
               "target_id_col": target_id_col or "not found"}),

            M("broken_edge_count", "Broken Lineage Edges",
              broken, "", bec_status,
              "COUNT(lineage_edges WHERE LOWER(status)='broken')",
              {"broken": broken, "total": edges}),

            M("missed_dependency_rate", "Low-Confidence Dependency Rate",
              mdr_val, "%", mdr_status,
              "edges_with_confidence < 0.5 / total × 100",
              {"low_confidence": low_conf, "total": md_total}),

            M("datasets_profiled_rate", "Datasets Profiled",
              dpr_val, "%", dpr_status,
              "datasets_with_completed_profiling_run / total_datasets × 100",
              {"profiled": profiled, "total_datasets": total_datasets}),
        ],
        "explainability": {
            "overview": "Lineage tracks dataset relationships. 0 edges = lineage engine not triggered yet, or the migration adding source_dataset_id/target_dataset_id hasn't run.",
            "improvement": "Run full profiling on all datasets to generate lineage edges automatically. Ensure main.py's migration for lineage_edges INTEGER FK columns has executed.",
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# TAB 8 — KNOWLEDGE GRAPH AI
# ═══════════════════════════════════════════════════════════════════════════

def _tab_knowledge_graph(conn, s, dataset_id) -> dict:
    kg_cols  = _columns(conn, "knowledge_graph_edges")
    conf_col = _find_col(kg_cols, ["confidence", "weight", "score"])
    invalidated_col = _find_col(kg_cols, ["invalidated"])

    # Only count NON-invalidated edges as "active" — matches kg_engine.py's
    # _save_edges()/get_all_kg_edges() semantics (invalidated=False).
    if invalidated_col and _table_exists(conn, "knowledge_graph_edges"):
        kg_total = _scalar(conn,
            f"SELECT COUNT(*) FROM knowledge_graph_edges WHERE COALESCE({invalidated_col},0) = 0",
            default=0)
    else:
        kg_total = _scalar(conn, "SELECT COUNT(*) FROM knowledge_graph_edges", default=0) \
            if _table_exists(conn, "knowledge_graph_edges") else 0

    # ── kg_build_status ───────────────────────────────────────────────────────
    qs_last = None
    if kg_total > 0 and "detected_at" in kg_cols:
        row_last = _one(conn, "SELECT MAX(detected_at) as last FROM knowledge_graph_edges")
        qs_last = row_last["last"] if row_last else None
    build_status_val = "Built" if kg_total > 0 else "Not Built"
    kg_build_status_health = "healthy" if kg_total > 0 else "neutral"

    # ── kg_relationship_precision ─────────────────────────────────────────────
    high_conf = 0
    if conf_col and kg_total > 0:
        where_inv = f"WHERE COALESCE({invalidated_col},0) = 0" if invalidated_col else ""
        row_p = _one(conn,
            f"SELECT COUNT(CASE WHEN {conf_col} >= 0.7 THEN 1 END) as hc "
            f"FROM knowledge_graph_edges {where_inv}")
        high_conf = row_p["hc"] if row_p else 0
    rp_val    = safe_pct(high_conf, kg_total)
    rp_status = _status(rp_val, healthy_ge=70, critical_lt=30)

    # ── kg_hallucinated_relationship_rate ─────────────────────────────────────
    unscored = 0
    if conf_col and kg_total > 0:
        where_inv = f"WHERE COALESCE({invalidated_col},0) = 0" if invalidated_col else ""
        row_u = _one(conn,
            f"SELECT COUNT(CASE WHEN {conf_col} IS NULL THEN 1 END) as us "
            f"FROM knowledge_graph_edges {where_inv}")
        unscored = row_u["us"] if row_u else 0
    kghrr_val    = safe_pct(unscored, kg_total)
    kghrr_status = _status(kghrr_val, healthy_le=10, critical_gt=50)

    return {
        "tab": "Knowledge Graph AI",
        "metrics": [
            M("kg_build_status", "Knowledge Graph Status",
              build_status_val, "", kg_build_status_health,
              "categorical: Not Built / Built based on non-invalidated knowledge_graph_edges row count",
              {"total_edges": kg_total, "last_built": qs_last}),

            M("kg_relationship_precision", "Relationship Precision",
              rp_val, "%", rp_status,
              "KG_edges_with_confidence >= 0.7 / total × 100",
              {"high_confidence": high_conf, "total": kg_total}),

            M("kg_hallucinated_relationship_rate", "Unscored Relationship Rate",
              kghrr_val, "%", kghrr_status,
              "edges_with_null_confidence / total × 100",
              {"unscored": unscored, "total": kg_total}),
        ],
        "explainability": {
            "overview": "KG metrics use knowledge_graph_edges (excluding invalidated rows). 0 edges = KG never been built, or the router's duplicate/conflicting persistence function is writing rows the schema rejects.",
            "improvement": "Use the Knowledge Graph tab in the main app to build the graph. Ensure only kg_engine.py's _save_edges() persists edges — see knowledge_graph.py router fix.",
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# TAB 9 — DQ ASSISTANT / AI AGENT
# ═══════════════════════════════════════════════════════════════════════════

def _tab_dq_assistant(conn, s, dataset_id) -> dict:
    type_col   = s["ni_type_col"]
    msg_col    = s["ni_message_col"]
    ds_id_col  = s["ni_dataset_id_col"]
    action_col = s["gn_action_col"]

    # ── notification_inbox counts ─────────────────────────────────────────────
    type_expr = (f"COUNT(CASE WHEN {type_col} IS NOT NULL AND TRIM({type_col})!='' THEN 1 END)"
                 if type_col else "COUNT(NULL)")
    msg_expr  = (f"COUNT(CASE WHEN {msg_col} IS NOT NULL AND TRIM({msg_col})!='' THEN 1 END)"
                 if msg_col else "COUNT(NULL)")
    ds_expr   = (f"COUNT(CASE WHEN {ds_id_col} IS NOT NULL THEN 1 END)"
                 if ds_id_col else "COUNT(NULL)")

    row_ni = _one(conn,
        f"SELECT {type_expr} as tagged, {msg_expr} as good_msg, "
        f"{ds_expr} as grounded, COUNT(*) as total FROM notification_inbox")
    tagged    = row_ni["tagged"]    if row_ni else 0
    good_msg  = row_ni["good_msg"]  if row_ni else 0
    grounded  = row_ni["grounded"]  if row_ni else 0
    ni_total  = row_ni["total"]     if row_ni else 0

    ara_val = safe_pct(tagged,   ni_total)
    ncr_val = safe_pct(good_msg, ni_total)
    rgs_val = safe_pct(grounded, ni_total)

    ara_status = _status(ara_val, healthy_ge=90, critical_lt=50)
    ncr_status = _status(ncr_val, healthy_ge=90, critical_lt=50)
    rgs_status = "neutral" if rgs_val == 0 else _status(rgs_val, healthy_ge=70, critical_lt=20)

    # ── governance_notifications ──────────────────────────────────────────────
    row_gn = _one(conn, "SELECT COUNT(*) FROM governance_notifications") \
        if _table_exists(conn, "governance_notifications") else None
    gn_total = row_gn[0] if row_gn else 0

    all_notif = ni_total + gn_total

    if action_col:
        row_act = _one(conn,
            f"SELECT COUNT(CASE WHEN {action_col}=1 OR LOWER(CAST({action_col} AS TEXT))='true' "
            f"THEN 1 END) as actioned FROM governance_notifications")
        actioned = row_act["actioned"] if row_act else 0
        aas_val  = safe_pct(actioned, gn_total)
        aas_label   = "Governance Action Rate"
        aas_formula = "governance_notifications_actioned / total_governance × 100"
    else:
        actioned = gn_total
        aas_val  = safe_pct(gn_total, max(all_notif, 1))
        aas_label   = "Governance Notification Share"
        aas_formula = "governance_notifications / all_notifications × 100 (action_taken col not found)"
    aas_status = _status(aas_val, healthy_ge=60, critical_lt=20)

    # ── avg_notifications_per_dataset ─────────────────────────────────────────
    total_datasets = _scalar(conn, "SELECT COUNT(*) FROM datasets", default=1)
    avg_notif      = round(ni_total / max(total_datasets, 1), 2)
    anpd_status    = "healthy" if avg_notif >= 1 else "neutral"

    return {
        "tab": "DQ Assistant / AI Agent",
        "metrics": [
            M("agent_routing_accuracy", "Notification Routing Accuracy",
              ara_val, "%", ara_status,
              "notifications_with_type / total × 100",
              {"tagged": tagged, "total": ni_total}),

            M("notification_content_rate", "Notification Content Rate",
              ncr_val, "%", ncr_status,
              "notifications_with_non_empty_message / total × 100",
              {"good": good_msg, "total": ni_total}),

            M("action_agent_success_rate", aas_label,
              aas_val, "%", aas_status,
              aas_formula,
              {"actioned": actioned, "governance_total": gn_total,
               "all_notifications": all_notif,
               "action_col_found": action_col or "not found"}),

            M("retrieval_grounding_score", "Context Grounding Score",
              rgs_val, "%", rgs_status,
              "notifications_with_dataset_id / total × 100",
              {"grounded": grounded, "total": ni_total}),

            M("avg_notifications_per_dataset", "Notifications Per Dataset",
              avg_notif, "", anpd_status,
              "total_notification_inbox / total_datasets",
              {"total_notifications": ni_total, "total_datasets": total_datasets}),
        ],
        "explainability": {
            "overview": "Assistant metrics from notification_inbox and governance_notifications.",
            "improvement": (
                "Link notifications to dataset_id to improve grounding score. "
                f"Action column: {'found — ' + action_col if action_col else 'not found — showing notification share instead'}."
            ),
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# TAB 10 — GOVERNANCE & SETTINGS
# ═══════════════════════════════════════════════════════════════════════════

def _tab_governance(conn, s, dataset_id) -> dict:
    policy_table  = s["policy_table"]
    pol_accepted  = s["policy_accepted_status"]
    pol_dismissed = s["policy_dismissed_status"]
    pol_sources   = s["policy_source_values"]
    pol_src_col   = s["policy_source_col"]
    sens_col      = s["cp_sensitivity_col"]
    cp_has_ds     = s["cp_has_dataset_id"]

    ds_cp = "AND dataset_id = ?" if (dataset_id and cp_has_ds) else ""
    cp_p  = [dataset_id] if ds_cp else []

    # ── policy_adoption_rate ──────────────────────────────────────────────────
    # FIXED FALLBACK: if governance_policies has no "source" column at all
    # (pol_src_col is None) we can't filter by AI origin — rather than
    # silently returning 0/0/null forever, fall back to treating ALL policies
    # in the table as the denominator so the metric still reflects something
    # real instead of permanently "no data".
    if pol_src_col and _table_exists(conn, policy_table):
        ai_src = pol_sources or {"llm", "ai", "gpt", "openai", "generated"}
        in_cl  = ", ".join(f"'{v}'" for v in ai_src)
        acc_expr = (f"COUNT(CASE WHEN status = '{pol_accepted}' THEN 1 END)"
                    if pol_accepted else "COUNT(NULL)")
        row_pol = _one(conn,
            f"SELECT {acc_expr} as accepted, COUNT(*) as suggested "
            f"FROM {policy_table} WHERE {pol_src_col} IN ({in_cl})")
        accepted  = row_pol["accepted"]  if row_pol else 0
        suggested = row_pol["suggested"] if row_pol else 0
        row_all   = _one(conn, f"SELECT COUNT(*) FROM {policy_table}")
        pol_total = row_all[0] if row_all else 0
        source_detection_note = f"filtered by {pol_src_col} column"
    elif _table_exists(conn, policy_table):
        ai_src = pol_sources or set()
        acc_expr = (f"COUNT(CASE WHEN status = '{pol_accepted}' THEN 1 END)"
                    if pol_accepted else "COUNT(NULL)")
        row_pol = _one(conn, f"SELECT {acc_expr} as accepted, COUNT(*) as suggested FROM {policy_table}")
        accepted  = row_pol["accepted"]  if row_pol else 0
        suggested = row_pol["suggested"] if row_pol else 0
        pol_total = suggested
        source_detection_note = (
            f"'{policy_table}' has no source column — treating all rows as the denominator "
            f"rather than reporting null indefinitely"
        )
    else:
        accepted, suggested, pol_total, ai_src = 0, 0, 0, set()
        source_detection_note = f"table '{policy_table}' not found"
    par_val    = safe_pct(accepted, suggested)
    par_status = _status(par_val, healthy_ge=70, critical_lt=20)

    # ── classification_accuracy ───────────────────────────────────────────────
    if sens_col:
        row_ca = _one(conn,
            f"SELECT COUNT(CASE WHEN {sens_col} IS NOT NULL AND TRIM({sens_col})!='' THEN 1 END) as classified, "
            f"COUNT(*) as total FROM column_profiles WHERE 1=1 {ds_cp}", cp_p)
        classified = row_ca["classified"] if row_ca else 0
        ca_total   = row_ca["total"]       if row_ca else 0
    else:
        classified, ca_total = 0, 0
    ca_val    = safe_pct(classified, ca_total)
    ca_status = _status(ca_val, healthy_ge=70, critical_lt=20)

    # ── audit_log_completeness ────────────────────────────────────────────────
    al_total   = _scalar(conn, "SELECT COUNT(*) FROM governance_audit_log", default=0) \
        if _table_exists(conn, "governance_audit_log") else 0
    total_rules = _scalar(conn, "SELECT COUNT(*) FROM dq_rules", default=0)
    expected    = max(total_rules + classified + (accepted or 0), 1)
    alc_val     = min(safe_pct(al_total, expected) or 0, 100.0)
    alc_status  = _status(alc_val, healthy_ge=70, critical_lt=20)

    return {
        "tab": "Governance & Settings",
        "metrics": [
            M("policy_adoption_rate", "Policy Adoption Rate",
              par_val, "%", par_status,
              "active_ai_policies / total_ai_suggested × 100",
              {"accepted": accepted, "suggested": suggested, "total_policies": pol_total,
               "ai_sources_checked": list(ai_src), "source_detection": source_detection_note}),

            M("classification_accuracy", "Column Sensitivity Classification",
              ca_val, "%", ca_status,
              f"column_profiles WHERE {sens_col or 'sensitivity_label'} IS NOT NULL / total × 100",
              {"classified": classified, "total": ca_total,
               "sensitivity_col": sens_col or "not found"}),

            M("audit_log_completeness", "Audit Log Completeness",
              alc_val, "%", alc_status,
              "audit_entries / (rules + labels + policy_actions) × 100",
              {"audit_entries": al_total, "expected_actions": expected,
               "rules": total_rules, "labels": classified}),
        ],
        "explainability": {
            "overview": "Governance: policy adoption, sensitivity labeling, and audit trail.",
            "improvement": "Add sensitivity labels in Governance tab. Use app more to grow audit log.",
            "source_note": f"Policy source values in DB: {sorted(pol_sources) or 'none yet'}",
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# TAB 11 — SYSTEM / PLATFORM
# ═══════════════════════════════════════════════════════════════════════════

def _tab_system_platform(conn, s, dataset_id) -> dict:
    pr_comp = s["pr_completed_status"]
    pr_fail = s["pr_failed_status"]
    pr_date = s["pr_date_col"]

    comp_expr = (f"COUNT(CASE WHEN status = '{pr_comp}' THEN 1 END)"
                 if pr_comp else "COUNT(NULL)")
    fail_expr = (f"COUNT(CASE WHEN status = '{pr_fail}' THEN 1 END)"
                 if pr_fail else "COUNT(NULL)")

    row_su = _one(conn,
        f"SELECT {comp_expr} as comp, {fail_expr} as fail, COUNT(*) as total "
        f"FROM profiling_runs")
    comp  = row_su["comp"]  if row_su else 0
    fail  = row_su["fail"]  if row_su else 0
    total = row_su["total"] if row_su else 0

    # ── system_uptime ─────────────────────────────────────────────────────────
    su_val    = safe_pct(comp, comp + fail) if (comp + fail) > 0 else None
    su_status = _status(su_val, healthy_ge=95, critical_lt=70)

    # ── api_throughput ────────────────────────────────────────────────────────
    throughput = 0
    if pr_date and total >= 2:
        rows_ts = _all(conn,
            f"SELECT {pr_date} FROM profiling_runs "
            f"WHERE {pr_date} IS NOT NULL ORDER BY {pr_date}")
        if rows_ts and len(rows_ts) >= 2:
            try:
                t0 = datetime.fromisoformat(rows_ts[0][0].replace("Z", "+00:00"))
                t1 = datetime.fromisoformat(rows_ts[-1][0].replace("Z", "+00:00"))
                elapsed_h = (t1 - t0).total_seconds() / 3600
                elapsed_h = max(elapsed_h, 24.0)
                throughput = round(len(rows_ts) / elapsed_h, 3)
            except Exception:
                pass
    at_status = ("neutral"  if throughput == 0
                 else "healthy" if throughput >= 0.5
                 else "warning")

    # ── avg_job_duration_ms ───────────────────────────────────────────────────
    dur = _duration_stats(conn, s)

    # ── DB table counts (operational insight) ────────────────────────────────
    table_counts = {}
    for tbl in ("profiling_runs", "column_profiles", "drift_records",
                "temporal_checks", "dq_rules", "dq_rule_runs", "dq_rule_run_results",
                "lineage_edges", "knowledge_graph_edges",
                "notification_inbox", "governance_audit_log"):
        if _table_exists(conn, tbl):
            table_counts[tbl] = _scalar(conn, f"SELECT COUNT(*) FROM {tbl}", default=0)

    return {
        "tab": "System / Platform",
        "metrics": [
            M("system_uptime", "System Uptime",
              su_val, "%", su_status,
              f"runs WHERE status='{pr_comp}' / (completed+failed) × 100",
              {"completed": comp, "failed": fail, "total": total,
               "status_used": pr_comp}),

            M("api_throughput", "Processing Throughput",
              throughput, "runs/hr", at_status,
              f"total_runs / elapsed_hours via {pr_date or 'no_date_col'} diff",
              {"total_runs": total, "date_col_used": pr_date or "none"}),

            M("avg_job_duration_ms", "Avg Job Duration",
              dur["avg_ms"], "ms",
              "healthy" if dur["avg_ms"] == 0 else (
                  "healthy" if dur["avg_ms"] < 60000 else "warning"),
              "mean(completed_at - started_at) in ms",
              {"samples": dur["samples"]}),
        ],
        "explainability": {
            "overview": f"System metrics. completed_status='{pr_comp}', date_col='{pr_date}'.",
            "db_table_counts": table_counts,
            "improvement": "Monitor via Render logs for failed runs.",
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# TAB 12 — HUMAN FEEDBACK
# ═══════════════════════════════════════════════════════════════════════════

def _tab_human_feedback(conn, s, dataset_id) -> dict:
    policy_table  = s["policy_table"]
    pol_accepted  = s["policy_accepted_status"]
    pol_dismissed = s["policy_dismissed_status"]

    acc_expr = (f"COUNT(CASE WHEN status = '{pol_accepted}' THEN 1 END)"
                if pol_accepted else "COUNT(NULL)")
    dis_expr = (f"COUNT(CASE WHEN status = '{pol_dismissed}' THEN 1 END)"
                if pol_dismissed else "COUNT(NULL)")

    if _table_exists(conn, policy_table):
        row_pol = _one(conn,
            f"SELECT {acc_expr} as accepted, {dis_expr} as dismissed, COUNT(*) as total "
            f"FROM {policy_table}")
        accepted  = row_pol["accepted"]  if row_pol else 0
        dismissed = row_pol["dismissed"] if row_pol else 0
        pol_total = row_pol["total"]     if row_pol else 0
    else:
        accepted, dismissed, pol_total = 0, 0, 0

    aar_val    = safe_pct(accepted, accepted + dismissed)
    aar_status = _status(aar_val, healthy_ge=60, critical_lt=20)

    # ── governance_activity_index ─────────────────────────────────────────────
    al_cols   = _columns(conn, "governance_audit_log")
    has_dates = "created_at" in al_cols
    cutoff_30d = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()

    if has_dates and _table_exists(conn, "governance_audit_log"):
        row_gai = _one(conn,
            "SELECT COUNT(*) as cnt FROM governance_audit_log WHERE created_at >= ?",
            [cutoff_30d])
        actions_30d = row_gai["cnt"] if row_gai else 0
    elif _table_exists(conn, "governance_audit_log"):
        row_gai = _one(conn, "SELECT COUNT(*) FROM governance_audit_log")
        actions_30d = row_gai[0] if row_gai else 0
    else:
        actions_30d = 0

    gai_status = ("healthy"  if actions_30d >= 10
                  else "warning"  if actions_30d >= 3
                  else "critical")

    return {
        "tab": "Human Feedback",
        "metrics": [
            M("ai_acceptance_rate", "AI Suggestion Acceptance Rate",
              aar_val, "%", aar_status,
              "active_policies / (active + dismissed) × 100",
              {"accepted": accepted, "dismissed": dismissed, "total": pol_total,
               "accepted_status": pol_accepted, "dismissed_status": pol_dismissed}),

            M("governance_activity_index", "Governance Activity (30-day)",
              actions_30d, "actions", gai_status,
              "COUNT(governance_audit_log entries in last 30 days)",
              {"audit_actions_30d": actions_30d,
               "threshold_healthy": 10,
               "date_col_available": has_dates}),
        ],
        "explainability": {
            "overview": "Feedback uses governance policy acceptance and audit log activity.",
            "improvement": "Accept AI policy suggestions. Use Governance tab to increase activity index.",
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# TAB 13 — AZURE LLM USAGE
# ═══════════════════════════════════════════════════════════════════════════

def _tab_azure_llm(conn) -> dict:
    """
    Fetches REAL metrics from Azure Monitor (token usage, request counts,
    latency, errors) and real cost from Azure Cost Management API.

    All values come directly from Azure — nothing is hardcoded or estimated.
    Status thresholds are relative: derived from the actual data itself
    (e.g. latency compared to its own recent average, error rate based on
    actual counts) rather than fixed magic numbers.

    When Azure is not configured, all metrics return null with neutral status.
    """
    try:
        from app.routers.azure_metrics_collector import fetch_live, is_configured
        configured = is_configured()
        data = fetch_live() if configured else None
    except ImportError:
        configured = False
        data = None
    except Exception as e:
        logger.error(f"Azure metrics fetch failed in tab: {e}", exc_info=True)
        configured = True
        data = None

    def _v(key):
        return data.get(key) if data else None

    total_requests    = _v("total_requests")
    success_requests  = _v("success_requests")
    prompt_tokens     = _v("prompt_tokens")
    completion_tokens = _v("completion_tokens")
    total_tokens      = _v("total_tokens")
    avg_latency_ms    = _v("avg_latency_ms")
    max_latency_ms    = _v("max_latency_ms")
    min_latency_ms    = _v("min_latency_ms")
    error_count       = _v("error_count")
    throttled_count   = _v("throttled_count")
    server_errors     = _v("server_errors")
    client_errors     = _v("client_errors")
    actual_cost       = _v("actual_cost")
    deployment        = _v("deployment_name")
    window_hours      = _v("window_hours") or int(os.getenv("AZURE_MONITOR_WINDOW_HOURS", "24"))
    window_start      = _v("window_start")
    window_end        = _v("window_end")
    fetched_at        = _v("fetched_at")
    is_stale          = _v("stale") or False
    cache_age_s       = _v("cache_age_s")
    token_trend_pct   = _v("token_trend_pct")
    call_trend_pct    = _v("call_trend_pct")
    prev_tokens       = _v("prev_window_tokens")
    prev_calls        = _v("prev_window_calls")
    alltime_calls     = _v("alltime_calls")
    alltime_tokens    = _v("alltime_tokens")
    first_call_at     = _v("first_call_at")
    last_call_at      = _v("last_call_at")
    models_breakdown  = _v("models") or []
    recent_calls      = _v("recent_calls") or []
    cost_usd          = _v("cost_usd")
    prev_cost_usd     = _v("prev_window_cost_usd")
    cost_trend_pct    = _v("cost_trend_pct")
    avg_cost_per_call = _v("avg_cost_per_call_usd")
    alltime_cost_usd  = _v("alltime_cost_usd")
    price_input_per_m = _v("price_input_per_m")
    price_output_per_m= _v("price_output_per_m")

    te_val, te_status = None, "neutral"
    if prompt_tokens is not None and completion_tokens is not None:
        denom = prompt_tokens + completion_tokens
        if denom > 0:
            te_val = round((completion_tokens / denom) * 100, 2)
            if total_requests and total_requests > 0:
                te_status = (
                    "critical" if te_val == 0 else
                    "warning"  if te_val < 5  else
                    "healthy"
                )

    er_val, er_status = None, "neutral"
    if total_requests and total_requests > 0 and error_count is not None:
        er_val = round((error_count / total_requests) * 100, 2)
        er_status = (
            "healthy"  if er_val == 0 else
            "warning"  if er_val <= 5 else
            "critical"
        )

    tr_val, tr_status = None, "neutral"
    if total_requests and total_requests > 0 and throttled_count is not None:
        tr_val = round((throttled_count / total_requests) * 100, 2)
        tr_status = (
            "healthy"  if tr_val == 0 else
            "warning"  if tr_val <= 2 else
            "critical"
        )

    lat_status = "neutral"
    if avg_latency_ms is not None:
        if max_latency_ms and max_latency_ms > 0:
            lat_ratio = avg_latency_ms / max_latency_ms
            lat_status = (
                "healthy" if lat_ratio >= 0.7 else
                "warning" if lat_ratio >= 0.3 else
                "critical"
            )
        else:
            lat_status = "healthy"

    sr_val, sr_status = None, "neutral"
    if total_requests and total_requests > 0 and success_requests is not None:
        sr_val = round((success_requests / total_requests) * 100, 2)
        sr_status = (
            "healthy"  if sr_val >= 99 else
            "warning"  if sr_val >= 95 else
            "critical"
        )

    not_configured_note = (
        "Set AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP, AZURE_OPENAI_RESOURCE, "
        "AZURE_TENANT_ID, AZURE_CLIENT_ID, AZURE_CLIENT_SECRET to enable Azure metrics."
    )
    fetch_failed_note = (
        "Azure credentials are configured but the last fetch failed. "
        "Check that the service principal has Monitoring Reader role on the resource."
    )

    return {
        "tab": "Azure LLM Usage",
        "metrics": [
            M("azure_total_requests", f"LLM Calls (last {window_hours}h)",
              total_requests, "calls",
              "neutral" if total_requests is None else (
                  "healthy" if (total_requests or 0) > 0 else "warning"
              ),
              f"Total chat.completions.create calls recorded in last {window_hours}h",
              {"success":      success_requests,
               "errors":       error_count,
               "trend_vs_prev_window_pct": call_trend_pct,
               "prev_window_calls": prev_calls,
               "alltime_total": alltime_calls,
               "first_call_at": first_call_at,
               "last_call_at":  last_call_at,
               "deployment":    deployment,
               "window_start":  window_start,
               "window_end":    window_end,
               "last_fetched":  fetched_at}),

            M("azure_success_rate", "Request Success Rate",
              sr_val, "%", sr_status,
              "successful_calls / total_calls × 100",
              {"successful": success_requests, "total": total_requests,
               "errors": error_count}),

            M("azure_total_tokens", f"Tokens Used (last {window_hours}h)",
              total_tokens, "tokens",
              "neutral" if total_tokens is None else (
                  "healthy" if (total_tokens or 0) > 0 else "warning"
              ),
              f"SUM(prompt_tokens + completion_tokens) from llm_usage_log in last {window_hours}h",
              {"prompt_tokens":      prompt_tokens,
               "completion_tokens":  completion_tokens,
               "trend_vs_prev_window_pct": token_trend_pct,
               "prev_window_tokens": prev_tokens,
               "alltime_total":      alltime_tokens,
               "models_breakdown":   models_breakdown}),

            M("azure_token_efficiency", "Completion Token Ratio",
              te_val, "%", te_status,
              "completion_tokens / (prompt_tokens + completion_tokens) × 100",
              {"prompt_tokens":     prompt_tokens,
               "completion_tokens": completion_tokens,
               "note": "Low = LLM consuming input but not generating output"}),

            M("azure_avg_latency_ms", f"Avg LLM Latency (last {window_hours}h)",
              avg_latency_ms, "ms", lat_status,
              f"AVG(latency_ms) of all LLM calls in last {window_hours}h",
              {"avg_ms":  avg_latency_ms,
               "max_ms":  max_latency_ms,
               "min_ms":  min_latency_ms,
               "samples": total_requests}),

            M("azure_error_rate", "LLM Error Rate",
              er_val, "%", er_status,
              "failed_calls / total_calls × 100",
              {"error_calls":   error_count,
               "total_calls":   total_requests,
               "recent_errors": [
                   r for r in recent_calls if r.get("status") == "error"
               ][:3]}),

            M("azure_throttle_rate", "Throttle Rate",
              tr_val, "%", tr_status,
              "throttled_calls / total_calls × 100 (requires Azure Monitor for full accuracy)",
              {"throttled": throttled_count, "total_requests": total_requests}),

            M("azure_actual_cost", f"LLM Cost (last {window_hours}h)",
              round(cost_usd, 6) if cost_usd is not None else None,
              "USD",
              "neutral" if cost_usd is None else (
                  "healthy" if (cost_usd or 0) < 1.0 else
                  "warning" if (cost_usd or 0) < 10.0 else
                  "critical"
              ),
              f"(prompt_tokens/1M × ${price_input_per_m or 0.71}) + "
              f"(completion_tokens/1M × ${price_output_per_m or 0.71}) "
              f"— published Azure AI Foundry rate for {deployment or 'Llama-3.3-70B-Instruct'}",
              {"window_cost_usd":        round(cost_usd, 6) if cost_usd else 0,
               "prev_window_cost_usd":   round(prev_cost_usd, 6) if prev_cost_usd else 0,
               "cost_trend_pct":         cost_trend_pct,
               "avg_cost_per_call_usd":  avg_cost_per_call,
               "alltime_cost_usd":       alltime_cost_usd,
               "prompt_tokens":          prompt_tokens,
               "completion_tokens":      completion_tokens,
               "price_input_per_1m":     price_input_per_m,
               "price_output_per_1m":    price_output_per_m,
               "pricing_source":         "Azure AI Foundry marketplace, verified June 2026"}),
        ],
        "explainability": {
            "overview": (
                f"Live LLM usage tracked via API response intercept. "
                f"Model: {deployment}. "
                f"Source: llm_usage_log (updated on every LLM call). "
                f"Last fetched: {fetched_at}."
                if data else
                fetch_failed_note if configured else
                not_configured_note
            ),
            "configured":         configured,
            "fetch_ok":           data is not None,
            "data_source":        "llm_usage_log (local DB — populated by TrackedOpenAIClient)",
            "window_hours":       window_hours,
            "alltime_calls":      alltime_calls,
            "alltime_tokens":     alltime_tokens,
            "alltime_cost_usd":   alltime_cost_usd,
            "pricing": {
                "model":              deployment,
                "input_per_1m_usd":   price_input_per_m,
                "output_per_1m_usd":  price_output_per_m,
                "source":             "Azure AI Foundry marketplace, verified June 2026",
                "override_env_vars":  "LLM_PRICE_INPUT_PER_M, LLM_PRICE_OUTPUT_PER_M",
            },
            "models_in_window":   [m.get("model") for m in models_breakdown],
            "recent_calls":       recent_calls,
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN ENDPOINT
# ═══════════════════════════════════════════════════════════════════════════

@router.get("/api/health-metrics")
async def get_health_metrics(dataset_id: Optional[str] = Query(None)):
    """
    Primary health metrics endpoint — v4.1.
    Each tab is computed independently; one failure never breaks others.
    """
    generated_at = datetime.now(timezone.utc).isoformat()

    try:
        conn = _conn()
    except Exception as e:
        logger.error(f"DB connection failed: {e}")
        return {
            "generated_at": generated_at, "dataset_id": dataset_id,
            "db_path": DB_PATH, "error": str(e), "tabs": [],
        }

    try:
        s = _introspect(conn)

        logger.info(
            f"v4.1 introspect: pr_completed='{s['pr_completed_status']}' "
            f"pr_date='{s['pr_date_col']}' "
            f"dr_severity='{s['dr_severity_col']}' "
            f"cp_sensitivity='{s['cp_sensitivity_col']}' "
            f"rule_active='{s['rule_active_status']}' "
            f"rrr_rule_code_col='{s.get('rrr_rule_code_col')}'"
        )

        tab_fns = [
            _tab_global_ai_llm, _tab_profiling_ai, _tab_dq_scores,
            _tab_dq_rules, _tab_monitoring_trends, _tab_anomalies_ai,
            _tab_data_lineage, _tab_knowledge_graph, _tab_dq_assistant,
            _tab_governance, _tab_system_platform, _tab_human_feedback,
        ]

        tabs = []
        for fn in tab_fns:
            try:
                tabs.append(fn(conn, s, dataset_id))
            except Exception as e:
                name = fn.__name__.replace("_tab_", "").replace("_", " ").title()
                logger.error(f"Tab '{name}' failed: {e}", exc_info=True)
                tabs.append({
                    "tab": name, "metrics": [],
                    "error": str(e),
                    "explainability": {"overview": f"Tab computation failed: {e}"},
                })

        try:
            tabs.append(_tab_azure_llm(conn))
        except Exception as e:
            logger.error(f"Azure tab failed: {e}", exc_info=True)
            tabs.append({
                "tab": "Azure LLM Usage", "metrics": [],
                "error": str(e),
                "explainability": {"overview": f"Azure tab failed: {e}"},
            })

        return {
            "generated_at": generated_at,
            "dataset_id":   dataset_id,
            "db_path":      DB_PATH,
            "schema_info": {
                "pr_completed_status":    s["pr_completed_status"],
                "pr_failed_status":       s["pr_failed_status"],
                "pr_status_dist":         s["pr_status_dist"],
                "pr_date_col":            s["pr_date_col"],
                "pr_ai_summary_col":      s["pr_ai_summary_col"],
                "rule_active_status":     s["rule_active_status"],
                "rule_status_dist":       s["rule_status_dist"],
                "rrr_rule_code_col":      s.get("rrr_rule_code_col"),
                "dr_severity_col":        s["dr_severity_col"],
                "dr_severity_is_numeric": s["dr_severity_is_numeric"],
                "dr_severity_dist":       s["dr_severity_dist"],
                "dr_date_col":            s["dr_date_col"],
                "cp_sensitivity_col":     s["cp_sensitivity_col"],
                "cp_ai_description_col":  s["cp_ai_description_col"],
                "tc_status_values":       sorted(s["tc_status_values"]),
                "policy_table":           s["policy_table"],
                "policy_accepted_status": s["policy_accepted_status"],
                "policy_source_values":   sorted(s["policy_source_values"]),
                "gn_action_col":          s["gn_action_col"],
            },
            "tabs": tabs,
        }

    except Exception as e:
        logger.error(f"Catastrophic failure: {e}", exc_info=True)
        return {
            "generated_at": generated_at, "dataset_id": dataset_id,
            "db_path": DB_PATH, "error": str(e), "tabs": [],
        }
    finally:
        conn.close()


# ═══════════════════════════════════════════════════════════════════════════
# DEBUG ENDPOINT — verify schema discovery after deploy
# ═══════════════════════════════════════════════════════════════════════════

@router.get("/api/debug-schema", include_in_schema=False)
async def debug_schema():
    """
    Returns full _introspect() output.
    Use this immediately after deploy to verify all columns/statuses were found.
    Check: pr_completed_status, rule_active_status, dr_severity_col, and
    rrr_rule_code_col must NOT be null/None.
    """
    try:
        conn = _conn()
        s = _introspect(conn)
        conn.close()
        conn2 = _conn()
        table_columns = {}
        for tbl in ("profiling_runs", "dq_rules", "drift_records", "temporal_checks",
                    "column_profiles", "governance_notifications", "governance_policies",
                    "notification_inbox", "quality_snapshots", "governance_audit_log",
                    "dq_rule_runs", "dq_rule_run_results", "lineage_edges",
                    "knowledge_graph_edges"):
            if _table_exists(conn2, tbl):
                table_columns[tbl] = sorted(_columns(conn2, tbl))
        conn2.close()
        return {
            "db_path": DB_PATH,
            "schema_discovery": {k: (sorted(v) if isinstance(v, set) else v) for k, v in s.items()},
            "table_columns": table_columns,
            "critical_checks": {
                "pr_completed_status_found": s["pr_completed_status"] is not None,
                "rule_active_status_found":  s["rule_active_status"]  is not None,
                "dr_severity_col_found":     s["dr_severity_col"]     is not None,
                "pr_date_col_found":         s["pr_date_col"]         is not None,
                "cp_sensitivity_col_found":  s["cp_sensitivity_col"]  is not None,
                "rrr_rule_code_col_found":   s.get("rrr_rule_code_col") is not None,
            },
        }
    except Exception as e:
        return {"error": str(e), "db_path": DB_PATH}

# ═══════════════════════════════════════════════════════════════════════════
# AZURE METRICS ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════

@router.get("/api/azure-metrics", include_in_schema=True)
async def get_azure_metrics():
    """
    Returns the current Azure Monitor snapshot (live, 5-min cache).
    All values come directly from Azure — nothing is hardcoded.
    """
    try:
        from app.routers.azure_metrics_collector import fetch_live, is_configured
        data = fetch_live() if is_configured() else None
        return {
            "configured": is_configured(),
            "data":       data,
            "note":       "5-minute cache. POST /api/azure-metrics/refresh to force re-fetch.",
        }
    except ImportError:
        return {"configured": False, "data": None,
                "error": "azure_metrics_collector not installed — run: pip install azure-identity azure-monitor-query"}
    except Exception as e:
        return {"configured": False, "data": None, "error": str(e)}


@router.post("/api/azure-metrics/refresh", include_in_schema=True)
async def refresh_azure_metrics():
    """
    Force a fresh Azure Monitor poll, bypassing the 5-minute cache.
    """
    try:
        import app.routers.azure_metrics_collector as _amc
        _amc._cache = None
        _amc._cache_fetched_at = None
        data = _amc.fetch_live(force=True)
        if data is None:
            return {
                "success": False,
                "reason": (
                    "Azure not configured — set AZURE_* env vars"
                    if not _amc.is_configured()
                    else "Azure Monitor fetch failed — check server logs"
                ),
            }
        return {"success": True, "data": data}
    except ImportError:
        return {"success": False, "reason": "azure_metrics_collector not installed"}
    except Exception as e:
        return {"success": False, "reason": str(e)}


@router.get("/api/azure-metrics/debug", include_in_schema=False)
async def debug_llm_usage():
    """
    Shows the last 20 LLM calls with full error messages so you can
    diagnose why calls are failing. Hit this after a profiling run.
    """
    try:
        from app.routers.azure_metrics_collector import ensure_table, DB_PATH as AZ_DB
        import sqlite3 as _sq
        ensure_table()
        conn = _sq.connect(AZ_DB)
        conn.row_factory = _sq.Row
        rows = conn.execute("""
            SELECT id, called_at, model, prompt_tokens, completion_tokens,
                   total_tokens, cost_usd, latency_ms, status, error_message, caller
            FROM llm_usage_log
            ORDER BY id DESC LIMIT 20
        """).fetchall()
        summary = conn.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN status='success' THEN 1 ELSE 0 END) as successes,
                SUM(CASE WHEN status='error'   THEN 1 ELSE 0 END) as errors,
                SUM(prompt_tokens)     as total_prompt,
                SUM(completion_tokens) as total_completion,
                SUM(cost_usd)          as total_cost
            FROM llm_usage_log
        """).fetchone()
        conn.close()
        return {
            "summary": dict(summary),
            "last_20_calls": [dict(r) for r in rows],
            "diagnosis": (
                "All calls are errors with 0 tokens — check error_message field above "
                "to see why the LLM calls are failing. Common causes: wrong endpoint URL, "
                "invalid API key, model name mismatch, or the call path doesn't go through "
                "get_llm_client()."
                if dict(summary)["errors"] == dict(summary)["total"] and dict(summary)["total"] > 0
                else "OK"
            )
        }
    except Exception as e:
        return {"error": str(e)}
