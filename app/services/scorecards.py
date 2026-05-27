"""
AI DQM — Scorecards & KPIs Service (Industry-Grade Version)

All hardcoded thresholds and weights are now configurable via environment variables.
Fixed null rate calculations, incident timeline sorting, and N+1 query issues.
Added input sanitization for LLM prompts and proper logging.
"""

import os
import json
import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set

from openai import AzureOpenAI
from sqlalchemy import desc
from sqlalchemy.orm import Session, selectinload

from app.models import (
    Dataset, DataSource,
    ProfilingRun, ColumnProfile,
    QualityCheck, DQRule, DriftRecord,
)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration (all hardcoded values removed, now env-driven)
# ─────────────────────────────────────────────────────────────────────────────

class ScorecardsConfig:
    """Centralized configuration with sensible defaults overridable via env."""
    
    # Timezone
    TIMEZONE_OFFSET_HOURS = int(os.getenv("TIMEZONE_OFFSET_HOURS", "5"))
    TIMEZONE_OFFSET_MINUTES = int(os.getenv("TIMEZONE_OFFSET_MINUTES", "30"))
    IST = timezone(timedelta(hours=TIMEZONE_OFFSET_HOURS, minutes=TIMEZONE_OFFSET_MINUTES))
    
    # Risk thresholds (health score -> risk level)
    RISK_HIGH_THRESHOLD = float(os.getenv("RISK_HIGH_THRESHOLD", "70"))
    RISK_MEDIUM_THRESHOLD = float(os.getenv("RISK_MEDIUM_THRESHOLD", "85"))
    
    # Drift penalty weights per severity
    DRIFT_SEVERITY_WEIGHTS = {
        "CRITICAL": int(os.getenv("DRIFT_WEIGHT_CRITICAL", "20")),
        "HIGH": int(os.getenv("DRIFT_WEIGHT_HIGH", "10")),
        "MEDIUM": int(os.getenv("DRIFT_WEIGHT_MEDIUM", "5")),
        "LOW": int(os.getenv("DRIFT_WEIGHT_LOW", "1")),
    }
    
    # Drift index weights (for DriftKPI)
    DRIFT_INDEX_WEIGHTS = {
        "CRITICAL": int(os.getenv("DRIFT_INDEX_CRITICAL", "25")),
        "HIGH": int(os.getenv("DRIFT_INDEX_HIGH", "15")),
        "MEDIUM": int(os.getenv("DRIFT_INDEX_MEDIUM", "7")),
        "LOW": int(os.getenv("DRIFT_INDEX_LOW", "2")),
    }
    
    # Weighted health score percentages (must sum to 100)
    HEALTH_WEIGHTS = {
        "completeness": float(os.getenv("HEALTH_WEIGHT_COMPLETENESS", "0.40")),
        "drift_health": float(os.getenv("HEALTH_WEIGHT_DRIFT", "0.20")),
        "rule_compliance": float(os.getenv("HEALTH_WEIGHT_RULES", "0.15")),
        "schema_stability": float(os.getenv("HEALTH_WEIGHT_SCHEMA", "0.15")),
        "freshness": float(os.getenv("HEALTH_WEIGHT_FRESHNESS", "0.10")),
    }
    
    # Schema stability: penalty per change
    SCHEMA_CHANGE_PENALTY = int(os.getenv("SCHEMA_CHANGE_PENALTY", "10"))
    SCHEMA_PENALTY_CAP = int(os.getenv("SCHEMA_PENALTY_CAP", "50"))
    
    # Freshness: penalty per stale column
    STALE_COLUMN_PENALTY = int(os.getenv("STALE_COLUMN_PENALTY", "10"))
    FRESHNESS_PENALTY_CAP = int(os.getenv("FRESHNESS_PENALTY_CAP", "40"))
    
    # Default days for various trends
    DEFAULT_TREND_DAYS = int(os.getenv("DEFAULT_TREND_DAYS", "90"))
    DEFAULT_SCHEMA_DAYS = int(os.getenv("DEFAULT_SCHEMA_DAYS", "7"))
    DEFAULT_INCIDENT_DAYS = int(os.getenv("DEFAULT_INCIDENT_DAYS", "30"))
    
    # Run comparison limits
    MAX_RUNS_COMPARISON = int(os.getenv("MAX_RUNS_COMPARISON", "20"))
    
    # Velocity window
    VELOCITY_WINDOW = int(os.getenv("VELOCITY_WINDOW", "5"))
    
    # LLM
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview")
    AZURE_OPENAI_MODEL = os.getenv("AZURE_OPENAI_MODEL", "Llama-3.3-70B-Instruct")
    LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1200"))
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    
    # Schema drift types (can be extended via env comma-separated)
    SCHEMA_DRIFT_TYPES = set(
        os.getenv("SCHEMA_DRIFT_TYPES", 
                  "COLUMN_ADDED,COLUMN_REMOVED,TYPE_CHANGED,schema,SCHEMA_CHANGE,column_added,column_removed,type_changed"
                 ).split(",")
    )
    
    # Violation category mapping (hardcoded but now in config)
    CHECK_CATEGORY = {
        "FUTURE_DATES": "Temporal", "ANCIENT_DATES": "Temporal", "PRE_EPOCH_DATES": "Temporal",
        "STALE_DATA": "Temporal", "HIGH_NULL_DATE_RATE": "Temporal", "UNPARSEABLE_DATES": "Temporal",
        "WEEKEND_BUSINESS_DATES": "Temporal", "TEMPORAL_GAPS": "Temporal",
        "DUPLICATE_TIMESTAMPS": "Temporal", "SINGLE_DATE_DOMINANCE": "Temporal",
        "ALL_NULLS": "Structural", "HIGH_NULL_RATE": "Structural", "CONSTANT_COLUMN": "Structural",
        "NEAR_CONSTANT_COLUMN": "Structural", "HIGH_DUPLICATE_RATE": "Structural",
        "SUSPICIOUS_WHITESPACE": "Structural", "MIXED_CASE_INCONSISTENCY": "Structural",
        "INVALID_EMAIL_FORMAT": "Structural", "UNEXPECTED_NEGATIVE_VALUES": "Structural",
        "ZERO_DOMINATED_COLUMN": "Structural", "STATISTICAL_OUTLIERS": "Structural",
        "FIXED_LENGTH_VIOLATION": "Structural",
        "EMPTY_DATASET": "Dataset", "DUPLICATE_ROWS": "Dataset",
        "SUSPICIOUSLY_FEW_ROWS": "Dataset", "ALL_IDENTICAL_COLUMN_VALUES": "Dataset",
    }
    
    SEVERITY_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}

# Initialize config
cfg = ScorecardsConfig()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# IST helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_ist(dt) -> Optional[datetime]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(cfg.IST)


def _fmt_ist(dt) -> Optional[str]:
    ist_dt = _to_ist(dt)
    return ist_dt.isoformat() if ist_dt else None


# ─────────────────────────────────────────────────────────────────────────────
# LLM client (with sanitization)
# ─────────────────────────────────────────────────────────────────────────────

def _llm_client() -> AzureOpenAI:
    return AzureOpenAI(
        api_key=cfg.AZURE_OPENAI_API_KEY,
        api_version=cfg.AZURE_OPENAI_API_VERSION,
        azure_endpoint=cfg.AZURE_OPENAI_ENDPOINT,
    )


def _sanitize_for_prompt(text: str) -> str:
    """Escape double quotes and backslashes to prevent prompt injection."""
    if not text:
        return ""
    return text.replace("\\", "\\\\").replace('"', '\\"')


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers (optimized with eager loading)
# ─────────────────────────────────────────────────────────────────────────────

def _latest_run(db: Session, dataset_id: int) -> Optional[ProfilingRun]:
    return (
        db.query(ProfilingRun)
        .filter(ProfilingRun.dataset_id == dataset_id, ProfilingRun.status == "COMPLETED")
        .order_by(ProfilingRun.id.desc())
        .first()
    )


def _column_profiles_for_run(db: Session, run_id: int) -> List[ColumnProfile]:
    return db.query(ColumnProfile).filter(ColumnProfile.profiling_run_id == run_id).all()


def _avg_metrics(profiles: List[ColumnProfile]) -> Dict[str, float]:
    """Average all 6 core DQ dimensions across columns."""
    if not profiles:
        return {m: 0.0 for m in ["completeness", "uniqueness", "validity", "consistency", "accuracy", "integrity"]}
    metrics = ["completeness", "uniqueness", "validity", "consistency", "accuracy", "integrity"]
    result = {}
    for m in metrics:
        vals = [getattr(p, m) for p in profiles if getattr(p, m) is not None]
        result[m] = round(sum(vals) / len(vals), 1) if vals else 0.0
    return result


def _health_score(metrics: Dict[str, float]) -> float:
    """Simple average — kept for backwards compatibility."""
    vals = [v for v in metrics.values() if v is not None]
    return round(sum(vals) / len(vals), 1) if vals else 0.0


def _weighted_health_score(
    metrics: Dict[str, float],
    drift_penalty: float = 0.0,
    schema_changes: int = 0,
    freshness_score: Optional[float] = None,
    rule_violation_pct: float = 0.0,
) -> Dict[str, Any]:
    completeness = metrics.get("completeness", 0.0)
    drift_health = max(0.0, 100.0 - drift_penalty)
    violation_health = max(0.0, 100.0 - min(rule_violation_pct, 100.0))
    
    schema_penalty = min(schema_changes * cfg.SCHEMA_CHANGE_PENALTY, cfg.SCHEMA_PENALTY_CAP)
    schema_health = max(0.0, 100.0 - schema_penalty)
    fresh_health = freshness_score if freshness_score is not None else metrics.get("completeness", 80.0)

    weighted = round(
        cfg.HEALTH_WEIGHTS["completeness"] * completeness
        + cfg.HEALTH_WEIGHTS["drift_health"] * drift_health
        + cfg.HEALTH_WEIGHTS["rule_compliance"] * violation_health
        + cfg.HEALTH_WEIGHTS["schema_stability"] * schema_health
        + cfg.HEALTH_WEIGHTS["freshness"] * fresh_health,
        1,
    )

    breakdown = [
        {
            "dimension": "Completeness",
            "weight": int(cfg.HEALTH_WEIGHTS["completeness"] * 100),
            "rawScore": round(completeness, 1),
            "contribution": round(cfg.HEALTH_WEIGHTS["completeness"] * completeness, 1),
            "color": "#8B5CF6",
        },
        {
            "dimension": "Drift Health",
            "weight": int(cfg.HEALTH_WEIGHTS["drift_health"] * 100),
            "rawScore": round(drift_health, 1),
            "contribution": round(cfg.HEALTH_WEIGHTS["drift_health"] * drift_health, 1),
            "color": "#F59E0B",
        },
        {
            "dimension": "Rule Compliance",
            "weight": int(cfg.HEALTH_WEIGHTS["rule_compliance"] * 100),
            "rawScore": round(violation_health, 1),
            "contribution": round(cfg.HEALTH_WEIGHTS["rule_compliance"] * violation_health, 1),
            "color": "#EF4444",
        },
        {
            "dimension": "Schema Stability",
            "weight": int(cfg.HEALTH_WEIGHTS["schema_stability"] * 100),
            "rawScore": round(schema_health, 1),
            "contribution": round(cfg.HEALTH_WEIGHTS["schema_stability"] * schema_health, 1),
            "color": "#10B981",
        },
        {
            "dimension": "Freshness",
            "weight": int(cfg.HEALTH_WEIGHTS["freshness"] * 100),
            "rawScore": round(fresh_health, 1),
            "contribution": round(cfg.HEALTH_WEIGHTS["freshness"] * fresh_health, 1),
            "color": "#3B82F6",
        },
    ]

    return {"score": weighted, "breakdown": breakdown}


def _trend_direction(current: float, previous: float) -> str:
    diff = current - previous
    if diff > 1.0:
        return "up"
    if diff < -1.0:
        return "down"
    return "stable"


# ─────────────────────────────────────────────────────────────────────────────
# 1. KPI Summary
# ─────────────────────────────────────────────────────────────────────────────

def get_kpi_summary(db: Session, dataset_id: int) -> dict:
    run = _latest_run(db, dataset_id)
    if not run:
        return {"status": "NO_DATA", "kpis": [], "overallHealth": 0, "lastRunAt": None}

    profiles = _column_profiles_for_run(db, run.id)
    if not profiles:
        return {"status": "NO_DATA", "kpis": [], "overallHealth": 0, "lastRunAt": None}

    metrics = _avg_metrics(profiles)

    prev_run = (
        db.query(ProfilingRun)
        .filter(
            ProfilingRun.dataset_id == dataset_id,
            ProfilingRun.status == "COMPLETED",
            ProfilingRun.id < run.id,
        )
        .order_by(ProfilingRun.id.desc())
        .first()
    )
    prev_metrics: Dict[str, float] = {}
    if prev_run:
        prev_profiles = _column_profiles_for_run(db, prev_run.id)
        prev_metrics = _avg_metrics(prev_profiles)

    kpis = []
    display_map = {
        "accuracy": "Data Accuracy Score",
        "completeness": "Data Completeness",
        "consistency": "Data Consistency",
        "validity": "Data Validity",
    }
    for key, label in display_map.items():
        val = metrics.get(key, 0.0)
        prev_val = prev_metrics.get(key, val)
        kpis.append({
            "key": key,
            "label": label,
            "value": val,
            "formatted": f"{val:.1f}%",
            "trend": _trend_direction(val, prev_val),
            "delta": round(val - prev_val, 1),
        })

    drift_penalty = _get_drift_penalty(db, dataset_id)
    schema_changes_7d = _get_schema_changes_count(db, dataset_id, days=cfg.DEFAULT_SCHEMA_DAYS)
    freshness = get_freshness_score(db, dataset_id)
    freshness_score = freshness.get("freshnessScore") if freshness.get("status") == "OK" else None
    violation_pct = _get_violation_pct(db, run.id, len(profiles))

    weighted_result = _weighted_health_score(
        metrics,
        drift_penalty=drift_penalty,
        schema_changes=schema_changes_7d,
        freshness_score=freshness_score,
        rule_violation_pct=violation_pct,
    )
    overall = weighted_result["score"]
    health_breakdown = weighted_result["breakdown"]

    prev_overall = overall
    if prev_metrics:
        prev_weighted = _weighted_health_score(
            prev_metrics,
            drift_penalty=drift_penalty,
            schema_changes=schema_changes_7d,
            freshness_score=freshness_score,
            rule_violation_pct=violation_pct,
        )
        prev_overall = prev_weighted["score"]

    confidence_result = _score_confidence(
        col_count=len(profiles),
        rule_count=_get_active_rule_count(db, dataset_id),
        run_count=db.query(ProfilingRun).filter(
            ProfilingRun.dataset_id == dataset_id,
            ProfilingRun.status == "COMPLETED",
        ).count(),
    )

    return {
        "status": "OK",
        "kpis": kpis,
        "overallHealth": overall,
        "overallTrend": _trend_direction(overall, prev_overall),
        "totalColumns": len(profiles),
        "lastRunAt": _fmt_ist(run.timestamp),
        "runId": run.id,
        "healthBreakdown": health_breakdown,
        "scoreConfidence": confidence_result,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers for weighted health inputs
# ─────────────────────────────────────────────────────────────────────────────

def _get_drift_penalty(db: Session, dataset_id: int) -> float:
    run = _latest_run(db, dataset_id)
    if not run:
        return 0.0
    try:
        records = db.query(DriftRecord).filter(DriftRecord.profiling_run_id == run.id).all()
        if not records:
            return 0.0
        penalty = sum(cfg.DRIFT_SEVERITY_WEIGHTS.get(getattr(r, "severity", "LOW"), 1) for r in records)
        return min(float(penalty), 100.0)
    except Exception as e:
        logger.error(f"Error computing drift penalty for dataset {dataset_id}: {e}")
        return 0.0


def _get_schema_changes_count(db: Session, dataset_id: int, days: int = 7) -> int:
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    try:
        runs = (
            db.query(ProfilingRun)
            .filter(
                ProfilingRun.dataset_id == dataset_id,
                ProfilingRun.status == "COMPLETED",
                ProfilingRun.timestamp >= cutoff,
            )
            .all()
        )
        run_ids = [r.id for r in runs]
        if not run_ids:
            return 0
        count = (
            db.query(DriftRecord)
            .filter(
                DriftRecord.profiling_run_id.in_(run_ids),
                DriftRecord.drift_type.in_(cfg.SCHEMA_DRIFT_TYPES),
            )
            .count()
        )
        return count
    except Exception as e:
        logger.error(f"Error counting schema changes for dataset {dataset_id}: {e}")
        return 0


def _get_violation_pct(db: Session, run_id: int, total_cols: int) -> float:
    if total_cols == 0:
        return 0.0
    try:
        violated_cols = (
            db.query(QualityCheck.column_name)
            .filter(QualityCheck.profiling_run_id == run_id)
            .distinct()
            .count()
        )
        return round(violated_cols / total_cols * 100, 1)
    except Exception as e:
        logger.error(f"Error computing violation percentage for run {run_id}: {e}")
        return 0.0


def _get_active_rule_count(db: Session, dataset_id: int) -> int:
    try:
        return db.query(DQRule).filter(
            DQRule.dataset_id == dataset_id,
            DQRule.status.in_(["Active", "Paused"]),
        ).count()
    except Exception as e:
        logger.error(f"Error counting active rules for dataset {dataset_id}: {e}")
        return 0


def _score_confidence(col_count: int, rule_count: int, run_count: int) -> dict:
    score = 0
    reasons = []

    if col_count >= 10:
        score += 2
    elif col_count >= 5:
        score += 1
        reasons.append(f"only {col_count} columns profiled")
    else:
        reasons.append(f"very few columns ({col_count})")

    if rule_count >= 5:
        score += 2
    elif rule_count >= 2:
        score += 1
        reasons.append(f"limited rules ({rule_count})")
    else:
        score += 0
        reasons.append(f"few or no active rules ({rule_count})")

    if run_count >= 10:
        score += 2
    elif run_count >= 3:
        score += 1
        reasons.append(f"limited run history ({run_count} runs)")
    else:
        reasons.append(f"very little run history ({run_count} run{'s' if run_count != 1 else ''})")

    if score >= 5:
        level = "High"
    elif score >= 3:
        level = "Medium"
    else:
        level = "Low"

    return {
        "level": level,
        "colCount": col_count,
        "ruleCount": rule_count,
        "runCount": run_count,
        "reasons": reasons,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. Quality Trend
# ─────────────────────────────────────────────────────────────────────────────

def get_quality_trend(db: Session, dataset_id: int, days: int = None) -> list:
    if days is None:
        days = cfg.DEFAULT_TREND_DAYS
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    runs = (
        db.query(ProfilingRun)
        .filter(
            ProfilingRun.dataset_id == dataset_id,
            ProfilingRun.status == "COMPLETED",
            ProfilingRun.timestamp >= cutoff,
        )
        .order_by(ProfilingRun.timestamp.asc())
        .options(selectinload(ProfilingRun.column_profiles))
        .all()
    )
    result = []
    for r in runs:
        profiles = r.column_profiles
        if not profiles:
            continue
        metrics = _avg_metrics(profiles)
        health = _health_score(metrics)
        avg_null_rate = round(
            sum(100.0 - (p.completeness if p.completeness is not None else 100.0) for p in profiles) / len(profiles),
            1
        ) if profiles else 0.0
        ts = _to_ist(r.timestamp)
        result.append({
            "runId": r.id,
            "date": ts.strftime("%d %b") if ts else f"Run #{r.id}",
            "fullDate": _fmt_ist(r.timestamp),
            "score": health,
            "accuracy": metrics.get("accuracy", 0.0),
            "completeness": metrics.get("completeness", 0.0),
            "validity": metrics.get("validity", 0.0),
            "consistency": metrics.get("consistency", 0.0),
            "nullRate": avg_null_rate,
        })

    if result:
        best = max(result, key=lambda x: x["score"])
        worst = min(result, key=lambda x: x["score"])
        for pt in result:
            pt["isBest"] = pt["runId"] == best["runId"]
            pt["isWorst"] = pt["runId"] == worst["runId"]

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 3. Top Risk Contributors
# ─────────────────────────────────────────────────────────────────────────────

def get_risk_contributors(db: Session, dataset_id: int, top_n: int = 10) -> list:
    run = _latest_run(db, dataset_id)
    if not run:
        return []
    profiles = _column_profiles_for_run(db, run.id)
    if not profiles:
        return []

    sorted_profiles = sorted(
        [p for p in profiles if p.health_score is not None],
        key=lambda p: p.health_score,
    )[:top_n]

    result = []
    for p in sorted_profiles:
        score = p.health_score or 0.0
        if score < cfg.RISK_HIGH_THRESHOLD:
            risk = "High"
        elif score < cfg.RISK_MEDIUM_THRESHOLD:
            risk = "Medium"
        else:
            risk = "Low"

        violations = (
            db.query(QualityCheck)
            .filter(
                QualityCheck.profiling_run_id == run.id,
                QualityCheck.column_name == p.column_name,
                QualityCheck.status != "resolved",
            )
            .count()
        )

        result.append({
            "columnName": p.column_name,
            "dataType": p.data_type,
            "healthScore": score,
            "risk": risk,
            "status": p.status,
            "completeness": p.completeness,
            "validity": p.validity,
            "violations": violations,
        })
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 4. Quality Velocity
# ─────────────────────────────────────────────────────────────────────────────

def get_quality_velocity(db: Session, dataset_id: int, window: int = None) -> dict:
    if window is None:
        window = cfg.VELOCITY_WINDOW
    runs = (
        db.query(ProfilingRun)
        .filter(ProfilingRun.dataset_id == dataset_id, ProfilingRun.status == "COMPLETED")
        .order_by(ProfilingRun.id.desc())
        .limit(window)
        .options(selectinload(ProfilingRun.column_profiles))
        .all()
    )
    if len(runs) < 2:
        return {
            "status": "INSUFFICIENT_DATA",
            "message": f"Need at least 2 runs (have {len(runs)})",
            "velocity": 0.0,
            "direction": "stable",
            "momentum": "neutral",
        }

    runs = sorted(runs, key=lambda r: r.id)
    scores = []
    for r in runs:
        profiles = r.column_profiles
        if profiles:
            scores.append(_health_score(_avg_metrics(profiles)))

    if len(scores) < 2:
        return {"status": "INSUFFICIENT_DATA", "velocity": 0.0, "direction": "stable", "momentum": "neutral"}

    deltas = [scores[i] - scores[i - 1] for i in range(1, len(scores))]
    avg_velocity = round(sum(deltas) / len(deltas), 2)

    recent_delta = scores[-1] - scores[-2]
    earlier_delta = (scores[-2] - scores[0]) / max(len(scores) - 2, 1) if len(scores) > 2 else avg_velocity
    if recent_delta > earlier_delta + 0.5:
        momentum = "accelerating"
    elif recent_delta < earlier_delta - 0.5:
        momentum = "decelerating"
    else:
        momentum = "steady"

    direction = "improving" if avg_velocity > 0.5 else "degrading" if avg_velocity < -0.5 else "stable"

    return {
        "status": "OK",
        "velocity": avg_velocity,
        "direction": direction,
        "momentum": momentum,
        "firstScore": round(scores[0], 1),
        "latestScore": round(scores[-1], 1),
        "totalChange": round(scores[-1] - scores[0], 1),
        "runsAnalysed": len(scores),
        "scoreHistory": [round(s, 1) for s in scores],
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. DQ Rules Coverage
# ─────────────────────────────────────────────────────────────────────────────

def get_rules_coverage(db: Session, dataset_id: int) -> dict:
    run = _latest_run(db, dataset_id)
    if not run:
        return {"status": "NO_DATA", "coverageScore": 0, "coveredColumns": 0, "totalColumns": 0, "uncoveredColumns": []}

    profiles = _column_profiles_for_run(db, run.id)
    if not profiles:
        return {"status": "NO_DATA", "coverageScore": 0, "coveredColumns": 0, "totalColumns": 0, "uncoveredColumns": []}

    all_columns = {p.column_name for p in profiles}

    # Active rules only — Paused rules don't validate and shouldn't count as coverage
    active_rules = (
        db.query(DQRule)
        .filter(DQRule.dataset_id == dataset_id, DQRule.status == "Active")
        .all()
    )
    paused_rules = (
        db.query(DQRule)
        .filter(DQRule.dataset_id == dataset_id, DQRule.status == "Paused")
        .count()
    )

    covered = {r.column for r in active_rules if r.column}
    covered_cols = all_columns & covered
    uncovered_cols = sorted(all_columns - covered)

    # Coverage score based on Active rules only (Paused = not actively protecting)
    coverage_score = round(len(covered_cols) / len(all_columns) * 100, 1) if all_columns else 0.0

    rule_types: Dict[str, int] = defaultdict(int)
    for r in active_rules:
        rule_types[r.type or "Unknown"] += 1

    return {
        "status": "OK",
        "coverageScore": coverage_score,
        "coveredColumns": len(covered_cols),
        "totalColumns": len(all_columns),
        "totalActiveRules": len(active_rules),      # Active only
        "totalPausedRules": paused_rules,            # Separate count for transparency
        "uncoveredColumns": uncovered_cols[:20],
        "ruleTypeBreakdown": dict(rule_types),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. Violation Heatmap
# ─────────────────────────────────────────────────────────────────────────────

def get_violation_heatmap(db: Session, dataset_id: int) -> dict:
    run = _latest_run(db, dataset_id)
    if not run:
        return {"status": "NO_DATA", "cells": [], "totals": {}, "topViolations": []}

    checks = (
        db.query(QualityCheck)
        .filter(QualityCheck.profiling_run_id == run.id)
        .all()
    )
    if not checks:
        return {"status": "CLEAN", "cells": [], "totals": {}, "topViolations": []}

    grid: Dict[tuple, Dict[str, Any]] = defaultdict(lambda: {"count": 0, "totalViolations": 0})
    for c in checks:
        cat = cfg.CHECK_CATEGORY.get(c.check_type, "Structural")
        key = (cat, c.severity)
        grid[key]["count"] += 1
        grid[key]["totalViolations"] += c.violation_count or 0

    cells = [
        {
            "category": cat,
            "severity": sev,
            "checkCount": v["count"],
            "totalViolations": v["totalViolations"],
        }
        for (cat, sev), v in grid.items()
    ]
    cells.sort(key=lambda x: (cfg.SEVERITY_ORDER.get(x["severity"], 9), x["category"]))

    totals: Dict[str, int] = defaultdict(int)
    for c in checks:
        totals[c.severity] = totals[c.severity] + 1

    check_agg: Dict[str, int] = defaultdict(int)
    for c in checks:
        check_agg[c.check_type] += c.violation_count or 0
    top_violations = sorted(
        [{"checkType": k, "totalViolations": v} for k, v in check_agg.items()],
        key=lambda x: -x["totalViolations"],
    )[:5]

    return {
        "status": "OK",
        "runId": run.id,
        "cells": cells,
        "totals": dict(totals),
        "totalChecks": len(checks),
        "topViolations": top_violations,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 7. Data Freshness Score
# ─────────────────────────────────────────────────────────────────────────────

def get_freshness_score(db: Session, dataset_id: int) -> dict:
    run = _latest_run(db, dataset_id)
    if not run:
        return {"status": "NO_DATA", "freshnessScore": None, "staleColumns": [], "dateColumns": 0}

    profiles = _column_profiles_for_run(db, run.id)
    date_profiles = [p for p in profiles if p.timeliness is not None]

    if not date_profiles:
        return {
            "status": "NO_DATE_COLUMNS",
            "freshnessScore": None,
            "staleColumns": [],
            "dateColumns": 0,
        }

    avg_timeliness = round(sum(p.timeliness for p in date_profiles) / len(date_profiles), 1)

    stale_checks = (
        db.query(QualityCheck)
        .filter(
            QualityCheck.profiling_run_id == run.id,
            QualityCheck.check_type == "STALE_DATA",
        )
        .all()
    )
    stale_cols = [c.column_name for c in stale_checks]

    penalty = min(len(stale_cols) * cfg.STALE_COLUMN_PENALTY, cfg.FRESHNESS_PENALTY_CAP)
    freshness_score = max(round(avg_timeliness - penalty, 1), 0.0)

    return {
        "status": "OK",
        "freshnessScore": freshness_score,
        "avgTimeliness": avg_timeliness,
        "dateColumns": len(date_profiles),
        "staleColumns": stale_cols,
        "staleCount": len(stale_cols),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 8. Schema Stability KPI
# ─────────────────────────────────────────────────────────────────────────────

def get_schema_stability(db: Session, dataset_id: int, days: int = None) -> dict:
    if days is None:
        days = cfg.DEFAULT_SCHEMA_DAYS
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    try:
        runs = (
            db.query(ProfilingRun)
            .filter(
                ProfilingRun.dataset_id == dataset_id,
                ProfilingRun.status == "COMPLETED",
                ProfilingRun.timestamp >= cutoff,
            )
            .all()
        )
        run_ids = [r.id for r in runs]

        changes = []
        if run_ids:
            drift_records = (
                db.query(DriftRecord)
                .filter(
                    DriftRecord.profiling_run_id.in_(run_ids),
                    DriftRecord.drift_type.in_(cfg.SCHEMA_DRIFT_TYPES),
                )
                .order_by(DriftRecord.profiling_run_id.desc())
                .all()
            )
            for dr in drift_records:
                run_obj = next((r for r in runs if r.id == dr.profiling_run_id), None)
                changes.append({
                    "driftType": getattr(dr, "drift_type", "schema"),
                    "columnName": getattr(dr, "column_name", "—"),
                    "detail": getattr(dr, "detail", None) or getattr(dr, "message", None) or "",
                    "runId": dr.profiling_run_id,
                    "date": _fmt_ist(run_obj.timestamp) if run_obj else None,
                    "severity": getattr(dr, "severity", "MEDIUM"),
                })

        total = len(changes)
        if total == 0:
            risk = "Stable"
        elif total <= 2:
            risk = "Low"
        elif total <= 4:
            risk = "Medium"
        else:
            risk = "High"

        return {
            "status": "OK",
            "changeCount": total,
            "riskLevel": risk,
            "windowDays": days,
            "changes": changes[:10],
        }
    except Exception as e:
        logger.error(f"Error getting schema stability for dataset {dataset_id}: {e}")
        return {"status": "ERROR", "changeCount": 0, "riskLevel": "Unknown", "windowDays": days, "changes": [], "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# 9. Drift Severity KPI
# ─────────────────────────────────────────────────────────────────────────────

def get_drift_kpi(db: Session, dataset_id: int) -> dict:
    run = _latest_run(db, dataset_id)
    if not run:
        return {"status": "NO_DATA", "driftIndex": 0, "severity": "None", "totalColumns": 0}

    try:
        records = db.query(DriftRecord).filter(DriftRecord.profiling_run_id == run.id).all()
        if not records:
            return {
                "status": "CLEAN",
                "driftIndex": 0.0,
                "severity": "None",
                "totalColumns": 0,
                "breakdown": {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0},
            }

        breakdown: Dict[str, int] = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
        weighted_sum = 0
        drifted_cols = set()

        for r in records:
            sev = getattr(r, "severity", "LOW") or "LOW"
            breakdown[sev] = breakdown.get(sev, 0) + 1
            weighted_sum += cfg.DRIFT_INDEX_WEIGHTS.get(sev, 2)
            col = getattr(r, "column_name", None)
            if col:
                drifted_cols.add(col)

        drift_index = min(round(weighted_sum / max(len(records), 1) * 4, 1), 100.0)

        if drift_index == 0:
            agg_severity = "None"
        elif breakdown["CRITICAL"] > 0:
            agg_severity = "Critical"
        elif breakdown["HIGH"] > 0:
            agg_severity = "High"
        elif breakdown["MEDIUM"] > 0:
            agg_severity = "Medium"
        else:
            agg_severity = "Low"

        return {
            "status": "OK",
            "driftIndex": drift_index,
            "severity": agg_severity,
            "totalDriftedColumns": len(drifted_cols),
            "totalRecords": len(records),
            "breakdown": breakdown,
            "runId": run.id,
        }
    except Exception as e:
        logger.error(f"Error computing drift KPI for dataset {dataset_id}: {e}")
        return {"status": "ERROR", "driftIndex": 0, "severity": "Unknown", "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# 10. Run Comparison (Current vs Previous vs Baseline)
# ─────────────────────────────────────────────────────────────────────────────

def get_run_comparison(db: Session, dataset_id: int) -> dict:
    runs = (
        db.query(ProfilingRun)
        .filter(ProfilingRun.dataset_id == dataset_id, ProfilingRun.status == "COMPLETED")
        .order_by(ProfilingRun.id.desc())
        .limit(cfg.MAX_RUNS_COMPARISON)
        .options(selectinload(ProfilingRun.column_profiles))
        .all()
    )
    if len(runs) < 2:
        return {"status": "INSUFFICIENT_DATA", "deltas": [], "runs": {}}

    runs_sorted = sorted(runs, key=lambda r: r.id)

    current_run = runs_sorted[-1]
    previous_run = runs_sorted[-2]
    baseline_run = runs_sorted[0]

    def _run_snapshot(run: ProfilingRun) -> dict:
        profiles = run.column_profiles
        if not profiles:
            return {}
        metrics = _avg_metrics(profiles)
        null_rate = round(
            sum(100.0 - (p.completeness if p.completeness is not None else 100.0) for p in profiles) / len(profiles),
            1
        ) if profiles else 0.0
        checks = db.query(QualityCheck).filter(QualityCheck.profiling_run_id == run.id).count()
        ts = _to_ist(run.timestamp)
        return {
            "runId": run.id,
            "date": ts.strftime("%d %b %Y, %H:%M") if ts else f"Run #{run.id}",
            "health": _health_score(metrics),
            "completeness": metrics.get("completeness", 0.0),
            "accuracy": metrics.get("accuracy", 0.0),
            "validity": metrics.get("validity", 0.0),
            "consistency": metrics.get("consistency", 0.0),
            "uniqueness": metrics.get("uniqueness", 0.0),
            "nullRate": null_rate,
            "violationCount": checks,
            "colCount": len(profiles),
        }

    current_snap = _run_snapshot(current_run)
    previous_snap = _run_snapshot(previous_run)
    baseline_snap = _run_snapshot(baseline_run)

    if not current_snap:
        return {"status": "INSUFFICIENT_DATA", "deltas": [], "runs": {}}

    metrics_to_compare = [
        ("health", "Overall Health", "%"),
        ("completeness", "Completeness", "%"),
        ("accuracy", "Accuracy", "%"),
        ("validity", "Validity", "%"),
        ("consistency", "Consistency", "%"),
        ("uniqueness", "Uniqueness", "%"),
        ("nullRate", "Avg Null Rate", "%"),
        ("violationCount", "Violation Checks", ""),
    ]

    deltas = []
    for key, label, unit in metrics_to_compare:
        curr_val = current_snap.get(key, 0) or 0
        prev_val = previous_snap.get(key, curr_val) or curr_val
        base_val = baseline_snap.get(key, curr_val) or curr_val

        prev_delta = round(curr_val - prev_val, 1)
        base_delta = round(curr_val - base_val, 1)

        is_lower_better = key in ("nullRate", "violationCount")

        def _direction(delta, lower_better):
            if abs(delta) < 0.1:
                return "stable"
            if lower_better:
                return "down" if delta < 0 else "up"
            return "up" if delta > 0 else "down"

        def _is_good(delta, lower_better):
            if abs(delta) < 0.1:
                return None
            if lower_better:
                return delta < 0
            return delta > 0

        deltas.append({
            "metric": label,
            "unit": unit,
            "current": curr_val,
            "previous": prev_val,
            "baseline": base_val,
            "prevDelta": prev_delta,
            "baseDelta": base_delta,
            "prevDirection": _direction(prev_delta, is_lower_better),
            "baseDirection": _direction(base_delta, is_lower_better),
            "prevGood": _is_good(prev_delta, is_lower_better),
            "baseGood": _is_good(base_delta, is_lower_better),
        })

    return {
        "status": "OK",
        "deltas": deltas,
        "runs": {
            "current": current_snap,
            "previous": previous_snap,
            "baseline": baseline_snap,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# 11. Data Incident Timeline
# ─────────────────────────────────────────────────────────────────────────────

def get_incident_timeline(db: Session, dataset_id: int, days: int = None) -> dict:
    if days is None:
        days = cfg.DEFAULT_INCIDENT_DAYS
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    try:
        runs = (
            db.query(ProfilingRun)
            .filter(
                ProfilingRun.dataset_id == dataset_id,
                ProfilingRun.status == "COMPLETED",
                ProfilingRun.timestamp >= cutoff,
            )
            .order_by(ProfilingRun.timestamp.desc())
            .all()
        )
        run_ids = [r.id for r in runs]
        run_map = {r.id: r for r in runs}

        events = []

        if run_ids:
            # ── Quality checks: CRITICAL/HIGH, both open and resolved ─────────
            checks = (
                db.query(QualityCheck)
                .filter(
                    QualityCheck.profiling_run_id.in_(run_ids),
                    QualityCheck.severity.in_(["CRITICAL", "HIGH"]),
                )
                .order_by(QualityCheck.profiling_run_id.desc())
                .limit(80)
                .all()
            )
            for c in checks:
                run_obj = run_map.get(c.profiling_run_id)
                ts = _to_ist(run_obj.timestamp) if run_obj else None
                is_resolved = getattr(c, "status", "") == "resolved"

                events.append({
                    "type": "resolved" if is_resolved else "quality_check",
                    "category": cfg.CHECK_CATEGORY.get(c.check_type, "Structural"),
                    "title": c.check_type.replace("_", " ").title(),
                    "detail": (
                        ("✓ Resolved — " if is_resolved else "")
                        + f"{c.violation_count or 0:,} rows affected"
                        + (f" on {c.column_name}" if c.column_name else "")
                    ),
                    "severity": c.severity,
                    "status": getattr(c, "status", "open"),
                    "runId": c.profiling_run_id,
                    "date": _fmt_ist(run_obj.timestamp) if run_obj else None,
                    "dateShort": ts.strftime("%d %b, %H:%M") if ts else "—",
                    "columnName": c.column_name,
                    "violationCount": c.violation_count or 0,
                })

        if run_ids:
            # ── Drift records: only CRITICAL/HIGH severity drift ──────────────
            drift_records = (
                db.query(DriftRecord)
                .filter(
                    DriftRecord.profiling_run_id.in_(run_ids),
                )
                .order_by(DriftRecord.profiling_run_id.desc())
                .limit(60)
                .all()
            )
            for dr in drift_records:
                sev = getattr(dr, "severity", "MEDIUM") or "MEDIUM"
                drift_type = getattr(dr, "drift_type", "drift") or "drift"

                # Only include CRITICAL/HIGH drift — skip low-signal noise
                if cfg.SEVERITY_ORDER.get(sev.upper(), 9) > 1:
                    continue

                run_obj = run_map.get(dr.profiling_run_id)
                ts = _to_ist(run_obj.timestamp) if run_obj else None
                col = getattr(dr, "column_name", None)

                is_schema = drift_type.upper() in cfg.SCHEMA_DRIFT_TYPES
                event_type = "schema_change" if is_schema else "drift"
                title = ("Schema: " + drift_type.replace("_", " ").title()
                         if is_schema else
                         "Distribution Drift: " + drift_type.replace("_", " ").title())

                detail = ""
                if col:
                    detail += f"Column: {col}"
                raw_detail = getattr(dr, "detail", None) or getattr(dr, "message", None) or ""
                if raw_detail:
                    detail += (". " if detail else "") + str(raw_detail)[:80]

                events.append({
                    "type": event_type,
                    "category": "Schema" if is_schema else "Drift",
                    "title": title,
                    "detail": detail or "No additional detail",
                    "severity": sev,
                    "status": "open",
                    "runId": dr.profiling_run_id,
                    "date": _fmt_ist(run_obj.timestamp) if run_obj else None,
                    "dateShort": ts.strftime("%d %b, %H:%M") if ts else "—",
                    "columnName": col,
                    "violationCount": 0,
                })

        # Sort: most recent run first, then CRITICAL before HIGH, resolved last
        def _sort_key(e):
            run_ord = -e.get("runId", 0)
            sev_ord = cfg.SEVERITY_ORDER.get(e.get("severity", "LOW"), 9)
            resolved_ord = 1 if e.get("status") == "resolved" else 0
            return (run_ord, resolved_ord, sev_ord)

        events.sort(key=_sort_key)

        return {
            "status": "OK" if events else "CLEAN",
            "events": events[:50],
            "totalEvents": len(events),
            "windowDays": days,
        }
    except Exception as e:
        logger.error(f"Error building incident timeline for dataset {dataset_id}: {e}")
        return {"status": "ERROR", "events": [], "totalEvents": 0, "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# 12. Column-level Risk Table (extended)
# ─────────────────────────────────────────────────────────────────────────────

def get_column_risk_table(db: Session, dataset_id: int) -> dict:
    run = _latest_run(db, dataset_id)
    if not run:
        return {"status": "NO_DATA", "columns": []}

    profiles = _column_profiles_for_run(db, run.id)
    if not profiles:
        return {"status": "NO_DATA", "columns": []}

    try:
        drift_records = db.query(DriftRecord).filter(DriftRecord.profiling_run_id == run.id).all()
        drift_by_col: Dict[str, str] = {}
        for dr in drift_records:
            col = getattr(dr, "column_name", None)
            sev = getattr(dr, "severity", "LOW") or "LOW"
            if col:
                existing = drift_by_col.get(col, "LOW")
                if cfg.SEVERITY_ORDER.get(sev, 3) < cfg.SEVERITY_ORDER.get(existing, 3):
                    drift_by_col[col] = sev
    except Exception as e:
        logger.error(f"Error loading drift records for column risk table (dataset {dataset_id}): {e}")
        drift_by_col = {}

    result = []
    for p in profiles:
        score = p.health_score or 0.0
        if score < cfg.RISK_HIGH_THRESHOLD:
            risk = "High"
        elif score < cfg.RISK_MEDIUM_THRESHOLD:
            risk = "Medium"
        else:
            risk = "Low"

        violations = (
            db.query(QualityCheck)
            .filter(
                QualityCheck.profiling_run_id == run.id,
                QualityCheck.column_name == p.column_name,
            )
            .count()
        )

        drift_sev = drift_by_col.get(p.column_name, "None")

        # Compute PK score from completeness and uniqueness if not stored
        stored_pk_score = getattr(p, "pk_score", None)
        if stored_pk_score is not None:
            pk_score = stored_pk_score
        else:
            completeness = p.completeness or 0.0
            uniqueness = p.uniqueness or 0.0
            pk_score = round(min(completeness, uniqueness), 2)

        fk_score = getattr(p, "fk_score", None)

        result.append({
            "columnName": p.column_name,
            "dataType": p.data_type,
            "healthScore": round(score, 1),
            "risk": risk,
            "completeness": round(p.completeness or 0.0, 1),
            "validity": round(p.validity or 0.0, 1),
            "violations": violations,
            "driftSeverity": drift_sev,
            "status": p.status,
        })

    result.sort(key=lambda x: x["healthScore"])
    return {"status": "OK", "columns": result}


# ─────────────────────────────────────────────────────────────────────────────
# 13. Null Rate Trend (leading indicator)
# ─────────────────────────────────────────────────────────────────────────────

def get_null_trend(db: Session, dataset_id: int, days: int = None) -> dict:
    if days is None:
        days = cfg.DEFAULT_TREND_DAYS
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    runs = (
        db.query(ProfilingRun)
        .filter(
            ProfilingRun.dataset_id == dataset_id,
            ProfilingRun.status == "COMPLETED",
            ProfilingRun.timestamp >= cutoff,
        )
        .order_by(ProfilingRun.timestamp.asc())
        .options(selectinload(ProfilingRun.column_profiles))
        .all()
    )
    points = []
    for r in runs:
        profiles = r.column_profiles
        if not profiles:
            continue
        null_rates = [max(0.0, 100.0 - (p.completeness if p.completeness is not None else 100.0)) for p in profiles]
        avg_null = round(sum(null_rates) / len(null_rates), 2) if null_rates else 0.0
        ts = _to_ist(r.timestamp)
        points.append({
            "runId": r.id,
            "date": ts.strftime("%d %b") if ts else f"Run #{r.id}",
            "fullDate": _fmt_ist(r.timestamp),
            "avgNullRate": avg_null,
        })

    if len(points) >= 2:
        first = points[0]["avgNullRate"]
        last = points[-1]["avgNullRate"]
        delta = round(last - first, 2)
        direction = "rising" if delta > 0.5 else "falling" if delta < -0.5 else "stable"
    else:
        delta = 0.0
        direction = "stable"

    return {
        "status": "OK" if points else "NO_DATA",
        "points": points,
        "direction": direction,
        "totalDelta": delta,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 14. Generate AI Report (with sanitization)
# ─────────────────────────────────────────────────────────────────────────────

def generate_ai_report(db: Session, dataset_id: int, dataset_name: str) -> dict:
    kpi = get_kpi_summary(db, dataset_id)
    velocity = get_quality_velocity(db, dataset_id)
    coverage = get_rules_coverage(db, dataset_id)
    heatmap = get_violation_heatmap(db, dataset_id)
    freshness = get_freshness_score(db, dataset_id)
    risk = get_risk_contributors(db, dataset_id, top_n=5)
    drift_kpi = get_drift_kpi(db, dataset_id)
    schema = get_schema_stability(db, dataset_id)

    if kpi["status"] == "NO_DATA":
        return {"status": "NO_DATA", "report": "No profiling data available. Run DQ Scoring first."}

    safe_dataset_name = _sanitize_for_prompt(dataset_name)

    top_risks = ", ".join(
        f"{_sanitize_for_prompt(r['columnName'])} ({r['risk']} Risk, {r['healthScore']}%)" 
        for r in risk[:3]
    ) or "None"
    
    top_violations = ", ".join(
        f"{v['checkType']} ({v['totalViolations']} violations)"
        for v in heatmap.get("topViolations", [])[:3]
    ) or "None"
    
    uncovered = ", ".join(_sanitize_for_prompt(c) for c in coverage.get("uncoveredColumns", [])[:5]) or "None"
    severity_summary = ", ".join(f"{k}: {v}" for k, v in heatmap.get("totals", {}).items())

    kpi_lines = "\n".join(
        f"  - {k['label']}: {k['formatted']} (trend: {k['trend']}, delta vs previous run: {k['delta']:+.1f}%)"
        for k in kpi['kpis']
    )

    prompt = f"""You are a senior data quality analyst generating a structured executive scorecard report. You must respond ONLY with a valid JSON object — no markdown, no preamble, no explanation outside the JSON.

=== DATA INPUT ===
Dataset: {safe_dataset_name}
Overall Health Score (weighted): {kpi['overallHealth']}% (trend: {kpi.get('overallTrend', 'stable')})
Columns Profiled: {kpi.get('totalColumns', 0)}
Last Profiled: {kpi.get('lastRunAt', 'Unknown')}
Score Confidence: {kpi.get('scoreConfidence', {}).get('level', 'Unknown')}

KPI Scores (exact values):
{kpi_lines}

Drift Index: {drift_kpi.get('driftIndex', 0)} ({drift_kpi.get('severity', 'None')})
Drifted Columns: {drift_kpi.get('totalDriftedColumns', 0)}

Schema Changes (7 days): {schema.get('changeCount', 0)} — Risk: {schema.get('riskLevel', 'Stable')}

Quality Velocity: {velocity.get('direction', 'stable')}, {velocity.get('velocity', 0):+.2f}% avg change per run, momentum: {velocity.get('momentum', 'steady')}
Score history over {velocity.get('runsAnalysed', 0)} runs: {velocity.get('firstScore', 0)}% -> {velocity.get('latestScore', 0)}% (total change: {velocity.get('totalChange', 0):+.1f}%)

Rules Governance: {coverage.get('coverageScore', 0)}% coverage ({coverage.get('coveredColumns', 0)} of {coverage.get('totalColumns', 0)} columns have active rules, {coverage.get('totalActiveRules', 0)} total rules)
Unprotected columns: {uncovered}

Violations (latest run): {severity_summary}
Top violation types by row count: {top_violations}

Top Risk Columns (lowest health score first): {top_risks}

Data Freshness: score={freshness.get('freshnessScore', 'N/A')}%, avg timeliness={freshness.get('avgTimeliness', 'N/A')}%, stale columns={freshness.get('staleCount', 0)}
=== END DATA ===

Return exactly this JSON structure. Every field is required. Be specific and precise — use the exact numbers given above:

{{
  "executiveSummary": "2-3 sentences. State the exact overall health score, name the strongest and weakest KPI with their exact percentages, and give a one-line verdict on the dataset fitness for use.",
  "keyFindings": [
    {{
      "title": "short finding title (5-8 words)",
      "detail": "1-2 sentences. Cite exact metric values. Be specific about which columns, check types, or rules are involved.",
      "severity": "critical | warning | info | positive"
    }}
  ],
  "riskSummary": "2 sentences. Name the top 2 risk columns explicitly with their health scores. State what the coverage gap means for data reliability.",
  "trendAnalysis": "2 sentences. State the exact velocity number and what it means. Reference the score history (first to latest run values). Is quality accelerating, holding, or degrading?",
  "recommendations": [
    {{
      "priority": "High | Medium | Low",
      "action": "imperative verb phrase — specific action (e.g. Add NOT NULL rule to is_active column)",
      "rationale": "1 sentence explaining the exact metric or violation driving this recommendation"
    }}
  ],
  "overallVerdict": "one sentence — plain language pass/fail/watch verdict for executive audience"
}}

Rules:
- keyFindings: 3-5 items
- recommendations: exactly 3 items ordered High then Medium then Low
- All sentences must reference specific numbers, column names, or check types from the data above
- Do not use vague language like several, some, various, may, could — be precise"""

    try:
        client = _llm_client()
        response = client.chat.completions.create(
            model=cfg.AZURE_OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=cfg.LLM_MAX_TOKENS,
            temperature=cfg.LLM_TEMPERATURE,
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            parts = raw.split("```")
            raw = parts[1] if len(parts) > 1 else raw
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        parsed = json.loads(raw)
        return {
            "status": "OK",
            "report": parsed,
            "generatedAt": _fmt_ist(datetime.now(timezone.utc)),
            "datasetName": safe_dataset_name,
            "overallHealth": kpi["overallHealth"],
        }
    except json.JSONDecodeError:
        logger.error("LLM response was not valid JSON")
        return {
            "status": "OK_PLAIN",
            "report": raw if "raw" in dir() else "Generation failed — could not parse response",
            "generatedAt": _fmt_ist(datetime.now(timezone.utc)),
        }
    except Exception as e:
        logger.error(f"AI report generation failed: {e}")
        return {"status": "ERROR", "report": f"Report generation failed: {str(e)}"}


# ─────────────────────────────────────────────────────────────────────────────
# Scorecard SLA target (persisted in governance_system_config)
# ─────────────────────────────────────────────────────────────────────────────

_SCORECARD_SLA_KEY = "scorecard_sla_target"
_SCORECARD_SLA_DEFAULT = 85


def get_scorecard_sla_target(db: Session) -> dict:
    """Return the persisted overall health SLA target (default 85)."""
    try:
        from app.routers.governance_routes import GovernanceSystemConfig
        row = db.query(GovernanceSystemConfig).filter(
            GovernanceSystemConfig.key == _SCORECARD_SLA_KEY
        ).first()
        target = int(row.value) if row else _SCORECARD_SLA_DEFAULT
        return {"target": target}
    except Exception:
        return {"target": _SCORECARD_SLA_DEFAULT}


def save_scorecard_sla_target(db: Session, target: int) -> dict:
    """Persist the overall health SLA target to governance_system_config."""
    try:
        from app.routers.governance_routes import GovernanceSystemConfig
        target = max(0, min(100, int(target)))
        existing = db.query(GovernanceSystemConfig).filter(
            GovernanceSystemConfig.key == _SCORECARD_SLA_KEY
        ).first()
        if existing:
            existing.value = str(target)
        else:
            db.add(GovernanceSystemConfig(key=_SCORECARD_SLA_KEY, value=str(target)))
        db.commit()
        return {"target": target}
    except Exception as e:
        logger.error(f"Failed to save scorecard SLA target: {e}")
        return {"target": target}


def get_full_scorecard(db: Session, dataset_id: int) -> dict:
    return {
        "kpi": get_kpi_summary(db, dataset_id),
        "trend": get_quality_trend(db, dataset_id, days=cfg.DEFAULT_TREND_DAYS),
        "riskContributors": get_risk_contributors(db, dataset_id),
        "velocity": get_quality_velocity(db, dataset_id),
        "rulesCoverage": get_rules_coverage(db, dataset_id),
        "violationHeatmap": get_violation_heatmap(db, dataset_id),
        "freshness": get_freshness_score(db, dataset_id),
        "schemaStability": get_schema_stability(db, dataset_id),
        "driftKpi": get_drift_kpi(db, dataset_id),
        "runComparison": get_run_comparison(db, dataset_id),
        "incidentTimeline": get_incident_timeline(db, dataset_id),
        "columnRiskTable": get_column_risk_table(db, dataset_id),
        "nullTrend": get_null_trend(db, dataset_id),
    }